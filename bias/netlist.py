"""Structured circuit representation.

This is the layer that makes "describe a circuit and get one" possible without
the usual failure mode. Asking a language model for raw SPICE produces text
that looks right and does not simulate: floating nodes, a MOSFET with three
terminals, a supply shorted to ground, a device referencing a net that appears
nowhere else. None of that is visible until ngspice rejects it, and by then the
error message is about line 14 rather than about the circuit.

So models do not write SPICE here. They emit a structured Circuit -- devices,
terminals, nets -- which is validated against the rules below before a netlist
is ever generated. A circuit that violates them is rejected with a message
describing the electrical problem, which is something a model can actually
repair.

The same representation is what lets circuits be compared (graph_hash) and
transformed programmatically, rather than by editing strings.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum


class Kind(str, Enum):
    """Device types, with their terminal names fixed by construction."""

    NMOS = "nmos"
    PMOS = "pmos"
    RES = "res"
    CAP = "cap"
    IND = "ind"
    VSOURCE = "vsource"
    ISOURCE = "isource"
    VCVS = "vcvs"


# Terminal order matters: it is the order SPICE expects on the device line.
TERMINALS: dict[Kind, tuple[str, ...]] = {
    Kind.NMOS: ("d", "g", "s", "b"),
    Kind.PMOS: ("d", "g", "s", "b"),
    Kind.RES: ("p", "n"),
    Kind.CAP: ("p", "n"),
    Kind.IND: ("p", "n"),
    Kind.VSOURCE: ("p", "n"),
    Kind.ISOURCE: ("p", "n"),
    Kind.VCVS: ("p", "n", "cp", "cn"),
}

# SPICE instance-name prefixes.
PREFIX: dict[Kind, str] = {
    Kind.NMOS: "M",
    Kind.PMOS: "M",
    Kind.RES: "R",
    Kind.CAP: "C",
    Kind.IND: "L",
    Kind.VSOURCE: "V",
    Kind.ISOURCE: "I",
    Kind.VCVS: "E",
}

MOS_KINDS = (Kind.NMOS, Kind.PMOS)

GROUND = "0"


@dataclass
class Device:
    """One circuit element and the nets its terminals attach to."""

    name: str
    kind: Kind
    nets: dict[str, str]
    params: dict[str, float | str] = field(default_factory=dict)

    def terminal_names(self) -> tuple[str, ...]:
        return TERMINALS[self.kind]

    def net_of(self, terminal: str) -> str:
        return self.nets[terminal]

    @property
    def instance(self) -> str:
        """SPICE instance name, with the type prefix applied if absent."""
        want = PREFIX[self.kind]
        return self.name if self.name.upper().startswith(want) else want + self.name


@dataclass
class Problem:
    """One validation failure, phrased so a model can act on it."""

    severity: str  # "error" or "warning"
    device: str | None
    message: str

    def __str__(self) -> str:
        where = f"{self.device}: " if self.device else ""
        return f"[{self.severity}] {where}{self.message}"


@dataclass
class Circuit:
    """A netlist as a graph, not as text."""

    name: str
    devices: list[Device] = field(default_factory=list)
    # Nets that connect to the outside world (supplies, inputs, outputs). These
    # are exempt from the floating-node rule.
    ports: list[str] = field(default_factory=list)
    comment: str = ""

    # -- structure ------------------------------------------------------

    def nets(self) -> set[str]:
        out: set[str] = set()
        for d in self.devices:
            out.update(d.nets.values())
        return out

    def device(self, name: str) -> Device:
        for d in self.devices:
            if d.name == name or d.instance == name:
                return d
        raise KeyError(f"no device named {name!r}")

    def connections(self) -> dict[str, list[tuple[str, str]]]:
        """net -> [(device name, terminal), ...]"""
        out: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for d in self.devices:
            for term, net in d.nets.items():
                out[net].append((d.name, term))
        return dict(out)

    def transistors(self) -> list[Device]:
        return [d for d in self.devices if d.kind in MOS_KINDS]

    # -- validation -----------------------------------------------------

    def validate(self, strict: bool = True) -> list[Problem]:
        """Structural and electrical checks that catch a broken circuit before
        the simulator does.

        These are the failures that actually occur when a circuit is generated
        rather than typed by hand. Each message names the electrical problem so
        it can be repaired without reading SPICE.
        """
        problems: list[Problem] = []
        seen_names: set[str] = set()

        for d in self.devices:
            # 1. Wrong terminal count is the most common generated error --
            #    a MOSFET written with three terminals, no bulk.
            expected = set(TERMINALS[d.kind])
            got = set(d.nets)
            if got != expected:
                missing = sorted(expected - got)
                extra = sorted(got - expected)
                bits = []
                if missing:
                    bits.append(f"missing terminal(s) {missing}")
                if extra:
                    bits.append(f"unknown terminal(s) {extra}")
                problems.append(
                    Problem("error", d.name,
                            f"{d.kind.value} needs terminals "
                            f"{list(TERMINALS[d.kind])}; " + ", ".join(bits))
                )

            # 2. Duplicate instance names silently overwrite in SPICE.
            if d.name in seen_names:
                problems.append(
                    Problem("error", d.name, "duplicate device name")
                )
            seen_names.add(d.name)

            # 3. A transistor with no geometry cannot be sized or simulated.
            if d.kind in MOS_KINDS:
                for p in ("W", "L"):
                    if p not in d.params:
                        problems.append(
                            Problem("error", d.name,
                                    f"transistor has no {p}; every MOSFET needs "
                                    "a width and a length")
                        )

            # 4. Shorted device: both ends on the same net does nothing.
            if d.kind in (Kind.RES, Kind.CAP, Kind.IND, Kind.VSOURCE, Kind.ISOURCE):
                if d.nets.get("p") == d.nets.get("n"):
                    problems.append(
                        Problem("error", d.name,
                                f"both terminals tie to net "
                                f"{d.nets.get('p')!r}; the device is shorted out")
                    )

            # 5. A transistor with drain tied to source carries no signal.
            if d.kind in MOS_KINDS and d.nets.get("d") == d.nets.get("s"):
                problems.append(
                    Problem("error", d.name,
                            f"drain and source are both net {d.nets.get('d')!r}; "
                            "the transistor cannot conduct a signal")
                )

        # 6. Floating nets. A net touched by exactly one terminal has nowhere
        #    for current to go, and ngspice reports it as a singular matrix.
        conns = self.connections()
        exempt = set(self.ports) | {GROUND}
        for net, attached in sorted(conns.items()):
            if net in exempt:
                continue
            if len(attached) < 2:
                who = ", ".join(f"{d}.{t}" for d, t in attached)
                problems.append(
                    Problem("error", None,
                            f"net {net!r} connects only to {who}; it is floating. "
                            "Every internal node needs at least two connections")
                )

        # 7. Ground must exist, or node voltages have no reference.
        if GROUND not in self.nets():
            problems.append(
                Problem("error", None,
                        f"no ground: no device connects to net {GROUND!r}")
            )

        # 8. Islands. A net with no conductive path to ground floats as a
        #    group even when each node individually looks connected.
        unreachable = self.nets() - self._reachable_from_ground()
        for net in sorted(unreachable):
            problems.append(
                Problem("error", None,
                        f"net {net!r} has no path to ground through any device")
            )

        if not strict:
            problems = [p for p in problems if p.severity == "error"]
        return problems

    def _reachable_from_ground(self) -> set[str]:
        """Nets connected to ground through any device, treating every device
        as conductive between all of its terminals."""
        adjacency: dict[str, set[str]] = defaultdict(set)
        for d in self.devices:
            nets = list(d.nets.values())
            for a in nets:
                for b in nets:
                    if a != b:
                        adjacency[a].add(b)

        seen = {GROUND}
        stack = [GROUND]
        while stack:
            net = stack.pop()
            for nxt in adjacency.get(net, ()):
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        return seen

    def is_valid(self) -> bool:
        return not any(p.severity == "error" for p in self.validate())

    def explain_problems(self) -> str:
        return "\n".join(str(p) for p in self.validate()) or "no problems found"

    # -- emission -------------------------------------------------------

    def to_spice(self, models: dict[str, str] | None = None) -> str:
        """Render device lines. `models` maps Kind values to model card names,
        so the same circuit emits against any PDK."""
        models = models or {}
        lines: list[str] = []
        if self.comment:
            lines.append(f"* {self.comment}")

        for d in self.devices:
            nets = " ".join(d.nets[t] for t in TERMINALS[d.kind])

            if d.kind in MOS_KINDS:
                model = models.get(d.kind.value, d.kind.value)
                params = " ".join(
                    f"{k}={_num(v)}" for k, v in d.params.items()
                )
                lines.append(f"{d.instance} {nets} {model} {params}".rstrip())

            elif d.kind == Kind.VCVS:
                gain = d.params.get("gain", 1.0)
                lines.append(f"{d.instance} {nets} {_num(gain)}")

            elif d.kind in (Kind.VSOURCE, Kind.ISOURCE):
                spec = d.params.get("spec")
                if spec is None:
                    spec = f"DC {_num(d.params.get('dc', 0.0))}"
                lines.append(f"{d.instance} {nets} {spec}")

            else:  # RES, CAP, IND
                value = d.params.get("value", 0.0)
                lines.append(f"{d.instance} {nets} {_num(value)}")

        return "\n".join(lines)

    # -- identity -------------------------------------------------------

    def graph_hash(self, iterations: int = 3) -> str:
        """A canonical fingerprint of the circuit's structure.

        Two circuits hash the same when they are the same topology under any
        renaming of internal nets and devices. Supply and ground nets are
        labelled distinctly because they are not interchangeable with signal
        nets -- swapping VDD and an internal node is a different circuit even
        though the graph shape is identical.

        Uses Weisfeiler-Lehman refinement over the bipartite device/net graph.
        """
        net_label: dict[str, str] = {}
        for net in self.nets():
            if net == GROUND:
                net_label[net] = "net:gnd"
            elif net in self.ports:
                # Ports keep their identity; they are externally meaningful.
                net_label[net] = f"net:port:{net}"
            else:
                net_label[net] = "net:internal"

        dev_label: dict[str, str] = {
            d.name: f"dev:{d.kind.value}" for d in self.devices
        }

        for _ in range(iterations):
            new_dev: dict[str, str] = {}
            for d in self.devices:
                # A device is characterised by its own label plus the labels of
                # the nets on each named terminal.
                parts = [
                    f"{t}={net_label[d.nets[t]]}"
                    for t in TERMINALS[d.kind]
                    if t in d.nets
                ]
                new_dev[d.name] = _digest(dev_label[d.name] + "|" + "|".join(parts))

            new_net: dict[str, str] = {}
            conns = self.connections()
            for net in self.nets():
                # A net is characterised by the multiset of (device, terminal)
                # pairs touching it -- sorted, so device order does not matter.
                parts = sorted(
                    f"{new_dev[dev]}@{term}" for dev, term in conns.get(net, [])
                )
                new_net[net] = _digest(net_label[net] + "|" + "|".join(parts))

            dev_label, net_label = new_dev, new_net

        return _digest("|".join(sorted(dev_label.values())))

    def summary(self) -> str:
        counts: dict[str, int] = defaultdict(int)
        for d in self.devices:
            counts[d.kind.value] += 1
        parts = ", ".join(f"{n} {k}" for k, n in sorted(counts.items()))
        return f"{self.name}: {parts}; {len(self.nets())} nets"


def _digest(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()[:16]


def _num(v: float | str) -> str:
    if isinstance(v, str):
        return v
    return f"{v:g}"
