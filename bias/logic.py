"""Transistor-level CMOS logic, built correct by construction.

"Build me a half adder" should produce a circuit that works, every time. That
rules out having a language model write the transistors: it will produce
something plausible that fails the truth table in one corner of the input
space, and nothing will notice.

So logic is synthesised deterministically here. Gates are built from static
CMOS primitives whose transistor networks are fixed and known correct; modules
are built by wiring gates. The model's job upstream is only to decide *what* to
build -- the structure comes from this file, and the truth table is then proved
by transient simulation rather than assumed.

Sizing follows the usual convention: PMOS wider than NMOS by `beta` to
compensate for lower hole mobility, and series stacks widened by their depth so
pull-down strength stays roughly constant.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product

from .netlist import Circuit, Device, Kind

VDD = "vdd"
GND = "0"


@dataclass
class Sizing:
    """Transistor geometry for synthesised logic."""

    wn: float = 1.0e-6
    ln: float = 0.18e-6
    # PMOS/NMOS width ratio. ~2-3 for most bulk CMOS.
    beta: float = 2.5

    @property
    def wp(self) -> float:
        return self.wn * self.beta


@dataclass
class Builder:
    """Accumulates devices and hands out unique names."""

    sizing: Sizing = field(default_factory=Sizing)
    devices: list[Device] = field(default_factory=list)
    _counts: dict[str, int] = field(default_factory=dict)
    _nets: int = 0

    def uid(self, stem: str) -> str:
        n = self._counts.get(stem, 0) + 1
        self._counts[stem] = n
        return f"{stem}{n}"

    def net(self, stem: str = "n") -> str:
        self._nets += 1
        return f"{stem}_{self._nets}"

    # -- primitives -----------------------------------------------------

    def nmos(self, d: str, g: str, s: str, w: float, l: float) -> None:
        self.devices.append(
            Device(self.uid("Mn"), Kind.NMOS,
                   {"d": d, "g": g, "s": s, "b": GND},
                   {"W": w, "L": l})
        )

    def pmos(self, d: str, g: str, s: str, w: float, l: float) -> None:
        self.devices.append(
            Device(self.uid("Mp"), Kind.PMOS,
                   {"d": d, "g": g, "s": s, "b": VDD},
                   {"W": w, "L": l})
        )

    # -- gates ----------------------------------------------------------

    def inverter(self, a: str, y: str) -> str:
        """Y = NOT A. Two transistors."""
        s = self.sizing
        self.nmos(y, a, GND, s.wn, s.ln)
        self.pmos(y, a, VDD, s.wp, s.ln)
        return y

    def nand(self, inputs: list[str], y: str) -> str:
        """Y = NOT(AND of inputs).

        NMOS in series (all inputs high pulls the output down), PMOS in
        parallel. The series stack is widened by its depth so the pull-down
        drive matches an inverter's.
        """
        s = self.sizing
        n = len(inputs)
        wn = s.wn * n

        # Series NMOS chain from y down to ground.
        node = y
        for i, a in enumerate(inputs):
            nxt = GND if i == n - 1 else self.net("ns")
            self.nmos(node, a, nxt, wn, s.ln)
            node = nxt

        # Parallel PMOS, each from VDD to y.
        for a in inputs:
            self.pmos(y, a, VDD, s.wp, s.ln)
        return y

    def nor(self, inputs: list[str], y: str) -> str:
        """Y = NOT(OR of inputs).

        The dual of NAND: NMOS in parallel, PMOS in series. The PMOS stack is
        widened by depth, which is why wide NORs are avoided in practice.
        """
        s = self.sizing
        n = len(inputs)
        wp = s.wp * n

        for a in inputs:
            self.nmos(y, a, GND, s.wn, s.ln)

        node = y
        for i, a in enumerate(inputs):
            nxt = VDD if i == n - 1 else self.net("ps")
            self.pmos(node, a, nxt, wp, s.ln)
            node = nxt
        return y

    # -- composites -----------------------------------------------------

    def and_gate(self, inputs: list[str], y: str) -> str:
        """AND = NAND followed by an inverter.

        Static CMOS gates are inherently inverting, so a non-inverting function
        always costs an extra stage. This is not an implementation detail to
        optimise away -- it is why NAND is the cheap primitive.
        """
        mid = self.net("nand")
        self.nand(inputs, mid)
        return self.inverter(mid, y)

    def or_gate(self, inputs: list[str], y: str) -> str:
        mid = self.net("nor")
        self.nor(inputs, mid)
        return self.inverter(mid, y)

    def xor2(self, a: str, b: str, y: str) -> str:
        """Y = A XOR B, as four NAND2 gates.

        Chosen over a transmission-gate XOR because every node is actively
        driven to a rail, so the truth table holds without relying on charge
        sharing or input drive strength.
        """
        n1 = self.net("x")
        n2 = self.net("x")
        n3 = self.net("x")
        self.nand([a, b], n1)
        self.nand([a, n1], n2)
        self.nand([b, n1], n3)
        self.nand([n2, n3], y)
        return y


# ---------------------------------------------------------------------------
# Modules
# ---------------------------------------------------------------------------


def half_adder(sizing: Sizing | None = None) -> Circuit:
    """A half adder: SUM = A XOR B, CARRY = A AND B.

    Shares work between the two outputs. The XOR's first NAND already computes
    NAND(A,B), and CARRY is just its inverse -- so the carry costs one inverter
    rather than a whole AND gate. 18 transistors.
    """
    b = Builder(sizing or Sizing())

    # NAND(a,b) is used by both outputs.
    nab = b.net("nab")
    b.nand(["a", "b"], nab)

    n2, n3 = b.net("x"), b.net("x")
    b.nand(["a", nab], n2)
    b.nand(["b", nab], n3)
    b.nand([n2, n3], "sum")

    b.inverter(nab, "carry")

    return Circuit(
        name="half_adder",
        devices=b.devices,
        ports=["a", "b", "sum", "carry", VDD],
        comment="half adder: sum = a XOR b, carry = a AND b (18T static CMOS)",
    )


def full_adder(sizing: Sizing | None = None) -> Circuit:
    """A full adder built from two half adders and an OR.

    SUM = A XOR B XOR CIN
    COUT = (A AND B) OR (CIN AND (A XOR B))
    """
    b = Builder(sizing or Sizing())

    ab = b.net("ab")
    b.xor2("a", "b", ab)

    b.xor2(ab, "cin", "sum")

    c1 = b.net("c")
    c2 = b.net("c")
    b.and_gate(["a", "b"], c1)
    b.and_gate([ab, "cin"], c2)
    b.or_gate([c1, c2], "cout")

    return Circuit(
        name="full_adder",
        devices=b.devices,
        ports=["a", "b", "cin", "sum", "cout", VDD],
        comment="full adder from two half adders and an OR",
    )


def inverter_cell(sizing: Sizing | None = None) -> Circuit:
    b = Builder(sizing or Sizing())
    b.inverter("a", "y")
    return Circuit("inverter", b.devices, ["a", "y", VDD], "CMOS inverter")


def nand2_cell(sizing: Sizing | None = None) -> Circuit:
    b = Builder(sizing or Sizing())
    b.nand(["a", "b"], "y")
    return Circuit("nand2", b.devices, ["a", "b", "y", VDD], "2-input NAND")


def nor2_cell(sizing: Sizing | None = None) -> Circuit:
    b = Builder(sizing or Sizing())
    b.nor(["a", "b"], "y")
    return Circuit("nor2", b.devices, ["a", "b", "y", VDD], "2-input NOR")


def xor2_cell(sizing: Sizing | None = None) -> Circuit:
    b = Builder(sizing or Sizing())
    b.xor2("a", "b", "y")
    return Circuit("xor2", b.devices, ["a", "b", "y", VDD], "2-input XOR (4x NAND2)")


# name -> (builder, input ports, output ports, reference function)
CELLS: dict[str, tuple] = {
    "inverter": (inverter_cell, ["a"], ["y"], lambda a: {"y": 1 - a}),
    "nand2": (nand2_cell, ["a", "b"], ["y"], lambda a, b: {"y": 1 - (a & b)}),
    "nor2": (nor2_cell, ["a", "b"], ["y"], lambda a, b: {"y": 1 - (a | b)}),
    "xor2": (xor2_cell, ["a", "b"], ["y"], lambda a, b: {"y": a ^ b}),
    "half_adder": (
        half_adder, ["a", "b"], ["sum", "carry"],
        lambda a, b: {"sum": a ^ b, "carry": a & b},
    ),
    "full_adder": (
        full_adder, ["a", "b", "cin"], ["sum", "cout"],
        lambda a, b, c: {"sum": a ^ b ^ c, "cout": 1 if (a + b + c) >= 2 else 0},
    ),
}


def get(name: str) -> tuple:
    if name not in CELLS:
        raise KeyError(f"unknown cell {name!r}; have {sorted(CELLS)}")
    return CELLS[name]


def truth_table(name: str) -> list[tuple[dict[str, int], dict[str, int]]]:
    """The reference truth table, generated from the golden function."""
    _, ins, _, fn = get(name)
    rows = []
    for combo in product([0, 1], repeat=len(ins)):
        rows.append((dict(zip(ins, combo)), fn(*combo)))
    return rows
