"""Natural language in, verified circuit out.

This is the entry point behind "build me a half adder". It resolves a request
to something the synthesiser can build, builds it, proves it with simulation,
and writes out the artifacts -- netlist, symbol, truth table.

Resolution is deterministic first and a language model only as a fallback.
That ordering is deliberate: for the circuits people actually ask for, a lookup
is exact, instant, free and cannot hallucinate. The model earns its place only
on requests the table does not cover, and even then it chooses a *cell name*
from a fixed menu rather than authoring a netlist -- so a wrong answer is a
wrong choice, not a broken circuit.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from . import logic, symbol, verify
from .netlist import Circuit
from .pdk import PDK

# Spelling variants map to a cell. Matched against the request with word
# boundaries after punctuation is stripped.
ALIASES: dict[str, str] = {
    # Adders and subtractors.
    "half adder": "half_adder", "halfadder": "half_adder", "half add": "half_adder",
    "full adder": "full_adder", "fulladder": "full_adder", "full add": "full_adder",
    "adder": "full_adder",
    "2 bit adder": "adder2", "two bit adder": "adder2",
    "2-bit adder": "adder2", "ripple carry": "adder2", "ripple carry adder": "adder2",
    "half subtractor": "half_subtractor", "half sub": "half_subtractor",
    "full subtractor": "full_subtractor", "full sub": "full_subtractor",
    "subtractor": "full_subtractor", "subtracter": "full_subtractor",

    # Basic gates.
    "xor": "xor2", "exclusive or": "xor2", "xor gate": "xor2",
    "xnor": "xnor2", "xnor gate": "xnor2", "equivalence": "xnor2",
    "nand": "nand2", "nand gate": "nand2",
    "3 input nand": "nand3", "three input nand": "nand3", "nand3": "nand3",
    "nor": "nor2", "nor gate": "nor2",
    "3 input nor": "nor3", "three input nor": "nor3", "nor3": "nor3",
    # No bare "and"/"or": they are ordinary English words that appear in
    # almost every analog request ("60dB gain and 10MHz bandwidth"), and an
    # alias table cannot tell the conjunction from the gate.
    "and gate": "and2", "2 input and": "and2", "and2": "and2",
    "or gate": "or2", "2 input or": "or2", "or2": "or2",
    "inverter": "inverter", "not gate": "inverter", "inv": "inverter",
    "buffer": "buffer", "buf": "buffer", "non inverting buffer": "buffer",

    # Blocks.
    "mux": "mux2", "multiplexer": "mux2", "2 to 1 mux": "mux2",
    "2:1 mux": "mux2", "selector": "mux2",
    "decoder": "decoder2to4", "2 to 4 decoder": "decoder2to4",
    "one hot": "decoder2to4", "one hot decoder": "decoder2to4",
    "comparator": "comparator1", "magnitude comparator": "comparator1",
    "compare": "comparator1",
    "majority": "majority3", "majority gate": "majority3", "voter": "majority3",
    "parity": "parity4", "parity checker": "parity4", "parity tree": "parity4",
}


@dataclass
class BuildResult:
    request: str
    cell: str
    circuit: Circuit
    report: verify.Report
    how: str = ""          # how the request was resolved
    files: dict[str, Path] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.report.passed

    def describe(self) -> str:
        lines = [
            f'request : "{self.request}"',
            f"resolved: {self.cell}  ({self.how})",
            f"built   : {self.circuit.summary()}",
            f"          {len(self.circuit.transistors())} transistors, "
            f"structurally valid",
        ]
        status = "VERIFIED" if self.ok else "FAILED"
        lines.append(f"verified: {status} against the truth table by transient "
                     f"simulation")
        if self.report.reason:
            lines.append(f"          {self.report.reason}")
        if self.report.rows:
            lines.append("")
            lines.append(self.report.table())
            lines.append("")
            lines.append(f"worst output rail error: "
                         f"{self.report.worst_rail_error * 100:.2f}% of VDD")
        if self.files:
            lines.append("")
            for what, path in self.files.items():
                lines.append(f"  {what:<9} {path}")
        return "\n".join(lines)


def resolve(request: str, use_llm: bool = True) -> tuple[str, str]:
    """Map a free-text request to a cell name.

    Returns (cell, how). Raises LookupError when nothing matches and no model
    is available to ask.
    """
    text = re.sub(r"[^a-z0-9 ]+", " ", request.lower())
    text = re.sub(r"\s+", " ", text).strip()

    # Exact cell name wins outright.
    for cell in logic.CELLS:
        if cell.replace("_", " ") in text or cell in text:
            return cell, "matched cell name"

    # Longest alias first, so "full adder" beats "adder".
    for alias in sorted(ALIASES, key=len, reverse=True):
        if re.search(rf"\b{re.escape(alias)}\b", text):
            return ALIASES[alias], f'matched "{alias}"'

    if use_llm and os.environ.get("ANTHROPIC_API_KEY"):
        cell = _ask_model(request)
        if cell:
            return cell, "chosen by model"

    raise LookupError(
        f"could not resolve {request!r} to a known cell.\n"
        f"Available: {', '.join(sorted(logic.CELLS))}"
    )


def _ask_model(request: str) -> str | None:
    """Let a model pick from the fixed menu. It never writes a netlist."""
    try:
        import anthropic
    except ImportError:
        return None

    menu = ", ".join(sorted(logic.CELLS))
    try:
        reply = anthropic.Anthropic().messages.create(
            model="claude-opus-4-5",
            max_tokens=64,
            system=(
                "You map a circuit request to exactly one cell name from a "
                "fixed list. Reply with the cell name alone and nothing else. "
                "If none of them is a reasonable match, reply NONE."
            ),
            messages=[{"role": "user",
                       "content": f"Available cells: {menu}\n\nRequest: {request}"}],
        )
        answer = "".join(b.text for b in reply.content if b.type == "text").strip()
    except Exception:
        return None

    answer = answer.strip().strip(".`").lower()
    return answer if answer in logic.CELLS else None


def build(
    request: str,
    pdk: PDK,
    *,
    outdir: Path | None = None,
    sizing: logic.Sizing | None = None,
    use_llm: bool = True,
) -> BuildResult:
    """Resolve, synthesise, verify, and optionally write artifacts."""
    cell, how = resolve(request, use_llm=use_llm)
    builder, inputs, outputs, _ = logic.get(cell)
    circuit = builder(sizing) if sizing else builder()

    report = verify.verify(cell, pdk)

    files: dict[str, Path] = {}
    if outdir:
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)

        sym = symbol.Symbol.from_circuit(
            circuit, inputs, outputs,
            subtitle=f"{len(circuit.transistors())} transistors, static CMOS",
        )
        files["symbol"] = _write(outdir / f"{cell}.svg", sym.render())

        netlist = circuit.to_spice(pdk=pdk)
        header = (f"* {cell} -- generated by bias\n"
                  f"* {circuit.comment}\n"
                  f"* pdk: {pdk.name}\n")
        files["netlist"] = _write(outdir / f"{cell}.sp", header + netlist + "\n")

        deck, _times = verify.build_deck(circuit, cell, pdk)
        files["testbench"] = _write(outdir / f"{cell}_tb.sp", deck + "\n")

        files["truth"] = _write(
            outdir / f"{cell}_truth.txt",
            report.table() + "\n",
        )

    return BuildResult(request, cell, circuit, report, how, files)


def _write(path: Path, text: str) -> Path:
    path.write_text(text)
    return path


# ---------------------------------------------------------------------------
# Analog: a spoken specification, sized against the simulator
# ---------------------------------------------------------------------------

# Where to start looking, not where to stop. Calibration on dev180 found the
# 5T OTA tops out near 49 dB once phase margin is held at 60 deg -- it buys
# gain with channel length, and length costs stability.
#
# That number is process-dependent: on sky130 the same topology reaches less,
# because real 130nm devices have worse output resistance. Rather than carry a
# per-PDK constant that silently goes stale, this is only the first guess --
# `size()` falls back to the two-stage topology when the single stage cannot
# meet the gain, so a wrong threshold costs simulations and not a wrong answer.
OTA5T_GAIN_CEILING_DB = 48.0
MILLER_GAIN_CEILING_DB = 80.0


@dataclass
class SizeResult:
    request: str
    topology_name: str
    parsed: "object"                 # specparse.ParsedSpec
    solved: bool
    sims_used: int
    sims_to_target: int | None
    values: dict[str, float]
    measured: dict[str, float]
    why_topology: str = ""
    infeasible: str = ""
    files: dict[str, Path] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.solved

    def describe(self) -> str:

        lines = [
            f'request : "{self.request}"',
            "targets :",
            self.parsed.describe(),
            f"topology: {self.topology_name}  ({self.why_topology})",
        ]
        if self.infeasible:
            lines += ["", f"NOT ATTEMPTED: {self.infeasible}"]
            return "\n".join(lines)

        lines.append("")
        if self.solved:
            lines.append(f"SIZED: all targets met after "
                         f"{self.sims_to_target} simulations "
                         f"({self.sims_used} used in total)")
        else:
            lines.append(f"NOT MET after {self.sims_used} simulations; "
                         f"closest result below")

        lines.append("")
        lines.append("sizing:")
        for k, v in self.values.items():
            lines.append(f"  {k:<8} {_fmt(v)}")

        lines.append("")
        lines.append("measured:")
        lines.append(self.parsed.spec.report(self.measured))

        if self.files:
            lines.append("")
            for what, path in self.files.items():
                lines.append(f"  {what:<9} {path}")
        return "\n".join(lines)


def choose_topology(gain_db: float | None) -> tuple[str, str]:
    """Pick the cheapest topology that can reach the requested gain.

    A single stage is preferred when it will do: fewer devices, no
    compensation network, and unconditional stability. The threshold is the
    measured 5T ceiling rather than a rule of thumb.
    """
    if gain_db is None:
        return "ota5t", "no gain target given; starting from the single stage"
    if gain_db <= OTA5T_GAIN_CEILING_DB:
        return "ota5t", (
            f"{gain_db:.0f}dB is within the single-stage ceiling "
            f"(~{OTA5T_GAIN_CEILING_DB:.0f}dB measured at 60deg phase margin)"
        )
    return "miller", (
        f"{gain_db:.0f}dB exceeds the single-stage ceiling "
        f"(~{OTA5T_GAIN_CEILING_DB:.0f}dB), so a two-stage amplifier is needed"
    )


def size(
    request: str,
    pdk: PDK,
    *,
    budget: int = 400,
    seed: int = 0,
    attempts: int = 3,
    optimizer: str = "de",
    outdir: Path | None = None,
    use_llm: bool = True,
) -> SizeResult:
    """Parse a spoken spec, choose a topology, and size it against ngspice."""
    import numpy as np

    from . import optimizers, plot, specparse, topology as topo_registry
    from .evaluate import Evaluator
    from .topology import Testbench

    parsed = specparse.parse(request)
    best_so_far = None
    gain_db = parsed.found.get("gain")
    topo_name, why = choose_topology(gain_db)
    topo = topo_registry.get(topo_name)

    # Refuse a request no topology here can reach, rather than burning a
    # budget discovering it. Being told "this needs a different topology" is
    # more useful than a near miss with no explanation.
    if gain_db is not None and gain_db > MILLER_GAIN_CEILING_DB:
        return SizeResult(
            request, topo_name, parsed, False, 0, None, {}, {}, why,
            infeasible=(
                f"{gain_db:.0f}dB is beyond what either topology here reaches "
                f"(~{MILLER_GAIN_CEILING_DB:.0f}dB for the two-stage). "
                "That needs gain boosting or a third stage."
            ),
        )

    tb = Testbench(cl=parsed.cl or 1e-12)
    name = "llm" if (optimizer in ("llm", "agent") and use_llm) else optimizer

    # Restart from a fresh seed rather than spending one long run. These
    # objectives have wide flat regions where a population collapses early,
    # and a second start is worth more than a longer first one. Successes
    # stop immediately, so the extra attempts only cost time on hard specs.
    ev = None
    total_sims = 0
    for attempt in range(max(1, attempts)):
        ev = Evaluator(topo, pdk, parsed.spec, tb, budget=budget,
                       stop_on_success=True)
        optimizers.get(name).run(ev, np.random.default_rng(seed + attempt))
        total_sims += ev.used
        if ev.solved():
            break
        if best_so_far is None or (ev.best and ev.best.score < best_so_far.score):
            best_so_far = ev.best

    best = ev.solved() or (
        min((x for x in (ev.best, best_so_far) if x), key=lambda e: e.score)
    )

    # The single stage was chosen on a threshold measured on another process.
    # If it could not reach the gain, that threshold was simply wrong here --
    # escalate to two stages rather than reporting a near miss.
    if (
        not ev.solved()
        and topo_name == "ota5t"
        and gain_db is not None
        and not best.measured.get("gain", 0) >= gain_db
    ):
        topo = topo_registry.get("miller")
        topo_name = "miller"
        why = (
            f"the single stage reached only "
            f"{best.measured.get('gain', float('nan')):.1f}dB against a "
            f"{gain_db:.0f}dB target on {pdk.name}, so a two-stage amplifier "
            "was tried instead"
        )
        fallback_best = best
        for attempt in range(max(1, attempts)):
            ev = Evaluator(topo, pdk, parsed.spec, tb, budget=budget,
                           stop_on_success=True)
            optimizers.get(name).run(ev, np.random.default_rng(seed + attempt))
            total_sims += ev.used
            if ev.solved():
                break
            if ev.best and ev.best.score < fallback_best.score:
                fallback_best = ev.best
        best = ev.solved() or fallback_best
    values = best.values if best else {}
    measured = best.measured if best else {}

    files: dict[str, Path] = {}
    if outdir and best:
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        stem = topo_name

        deck = topo.deck(pdk, values, tb)
        files["testbench"] = _write(outdir / f"{stem}_sized.sp", deck + "\n")

        sym = symbol.Symbol(
            name=stem,
            inputs=["vinp", "vinn"],
            outputs=["vout"],
            power=["vdd"],
            ground=["0"],
            subtitle=f"{len(topo.params)} sized parameters",
        )
        files["symbol"] = _write(outdir / f"{stem}.svg", sym.render())

        files["report"] = _write(
            outdir / f"{stem}_result.txt",
            parsed.spec.report(measured) + "\n",
        )

        bode = plot.bode(topo, pdk, values, tb, measured)
        if bode:
            files["bode"] = _write(outdir / f"{stem}_bode.svg", bode)

    return SizeResult(
        request, topo_name, parsed,
        solved=ev.solved() is not None,
        sims_used=total_sims,
        sims_to_target=ev.sims_to_target(),
        values=values,
        measured=measured,
        why_topology=why,
        files=files,
    )


def _fmt(v: float) -> str:
    from .spec import _eng
    return _eng(v)


# ---------------------------------------------------------------------------


def build_any(
    request: str,
    pdk: PDK,
    *,
    outdir: Path | None = None,
    use_llm: bool = True,
    **kwargs,
):
    """One entry point for both halves.

    Digital first: a named cell is an exact, instant match and cannot be
    confused for anything else. Only if no cell matches is the request read as
    an analog specification, which is the case that needs numbers and units.
    """
    from . import specparse

    # Analog is tested first, and the order matters. A specification carries
    # units -- dB, Hz, watts -- and no logic cell ever does, so a successful
    # parse is decisive. Resolving the cell name first instead lets a stray
    # English word win: "60dB gain and 10MHz bandwidth" contains "and", and an
    # alias table has no way to tell the conjunction from the gate. That is
    # not hypothetical; it shipped, and routed every op-amp request to an AND
    # gate until a container test caught it.
    try:
        specparse.parse(request)
    except ValueError:
        pass
    else:
        return size(
            request, pdk, outdir=outdir, use_llm=use_llm,
            **{k: v for k, v in kwargs.items() if k != "sizing"},
        )

    try:
        cell, _how = resolve(request, use_llm=use_llm)
    except LookupError as exc:
        raise LookupError(
            f"{exc}\n\nFor an amplifier, give targets with units -- "
            "for example: 60dB gain, 10MHz bandwidth, under 100uW"
        ) from exc

    return build(request, pdk, outdir=outdir,
                 sizing=kwargs.get("sizing"), use_llm=use_llm)
