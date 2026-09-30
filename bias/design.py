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
    "half adder": "half_adder", "halfadder": "half_adder", "half add": "half_adder",
    "full adder": "full_adder", "fulladder": "full_adder", "full add": "full_adder",
    "xor": "xor2", "exclusive or": "xor2", "xor gate": "xor2",
    "nand": "nand2", "nand gate": "nand2",
    "nor": "nor2", "nor gate": "nor2",
    "inverter": "inverter", "not gate": "inverter", "inv": "inverter",
    "adder": "full_adder",
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

        netlist = circuit.to_spice({"nmos": pdk.nmos, "pmos": pdk.pmos})
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
