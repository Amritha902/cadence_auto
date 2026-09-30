"""Proving a synthesised logic circuit actually works.

A netlist that passes structural validation can still be wrong: a mis-wired
gate, an inverted polarity, a stack that never pulls the output to a rail. The
only way to know is to run it.

This drives every input combination through the circuit as a transient
simulation and checks the measured output voltages against the reference truth
table. It also checks *how well* each output reaches its rail -- a gate that
settles at 60% of VDD is logically "1" by a threshold test but is a broken
design, and that distinction is what separates a real check from a rubber
stamp.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import logic
from .netlist import Circuit
from .pdk import PDK
from .sim import run_deck


@dataclass
class Row:
    """One input combination and what the circuit actually did."""

    inputs: dict[str, int]
    expected: dict[str, int]
    measured_v: dict[str, float]
    actual: dict[str, int]
    passed: bool
    # Worst distance from the correct rail, as a fraction of VDD. 0.0 is a
    # perfect rail; anything above ~0.1 is a weak or broken output.
    worst_rail_error: float


@dataclass
class Report:
    cell: str
    passed: bool
    rows: list[Row] = field(default_factory=list)
    transistors: int = 0
    reason: str = ""
    worst_rail_error: float = 0.0

    def table(self) -> str:
        if not self.rows:
            return f"  no results ({self.reason})"
        ins = list(self.rows[0].inputs)
        outs = list(self.rows[0].expected)

        head = "  " + " ".join(f"{i:>3}" for i in ins) + "  | "
        head += "  ".join(f"{o:>6}" for o in outs)
        head += "   | " + "  ".join(f"{o + '(V)':>8}" for o in outs) + "   "
        lines = [head, "  " + "-" * (len(head) - 2)]

        for r in self.rows:
            mark = "ok " if r.passed else "FAIL"
            line = "  " + " ".join(f"{r.inputs[i]:>3}" for i in ins) + "  | "
            line += "  ".join(
                f"{r.actual[o]:>6}" if r.actual[o] == r.expected[o]
                else f"{r.actual[o]}!={r.expected[o]:<3}"
                for o in outs
            )
            line += "   | " + "  ".join(f"{r.measured_v[o]:>8.3f}" for o in outs)
            line += f"   {mark}"
            lines.append(line)
        return "\n".join(lines)


def build_deck(
    circuit: Circuit,
    cell_name: str,
    pdk: PDK,
    *,
    hold: float = 20e-9,
    trise: float = 100e-12,
    cload: float = 5e-15,
) -> tuple[str, list[float]]:
    """A transient deck stepping through every input combination.

    Each combination is held for `hold` so the outputs settle, and sampled at
    90% through the interval -- late enough that the measurement reflects the
    settled logic level rather than a switching transient.
    """
    rows = logic.truth_table(cell_name)
    _, in_ports, out_ports, _ = logic.get(cell_name)
    vdd = pdk.vdd

    lines = [
        f"* bias :: {cell_name} truth-table verification :: pdk={pdk.name}",
        pdk.preamble(),
        f"Vsup {logic.VDD} 0 DC {vdd:g}",
    ]

    # Drive each input with a PWL built from its column of the truth table.
    for port in in_ports:
        seq = [r[0][port] for r in rows]
        points: list[tuple[float, float]] = [(0.0, seq[0] * vdd)]
        for k in range(1, len(seq)):
            t = k * hold
            points.append((t, seq[k - 1] * vdd))
            points.append((t + trise, seq[k] * vdd))
        points.append((len(seq) * hold, seq[-1] * vdd))
        pwl = " ".join(f"{t:.12g} {v:.6g}" for t, v in points)
        lines.append(f"V{port} {port} 0 PWL({pwl})")

    # The circuit itself.
    lines.append(circuit.to_spice({"nmos": pdk.nmos, "pmos": pdk.pmos}))

    # Every output needs a load, or it drives nothing and the node is floating
    # as far as the validator is concerned.
    for port in out_ports:
        lines.append(f"Cl_{port} {port} 0 {cload:g}")

    sample_times = [k * hold + 0.9 * hold for k in range(len(rows))]

    ctrl = [".control", "set noaskquit", f"tran {trise / 10:.12g} {len(rows) * hold:.12g}"]
    for k, t in enumerate(sample_times):
        for port in out_ports:
            ctrl.append(f"meas tran {port}_{k} FIND v({port}) AT={t:.12g}")
    ctrl += ["quit", ".endc"]

    lines.append("\n".join(ctrl))
    return "\n".join(lines), sample_times


def verify(cell_name: str, pdk: PDK, **kwargs) -> Report:
    """Build the cell, simulate every input combination, check the truth table."""
    builder, in_ports, out_ports, _ = logic.get(cell_name)
    circuit = builder()

    problems = [p for p in circuit.validate() if p.severity == "error"]
    if problems:
        return Report(
            cell_name, False, [],
            len(circuit.transistors()),
            "structural errors: " + "; ".join(str(p) for p in problems),
        )

    deck, _ = build_deck(circuit, cell_name, pdk, **kwargs)
    result = run_deck(deck, timeout=120.0)

    if not result.ok:
        return Report(
            cell_name, False, [], len(circuit.transistors()),
            result.reason or "simulation failed",
        )

    rows_ref = logic.truth_table(cell_name)
    rows: list[Row] = []
    threshold = pdk.vdd / 2.0
    worst_overall = 0.0

    for k, (inputs, expected) in enumerate(rows_ref):
        measured_v: dict[str, float] = {}
        actual: dict[str, int] = {}
        worst = 0.0
        for port in out_ports:
            v = result.values.get(f"{port}_{k}", float("nan"))
            measured_v[port] = v
            actual[port] = 1 if v > threshold else 0
            # Distance from the rail the output *should* have reached.
            ideal = pdk.vdd if expected[port] else 0.0
            worst = max(worst, abs(v - ideal) / pdk.vdd)

        ok = all(actual[p] == expected[p] for p in out_ports)
        worst_overall = max(worst_overall, worst)
        rows.append(Row(inputs, expected, measured_v, actual, ok, worst))

    passed = all(r.passed for r in rows)
    return Report(
        cell_name, passed, rows, len(circuit.transistors()),
        "" if passed else "truth table mismatch",
        worst_overall,
    )
