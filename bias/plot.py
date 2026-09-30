"""Bode plots from a real AC sweep.

The sizing loop reports scalars -- gain, GBW, phase margin -- which is what an
optimizer needs but not what a designer reads. The shape of the response is
where the design actually shows itself: where the poles sit, whether the
crossover is clean, how much margin there really is either side of it.

So this re-runs the sized circuit with the sweep written out, and draws it.
Nothing here is interpolated or idealised; every point is a simulated point.
"""

from __future__ import annotations

import math
import re
import shutil
import tempfile
from pathlib import Path

from .pdk import PDK
from .sim import run_deck
from .topology import Testbench, Topology

_CONTROL = re.compile(r"\.control.*?\.endc", re.DOTALL | re.IGNORECASE)


def sweep(
    topology: Topology,
    pdk: PDK,
    values: dict[str, float],
    tb: Testbench,
) -> list[tuple[float, float, float]]:
    """Run an AC sweep and return [(freq, magnitude_dB, phase_deg), ...]."""
    base = topology.deck(pdk, values, tb)
    workdir = Path(tempfile.mkdtemp(prefix="bias-bode-"))
    data = workdir / "ac.data"

    control = f""".control
set units=degrees
set noaskquit
option temp={tb.temp}
ac dec {tb.pts_per_dec} {tb.fstart:g} {tb.fstop:g}
wrdata {data} vdb(vout) vp(vout)
quit
.endc"""

    deck = _CONTROL.sub(lambda _m: control, base, count=1)

    try:
        run_deck(deck, timeout=60.0)
        if not data.exists():
            return []
        rows: list[tuple[float, float, float]] = []
        for line in data.read_text().splitlines():
            parts = line.split()
            # wrdata repeats the sweep variable before each vector:
            # freq vdb freq vphase
            if len(parts) < 4:
                continue
            try:
                f, mag, _f2, ph = (float(parts[0]), float(parts[1]),
                                   float(parts[2]), float(parts[3]))
            except ValueError:
                continue
            if f > 0:
                rows.append((f, mag, ph))
        return rows
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def bode(
    topology: Topology,
    pdk: PDK,
    values: dict[str, float],
    tb: Testbench,
    measured: dict[str, float] | None = None,
) -> str | None:
    """Render the sized circuit's frequency response as standalone SVG."""
    rows = sweep(topology, pdk, values, tb)
    if len(rows) < 8:
        return None

    measured = measured or {}
    freqs = [r[0] for r in rows]
    mags = [r[1] for r in rows]
    phases = _unwrap([r[2] for r in rows])

    W, H = 660.0, 420.0
    L, R, T = 62.0, 22.0, 26.0
    gap, axis_h = 34.0, 150.0
    pw = W - L - R

    fx0, fx1 = math.log10(min(freqs)), math.log10(max(freqs))
    my0, my1 = _nice_range(min(mags), max(mags))
    py0, py1 = _nice_range(min(phases), max(phases))

    def x(f: float) -> float:
        return L + pw * (math.log10(f) - fx0) / max(fx1 - fx0, 1e-9)

    def ym(v: float) -> float:
        return T + axis_h * (1 - (v - my0) / max(my1 - my0, 1e-9))

    def yp(v: float) -> float:
        top = T + axis_h + gap
        return top + axis_h * (1 - (v - py0) / max(py1 - py0, 1e-9))

    p: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W:.0f}" height="{H:.0f}" '
        f'viewBox="0 0 {W:.0f} {H:.0f}" role="img" '
        f'aria-label="frequency response of {topology.name}">',
        _STYLE,
        f'<rect class="bg" width="{W:.0f}" height="{H:.0f}"/>',
    ]

    # Decade gridlines, shared by both panels.
    for d in range(math.floor(fx0), math.ceil(fx1) + 1):
        f = 10.0 ** d
        if not (min(freqs) <= f <= max(freqs)):
            continue
        gx = x(f)
        p.append(f'<line class="grid" x1="{gx:.1f}" y1="{T:.1f}" '
                 f'x2="{gx:.1f}" y2="{T + axis_h:.1f}"/>')
        p.append(f'<line class="grid" x1="{gx:.1f}" y1="{T + axis_h + gap:.1f}" '
                 f'x2="{gx:.1f}" y2="{T + 2 * axis_h + gap:.1f}"/>')
        p.append(f'<text class="tick" x="{gx:.1f}" '
                 f'y="{T + 2 * axis_h + gap + 16:.1f}" text-anchor="middle">'
                 f'{_hz(f)}</text>')

    # Horizontal references: 0 dB and the phase axis ends.
    for value, label, yfun, lo, hi in (
        (0.0, "0 dB", ym, my0, my1),
    ):
        if lo <= value <= hi:
            gy = yfun(value)
            p.append(f'<line class="zero" x1="{L:.1f}" y1="{gy:.1f}" '
                     f'x2="{L + pw:.1f}" y2="{gy:.1f}"/>')

    # Traces.
    p.append(f'<polyline class="mag" points="'
             + " ".join(f"{x(f):.1f},{ym(m):.1f}" for f, m in zip(freqs, mags))
             + '"/>')
    p.append(f'<polyline class="ph" points="'
             + " ".join(f"{x(f):.1f},{yp(v):.1f}" for f, v in zip(freqs, phases))
             + '"/>')

    # Crossover marker, drawn from the measured GBW so the plot and the
    # reported numbers cannot disagree.
    gbw = measured.get("gbw")
    if gbw and min(freqs) <= gbw <= max(freqs):
        gx = x(gbw)
        p.append(f'<line class="mark" x1="{gx:.1f}" y1="{T:.1f}" '
                 f'x2="{gx:.1f}" y2="{T + 2 * axis_h + gap:.1f}"/>')
        p.append(f'<text class="note" x="{gx + 5:.1f}" y="{T + 12:.1f}">'
                 f'GBW {_hz(gbw)}</text>')

    # Panel frames and labels.
    for top, label in ((T, "magnitude (dB)"),
                       (T + axis_h + gap, "phase (deg)")):
        p.append(f'<rect class="frame" x="{L:.1f}" y="{top:.1f}" '
                 f'width="{pw:.1f}" height="{axis_h:.1f}"/>')
        p.append(f'<text class="axis" x="{10:.1f}" y="{top + axis_h / 2:.1f}" '
                 f'transform="rotate(-90 10 {top + axis_h / 2:.1f})" '
                 f'text-anchor="middle">{label}</text>')

    for value, top, lo, hi in ((my1, T, my0, my1), (my0, T, my0, my1)):
        p.append(f'<text class="tick" x="{L - 7:.1f}" y="{ym(value) + 4:.1f}" '
                 f'text-anchor="end">{value:.0f}</text>')
    for value in (py1, py0):
        p.append(f'<text class="tick" x="{L - 7:.1f}" y="{yp(value) + 4:.1f}" '
                 f'text-anchor="end">{value:.0f}</text>')

    bits = []
    if "gain" in measured:
        bits.append(f"gain {measured['gain']:.1f} dB")
    if "pm" in measured:
        bits.append(f"phase margin {measured['pm']:.1f}°")
    if "pwr" in measured:
        from .spec import _eng
        bits.append(f"power {_eng(measured['pwr'])}W")
    if bits:
        p.append(f'<text class="sub" x="{L:.1f}" y="{H - 8:.1f}">'
                 f'{topology.name} — ' + "   ".join(bits) + '</text>')

    p.append("</svg>")
    return "\n".join(p)


def _unwrap(phases: list[float]) -> list[float]:
    """Remove 360-degree jumps so the phase trace is continuous."""
    out = [phases[0]]
    offset = 0.0
    for prev, cur in zip(phases, phases[1:]):
        delta = cur - prev
        if delta > 180:
            offset -= 360
        elif delta < -180:
            offset += 360
        out.append(cur + offset)
    return out


def _nice_range(lo: float, hi: float) -> tuple[float, float]:
    if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
        return (0.0, 1.0)
    pad = (hi - lo) * 0.08
    step = 10.0 ** max(0, math.floor(math.log10(max(hi - lo, 1e-9))) - 1)
    return (math.floor((lo - pad) / step) * step,
            math.ceil((hi + pad) / step) * step)


def _hz(f: float) -> str:
    """Three significant figures. '12.2M' reads; '12.1716M' does not."""
    for scale, suffix in ((1e9, "G"), (1e6, "M"), (1e3, "k"), (1.0, "")):
        if f >= scale:
            return f"{f / scale:.3g}{suffix}"
    return f"{f:.3g}"


_STYLE = """<style>
  .bg    { fill: #fbfaf7; }
  .frame { fill: none; stroke: #2a2724; stroke-width: 1.1; }
  .grid  { stroke: #ddd6cb; stroke-width: .8; }
  .zero  { stroke: #b9b0a3; stroke-width: .9; stroke-dasharray: 4 3; }
  .mag   { fill: none; stroke: #b4552d; stroke-width: 1.9;
           stroke-linejoin: round; }
  .ph    { fill: none; stroke: #3d6b8e; stroke-width: 1.9;
           stroke-linejoin: round; }
  .mark  { stroke: #8a8179; stroke-width: 1; stroke-dasharray: 3 3; }
  .tick  { font: 400 10px ui-monospace, SFMono-Regular, Menlo, monospace;
           fill: #8a8179; }
  .axis  { font: 500 10.5px ui-sans-serif, system-ui, sans-serif; fill: #55504a; }
  .note  { font: 500 10px ui-monospace, Menlo, monospace; fill: #55504a; }
  .sub   { font: 400 11px ui-serif, Georgia, serif; fill: #55504a; }
  @media (prefers-color-scheme: dark) {
    .bg    { fill: #14120f; }
    .frame { stroke: #e8e2d9; }
    .grid  { stroke: #322c25; }
    .zero  { stroke: #4a4238; }
    .mag   { stroke: #e08a5c; }
    .ph    { stroke: #7aa9cc; }
    .tick, .note, .axis, .sub { fill: #b8afa4; }
  }
</style>"""
