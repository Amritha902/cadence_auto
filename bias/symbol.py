"""Schematic symbol generation.

The symbol is the thing you place in a higher-level schematic: a block with
named pins, hiding the transistors underneath. This renders one as standalone
SVG from a Circuit's port list, which is the same information a Virtuoso symbol
view carries.

Pin sides are inferred from the circuit's own port roles rather than guessed:
supplies go top and bottom, inputs left, outputs right. That inference is
explicit below so it can be overridden when a circuit does something unusual.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .netlist import GROUND, Circuit

# Geometry, in SVG user units.
PIN = 34.0          # pin lead length
ROW = 30.0          # vertical spacing between pins
PAD = 26.0          # box padding above first / below last pin
MIN_H = 96.0
CHAR = 7.4          # approximate advance of the label font
MARGIN = 7.0        # keeps pin dots off the canvas edge, where they clip


@dataclass
class Symbol:
    name: str
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    power: list[str] = field(default_factory=list)
    ground: list[str] = field(default_factory=list)
    subtitle: str = ""

    @classmethod
    def from_circuit(
        cls,
        circuit: Circuit,
        inputs: list[str] | None = None,
        outputs: list[str] | None = None,
        subtitle: str = "",
    ) -> "Symbol":
        """Infer pin roles from the port list when they are not given."""
        power, ground, ins, outs = [], [], [], []
        for port in circuit.ports:
            low = port.lower()
            if port == GROUND or low in ("gnd", "vss"):
                ground.append(port)
            elif low in ("vdd", "vcc", "vpwr"):
                power.append(port)
            elif inputs is not None and port in inputs:
                ins.append(port)
            elif outputs is not None and port in outputs:
                outs.append(port)
            elif inputs is None and outputs is None:
                # No hint: treat conventional output names as outputs.
                (outs if low in ("y", "out", "vout", "sum", "carry", "cout", "q")
                 else ins).append(port)
        return cls(circuit.name, ins, outs, power, ground, subtitle)

    # -- rendering ------------------------------------------------------

    def render(self) -> str:
        rows = max(len(self.inputs), len(self.outputs), 1)
        h = max(MIN_H, rows * ROW + 2 * PAD)

        label_w = max(
            [CHAR * len(self.name) + 40]
            + [CHAR * len(p) * 2 + 60 for p in (self.inputs + self.outputs)] or [0]
        )
        w = max(150.0, label_w)

        x0, y0 = PIN + MARGIN, 46.0
        svg_w = w + 2 * PIN + 2 * MARGIN
        svg_h = y0 + h + 58

        parts: list[str] = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{svg_w:.0f}" '
            f'height="{svg_h:.0f}" viewBox="0 0 {svg_w:.0f} {svg_h:.0f}" '
            f'role="img" aria-label="{_esc(self.name)} schematic symbol">',
            _STYLE,
            f'<rect class="bg" x="0" y="0" width="{svg_w:.0f}" height="{svg_h:.0f}"/>',
        ]

        # Body.
        parts.append(
            f'<rect class="body" x="{x0:.1f}" y="{y0:.1f}" '
            f'width="{w:.1f}" height="{h:.1f}" rx="4"/>'
        )

        # Title, centred in the body.
        parts.append(
            f'<text class="title" x="{x0 + w / 2:.1f}" y="{y0 + h / 2 + 5:.1f}" '
            f'text-anchor="middle">{_esc(self.name)}</text>'
        )

        # Input pins: lead to the left edge.
        for i, port in enumerate(self.inputs):
            y = self._pin_y(y0, h, i, len(self.inputs))
            parts.append(f'<line class="wire" x1="{x0 - PIN:.1f}" y1="{y:.1f}" '
                         f'x2="{x0:.1f}" y2="{y:.1f}"/>')
            parts.append(f'<circle class="dot" cx="{x0 - PIN:.1f}" cy="{y:.1f}" r="2.6"/>')
            parts.append(f'<text class="pin" x="{x0 + 9:.1f}" y="{y + 4:.1f}">'
                         f'{_esc(port)}</text>')

        # Output pins: lead to the right edge.
        for i, port in enumerate(self.outputs):
            y = self._pin_y(y0, h, i, len(self.outputs))
            parts.append(f'<line class="wire" x1="{x0 + w:.1f}" y1="{y:.1f}" '
                         f'x2="{x0 + w + PIN:.1f}" y2="{y:.1f}"/>')
            parts.append(f'<circle class="dot" cx="{x0 + w + PIN:.1f}" cy="{y:.1f}" r="2.6"/>')
            parts.append(f'<text class="pin" x="{x0 + w - 9:.1f}" y="{y + 4:.1f}" '
                         f'text-anchor="end">{_esc(port)}</text>')

        # Supply stubs, top and bottom.
        cx = x0 + w / 2
        for port in self.power:
            parts.append(f'<line class="wire" x1="{cx:.1f}" y1="{y0 - 22:.1f}" '
                         f'x2="{cx:.1f}" y2="{y0:.1f}"/>')
            parts.append(f'<text class="rail" x="{cx:.1f}" y="{y0 - 28:.1f}" '
                         f'text-anchor="middle">{_esc(port)}</text>')
        for port in self.ground:
            yb = y0 + h
            parts.append(f'<line class="wire" x1="{cx:.1f}" y1="{yb:.1f}" '
                         f'x2="{cx:.1f}" y2="{yb + 22:.1f}"/>')
            parts.append(f'<text class="rail" x="{cx:.1f}" y="{yb + 38:.1f}" '
                         f'text-anchor="middle">{_esc(port)}</text>')

        if self.subtitle:
            parts.append(
                f'<text class="sub" x="{svg_w / 2:.1f}" y="{svg_h - 12:.1f}" '
                f'text-anchor="middle">{_esc(self.subtitle)}</text>'
            )

        parts.append("</svg>")
        return "\n".join(parts)

    @staticmethod
    def _pin_y(y0: float, h: float, i: int, n: int) -> float:
        """Distribute n pins evenly down the body edge."""
        if n == 1:
            return y0 + h / 2
        span = h - 2 * PAD
        return y0 + PAD + span * i / (n - 1)


_STYLE = """<style>
  .bg    { fill: #fbfaf7; }
  .body  { fill: #ffffff; stroke: #2a2724; stroke-width: 1.6; }
  .wire  { stroke: #2a2724; stroke-width: 1.4; stroke-linecap: round; }
  .dot   { fill: #2a2724; }
  .title { font: 600 15px ui-serif, Georgia, serif; fill: #2a2724;
           letter-spacing: .01em; }
  .pin   { font: 500 11px ui-monospace, SFMono-Regular, Menlo, monospace;
           fill: #55504a; }
  .rail  { font: 500 10.5px ui-monospace, SFMono-Regular, Menlo, monospace;
           fill: #8a8179; letter-spacing: .04em; }
  .sub   { font: 400 10.5px ui-sans-serif, system-ui, sans-serif; fill: #8a8179; }
  @media (prefers-color-scheme: dark) {
    .bg    { fill: #14120f; }
    .body  { fill: #1c1917; stroke: #e8e2d9; }
    .wire  { stroke: #e8e2d9; }
    .dot   { fill: #e8e2d9; }
    .title { fill: #f5f0e8; }
    .pin   { fill: #b8afa4; }
    .rail  { fill: #8a8179; }
    .sub   { fill: #8a8179; }
  }
</style>"""


def _esc(s: str) -> str:
    return (s.replace("&", "&amp;").replace("<", "&lt;")
             .replace(">", "&gt;").replace('"', "&quot;"))
