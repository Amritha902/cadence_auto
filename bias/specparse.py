"""Natural language to a Spec.

"an op-amp with 60dB gain and 10MHz bandwidth under 100uW" has to become
Metric objects the optimizer can chase. The parsing is deliberately
deterministic: engineering notation and the vocabulary analog designers use are
both small and regular, so a lookup beats a model on accuracy, latency and
cost, and it cannot invent a target the user did not ask for -- which for a
*specification* is the failure that matters most.

Numbers and keywords are found independently and then matched by unit and
proximity, so "60dB gain" and "gain of 60 dB" parse identically without needing
a pattern for each phrasing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .spec import Metric, Spec

# SI prefixes as they actually appear in datasheets and conversation.
PREFIX: dict[str, float] = {
    "f": 1e-15, "p": 1e-12, "n": 1e-9,
    "u": 1e-6, "µ": 1e-6, "μ": 1e-6,
    "m": 1e-3, "k": 1e3, "K": 1e3,
    "meg": 1e6, "M": 1e6, "G": 1e9, "T": 1e12,
}

# Canonical unit -> the spellings that mean it.
UNIT_WORDS: dict[str, tuple[str, ...]] = {
    "dB": ("db",),
    "Hz": ("hz", "hertz"),
    "W": ("w", "watt", "watts"),
    "deg": ("deg", "degree", "degrees", "°"),
    "F": ("f", "farad", "farads"),
    "V": ("v", "volt", "volts"),
    "A": ("a", "amp", "amps", "ampere", "amperes"),
}

_UNIT_ALTS = "|".join(
    sorted((w for words in UNIT_WORDS.values() for w in words), key=len, reverse=True)
)
_PREFIX_ALTS = "|".join(
    sorted((p for p in PREFIX if p != "m"), key=len, reverse=True)
)

# A number, an optional SI prefix, and a unit. "m" is handled separately below
# because "10 mW" and "10 MHz" differ only by case.
_QTY = re.compile(
    rf"(?P<num>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*"
    rf"(?P<prefix>{_PREFIX_ALTS}|m)?\s*"
    rf"(?P<unit>{_UNIT_ALTS})\b",
    re.IGNORECASE,
)


@dataclass
class Quantity:
    value: float
    unit: str
    at: int          # character offset, for proximity matching
    text: str = ""


@dataclass
class Target:
    """One metric the parser knows how to recognise."""

    name: str
    unit: str
    direction: str                       # "max" = floor, "min" = ceiling
    keywords: tuple[str, ...]
    weight: float = 1.0


TARGETS: tuple[Target, ...] = (
    Target("gain", "dB", "max",
           ("gain", "a0", "av", "amplification", "db of gain")),
    Target("gbw", "Hz", "max",
           ("bandwidth", "gbw", "gain bandwidth", "unity gain", "ugf",
            "unity-gain", "gain-bandwidth", "bw", "frequency")),
    Target("pm", "deg", "max",
           ("phase margin", "phase-margin", "pm", "stability", "margin")),
    Target("pwr", "W", "min",
           ("power", "consumption", "consume", "budget", "dissipation",
            "dissipate")),
)

# The load is a testbench condition, not something to optimize toward.
LOAD_KEYWORDS = ("load", "cl", "driving", "drive", "capacitive")

# Sensible floors for anything the user did not mention. An amplifier with an
# unstated phase margin still has to be stable, and an unstated output still
# has to sit off the rails -- leaving those unconstrained produces a circuit
# that meets the letter of the request and is not usable.
IMPLIED: dict[str, Metric] = {
    "pm": Metric("pm", "max", 55.0, "deg"),
    "vout_margin": Metric("vout_margin", "max", 0.20, "", weight=2.0),
}


@dataclass
class ParsedSpec:
    spec: Spec
    cl: float | None = None
    found: dict[str, float] = field(default_factory=dict)
    implied: list[str] = field(default_factory=list)
    unmatched: list[str] = field(default_factory=list)

    def describe(self) -> str:
        lines = []
        for m in self.spec.metrics:
            rel = ">=" if m.direction == "max" else "<="
            note = "  (implied, not stated)" if m.name in self.implied else ""
            lines.append(f"  {m.name:<12} {rel} {_eng(m.target)}{m.unit}{note}")
        if self.cl:
            lines.append(f"  {'load':<12} =  {_eng(self.cl)}F")
        return "\n".join(lines)


def quantities(text: str) -> list[Quantity]:
    """Every number-with-unit in the text, normalised to SI base units."""
    out: list[Quantity] = []
    for m in _QTY.finditer(text):
        raw_unit = m.group("unit").lower()
        unit = next(
            (canon for canon, words in UNIT_WORDS.items() if raw_unit in words),
            None,
        )
        if unit is None:
            continue

        prefix = m.group("prefix") or ""
        # "M" is mega and "m" is milli, and lowercasing the pattern would
        # silently turn 10 MHz into 10 mHz.
        if prefix:
            literal = m.group(0)[m.start("prefix") - m.start(): ][:len(prefix)]
            scale = PREFIX.get(literal, PREFIX.get(prefix.lower(), 1.0))
        else:
            scale = 1.0

        # dB and degrees are not scaled by SI prefixes in practice.
        if unit in ("dB", "deg"):
            scale = 1.0

        out.append(
            Quantity(float(m.group("num")) * scale, unit, m.start(), m.group(0))
        )
    return out


def parse(text: str, *, strict: bool = False) -> ParsedSpec:
    """Turn a request into a Spec.

    Raises ValueError when nothing recognisable is found, because silently
    returning an empty spec would let the optimizer "succeed" against no
    constraints at all.
    """
    low = text.lower()
    qtys = quantities(text)
    used: set[int] = set()
    metrics: list[Metric] = []
    found: dict[str, float] = {}

    for target in TARGETS:
        hits = [
            m.start() for kw in target.keywords
            for m in re.finditer(rf"\b{re.escape(kw)}\b", low)
        ]
        if not hits:
            continue

        # Closest unused quantity with the right unit.
        candidates = [
            (min(abs(q.at - h) for h in hits), i, q)
            for i, q in enumerate(qtys)
            if q.unit == target.unit and i not in used
        ]
        if not candidates:
            continue
        distance, index, qty = min(candidates, key=lambda c: c[0])

        used.add(index)
        found[target.name] = qty.value
        metrics.append(
            Metric(target.name, target.direction, qty.value, target.unit,
                   weight=target.weight)
        )

    # Second pass: a quantity whose unit belongs to exactly one target is
    # unambiguous even with no keyword near it. "under 100uW" names no metric,
    # but watts in an amplifier request can only be power.
    by_unit: dict[str, list[Target]] = {}
    for t in TARGETS:
        by_unit.setdefault(t.unit, []).append(t)

    for index, qty in enumerate(qtys):
        if index in used:
            continue
        owners = by_unit.get(qty.unit, [])
        if len(owners) != 1:
            continue
        target = owners[0]
        if target.name in found:
            continue
        used.add(index)
        found[target.name] = qty.value
        metrics.append(
            Metric(target.name, target.direction, qty.value, target.unit,
                   weight=target.weight)
        )

    # Load capacitance: a condition, not a target.
    cl: float | None = None
    load_hits = [
        m.start() for kw in LOAD_KEYWORDS
        for m in re.finditer(rf"\b{re.escape(kw)}\b", low)
    ]
    farads = [(i, q) for i, q in enumerate(qtys) if q.unit == "F" and i not in used]
    if farads:
        if load_hits:
            distance, index, qty = min(
                ((min(abs(q.at - h) for h in load_hits), i, q) for i, q in farads),
                key=lambda c: c[0],
            )
            used.add(index)
            cl = qty.value
        else:
            # A lone capacitance in an amplifier request is the load.
            index, qty = farads[0]
            used.add(index)
            cl = qty.value

    if not metrics:
        raise ValueError(
            f"no design targets found in {text!r}.\n"
            "Say something like: 60dB gain, 10MHz bandwidth, under 100uW"
        )

    implied: list[str] = []
    have = {m.name for m in metrics}
    for name, metric in IMPLIED.items():
        if name not in have:
            metrics.append(metric)
            implied.append(name)

    unmatched = [q.text for i, q in enumerate(qtys) if i not in used]
    if strict and unmatched:
        raise ValueError(f"could not place these values: {unmatched}")

    spec = Spec(
        name="request",
        metrics=tuple(metrics),
        description=text.strip(),
    )
    return ParsedSpec(spec, cl, found, implied, unmatched)


def _eng(x: float) -> str:
    from .spec import _eng as fmt
    return fmt(x)
