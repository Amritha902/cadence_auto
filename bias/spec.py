"""Design specifications and how a candidate is scored against them.

A Spec is the contract an optimizer is trying to satisfy: a list of metrics, each
with a direction and a target. Scoring is deliberately simple and reported in a
single scalar so that every optimizer -- LLM or classical -- optimizes the exact
same objective. Nothing here knows about SPICE.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

Direction = Literal["max", "min"]

# Metrics that span decades (Hz, watts, amps) are compared in the log domain so
# that "2x short on GBW" costs the same as "2x over on power". Anything not
# listed is compared linearly -- correct for dB, degrees and volts.
LOG_DOMAIN_UNITS = {"Hz", "W", "A", "F", "ohm"}


@dataclass(frozen=True)
class Metric:
    """One line of a datasheet.

    direction="max" means larger is better and `target` is a floor.
    direction="min" means smaller is better and `target` is a ceiling.
    """

    name: str
    direction: Direction
    target: float
    unit: str = ""
    # Objectives keep contributing to the score after they are met, so the
    # optimizer is rewarded for exceeding them. Constraints stop at zero.
    objective: bool = False
    weight: float = 1.0

    def satisfied(self, value: float) -> bool:
        if not math.isfinite(value):
            return False
        return value >= self.target if self.direction == "max" else value <= self.target

    def shortfall(self, value: float) -> float:
        """How far short of target, normalized. 0.0 means met.

        A non-finite measurement (failed sim, no gain crossing) is a hard miss
        rather than an exception -- optimizers must be able to score a bad point.
        """
        if not math.isfinite(value):
            return BAD_POINT_SHORTFALL

        if self.unit in LOG_DOMAIN_UNITS and value > 0 and self.target > 0:
            err = math.log10(self.target) - math.log10(value)
        else:
            scale = abs(self.target) if self.target != 0 else 1.0
            err = (self.target - value) / scale

        if self.direction == "min":
            err = -err

        if self.objective:
            # Keep rewarding improvement past the target, but at a shallower
            # slope so meeting the hard constraints always dominates.
            return err if err > 0 else 0.25 * err
        return max(0.0, err)


# Score assigned to a single metric when the simulator gives us nothing usable.
# Large enough to dominate any real shortfall, finite so optimizers can subtract.
BAD_POINT_SHORTFALL = 10.0


@dataclass(frozen=True)
class Spec:
    name: str
    metrics: tuple[Metric, ...]
    description: str = ""

    def score(self, measured: dict[str, float]) -> float:
        """Total weighted shortfall. 0.0 or below means every constraint is met.

        Lower is better. Optimizers minimize this and nothing else.
        """
        total = 0.0
        for m in self.metrics:
            value = measured.get(m.name, float("nan"))
            total += m.weight * m.shortfall(value)
        return total

    def satisfied(self, measured: dict[str, float]) -> bool:
        return all(
            m.satisfied(measured.get(m.name, float("nan")))
            for m in self.metrics
            if not m.objective
        )

    def report(self, measured: dict[str, float]) -> str:
        """Human-readable pass/fail table -- also fed verbatim to the LLM agent."""
        lines = []
        for m in self.metrics:
            value = measured.get(m.name, float("nan"))
            ok = "PASS" if m.satisfied(value) else "FAIL"
            rel = ">=" if m.direction == "max" else "<="
            kind = " (objective)" if m.objective else ""
            lines.append(
                f"  [{ok}] {m.name}: {_eng(value)}{m.unit} "
                f"(want {rel} {_eng(m.target)}{m.unit}){kind}"
            )
        return "\n".join(lines)


def _eng(x: float) -> str:
    """Engineering notation, because analog people read 1.2u not 1.2e-06."""
    if not math.isfinite(x):
        return "n/a"
    if x == 0:
        return "0"
    prefixes = {
        -18: "a", -15: "f", -12: "p", -9: "n", -6: "u",
        -3: "m", 0: "", 3: "k", 6: "M", 9: "G", 12: "T",
    }
    exp = int(math.floor(math.log10(abs(x)) / 3) * 3)
    exp = max(-18, min(12, exp))
    return f"{x / 10**exp:.4g}{prefixes[exp]}"


@dataclass
class Design:
    """A candidate point: the free parameters an optimizer may set."""

    values: dict[str, float] = field(default_factory=dict)

    def clipped(self, bounds: dict[str, tuple[float, float]]) -> "Design":
        out = {}
        for k, v in self.values.items():
            if k in bounds:
                lo, hi = bounds[k]
                v = min(hi, max(lo, v))
            out[k] = v
        return Design(out)
