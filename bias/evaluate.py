"""The single evaluation path.

Every optimizer -- LLM or classical -- reaches the simulator through an
Evaluator and nothing else. That is the whole point: the simulation budget is
counted in one place, with one definition of "one simulation", so
`sims_to_target` means the same thing for a random search and for an agent.

The Evaluator also caches on the exact parameter vector, so an optimizer that
re-proposes a point it already tried does not get charged for it twice -- and
cannot accidentally look more efficient by rediscovering its own history.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from .pdk import PDK
from .sim import SimResult, run_deck
from .spec import Spec
from .topology import Testbench, Topology


@dataclass
class Evaluation:
    """One scored candidate."""

    index: int
    values: dict[str, float]
    measured: dict[str, float]
    score: float
    satisfied: bool
    ok: bool
    reason: str = ""
    seconds: float = 0.0
    cached: bool = False

    def summary(self, spec: Spec) -> str:
        head = (
            f"sim #{self.index}  score={self.score:.4f}  "
            f"{'ALL SPECS MET' if self.satisfied else 'not yet'}"
        )
        if not self.ok:
            head += f"  [simulation failed: {self.reason}]"
        return head + "\n" + spec.report(self.measured)


@dataclass
class Evaluator:
    topology: Topology
    pdk: PDK
    spec: Spec
    testbench: Testbench = field(default_factory=Testbench)
    budget: int = 100
    timeout: float = 30.0

    history: list[Evaluation] = field(default_factory=list)
    _cache: dict[tuple, Evaluation] = field(default_factory=dict, repr=False)

    @property
    def used(self) -> int:
        """Simulations actually spent. Cache hits are free and excluded."""
        return sum(1 for e in self.history if not e.cached)

    @property
    def exhausted(self) -> bool:
        return self.used >= self.budget

    @property
    def best(self) -> Evaluation | None:
        return min(self.history, key=lambda e: e.score) if self.history else None

    def solved(self) -> Evaluation | None:
        """First evaluation that met every hard constraint, if any."""
        for e in self.history:
            if e.satisfied:
                return e
        return None

    def sims_to_target(self) -> int | None:
        hit = self.solved()
        return hit.index if hit else None

    def __call__(self, values: dict[str, float]) -> Evaluation:
        return self.evaluate(values)

    def evaluate(self, values: dict[str, float]) -> Evaluation:
        clipped = {p.name: p.clip(float(values[p.name])) for p in self.topology.params}
        key = tuple(round(clipped[p.name], 15) for p in self.topology.params)

        if key in self._cache:
            prior = self._cache[key]
            hit = Evaluation(
                index=prior.index,
                values=prior.values,
                measured=prior.measured,
                score=prior.score,
                satisfied=prior.satisfied,
                ok=prior.ok,
                reason=prior.reason,
                seconds=0.0,
                cached=True,
            )
            self.history.append(hit)
            return hit

        if self.exhausted:
            raise BudgetExhausted(
                f"simulation budget of {self.budget} is spent"
            )

        deck = self.topology.deck(self.pdk, clipped, self.testbench)
        t0 = time.perf_counter()
        result: SimResult = run_deck(deck, timeout=self.timeout)
        elapsed = time.perf_counter() - t0

        measured = self.topology.postprocess(result.values, clipped, self.pdk)
        ev = Evaluation(
            index=self.used + 1,
            values=clipped,
            measured=measured,
            score=self.spec.score(measured),
            satisfied=result.ok and self.spec.satisfied(measured),
            ok=result.ok,
            reason=result.reason,
            seconds=elapsed,
        )
        self._cache[key] = ev
        self.history.append(ev)
        return ev


class BudgetExhausted(RuntimeError):
    pass
