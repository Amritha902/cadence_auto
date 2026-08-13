"""Classical sizing baselines.

The point of this module is to make the LLM agent prove itself. A language
model that sizes an op-amp is only interesting if it beats the optimizers a
designer could have run for free -- so these are implemented honestly, not as
strawmen, and they get the identical simulation budget.

Everything searches the unit hypercube: Param.from_unit maps each coordinate
back to a width, length or current, in log space where that is the natural
scale. Implementations are pure numpy so results are reproducible from a seed
with no solver-version drift.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .evaluate import BudgetExhausted, Evaluator


class Optimizer(ABC):
    name: str = ""

    @abstractmethod
    def run(self, ev: Evaluator, rng: np.random.Generator) -> None: ...

    # -- helpers shared by every optimizer -------------------------------

    @staticmethod
    def _to_values(ev: Evaluator, u: np.ndarray) -> dict[str, float]:
        return {
            p.name: p.from_unit(float(x)) for p, x in zip(ev.topology.params, u)
        }

    @classmethod
    def _score(cls, ev: Evaluator, u: np.ndarray) -> float:
        return ev.evaluate(cls._to_values(ev, u)).score

    @staticmethod
    def _dim(ev: Evaluator) -> int:
        return len(ev.topology.params)


class RandomSearch(Optimizer):
    """Uniform sampling of the unit cube. The floor any method must clear."""

    name = "random"

    def run(self, ev: Evaluator, rng: np.random.Generator) -> None:
        n = self._dim(ev)
        try:
            while not ev.exhausted:
                self._score(ev, rng.random(n))
        except BudgetExhausted:
            pass


class LatinHypercube(Optimizer):
    """Stratified sampling -- a genuinely better random baseline.

    Each coordinate is divided into `budget` strata and visited exactly once,
    so the sample cannot clump the way uniform draws do in high dimensions.
    """

    name = "lhs"

    def run(self, ev: Evaluator, rng: np.random.Generator) -> None:
        n, budget = self._dim(ev), ev.budget - ev.used
        if budget <= 0:
            return
        # Column i is a permutation of the strata, jittered within each stratum.
        grid = np.empty((budget, n))
        for i in range(n):
            perm = rng.permutation(budget)
            grid[:, i] = (perm + rng.random(budget)) / budget
        try:
            for row in grid:
                if ev.exhausted:
                    break
                self._score(ev, row)
        except BudgetExhausted:
            pass


class NelderMead(Optimizer):
    """Multi-start Nelder-Mead simplex, restarted until the budget runs out.

    Derivative-free and the standard workhorse for this kind of problem. The
    restarts matter: the objective is riddled with flat regions where the
    simulator fails to converge, and a single simplex gets stuck in them.
    """

    name = "nelder-mead"

    def __init__(self, max_iter_per_start: int = 60) -> None:
        self.max_iter_per_start = max_iter_per_start

    def run(self, ev: Evaluator, rng: np.random.Generator) -> None:
        try:
            while not ev.exhausted:
                self._one_start(ev, rng)
        except BudgetExhausted:
            pass

    def _one_start(self, ev: Evaluator, rng: np.random.Generator) -> None:
        n = self._dim(ev)
        alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5

        # Initial simplex: a random point plus n perturbations of it.
        x0 = rng.random(n)
        simplex = [np.clip(x0, 0, 1)]
        for i in range(n):
            x = x0.copy()
            x[i] = np.clip(x[i] + rng.uniform(0.15, 0.35) * rng.choice([-1, 1]), 0, 1)
            simplex.append(x)
        pts = np.array(simplex)
        vals = np.array([self._score(ev, p) for p in pts])

        for _ in range(self.max_iter_per_start):
            if ev.exhausted:
                return
            order = np.argsort(vals)
            pts, vals = pts[order], vals[order]

            centroid = pts[:-1].mean(axis=0)
            worst = pts[-1]

            xr = np.clip(centroid + alpha * (centroid - worst), 0, 1)
            fr = self._score(ev, xr)

            if fr < vals[0]:
                xe = np.clip(centroid + gamma * (xr - centroid), 0, 1)
                fe = self._score(ev, xe)
                pts[-1], vals[-1] = (xe, fe) if fe < fr else (xr, fr)
            elif fr < vals[-2]:
                pts[-1], vals[-1] = xr, fr
            else:
                xc = np.clip(centroid + rho * (worst - centroid), 0, 1)
                fc = self._score(ev, xc)
                if fc < vals[-1]:
                    pts[-1], vals[-1] = xc, fc
                else:
                    # Shrink toward the best vertex.
                    for i in range(1, len(pts)):
                        pts[i] = np.clip(pts[0] + sigma * (pts[i] - pts[0]), 0, 1)
                        vals[i] = self._score(ev, pts[i])

            # Converged to a point: restart rather than burn budget in place.
            if np.max(np.abs(pts - pts[0])) < 1e-4:
                return


class DifferentialEvolution(Optimizer):
    """Classic DE/rand/1/bin.

    The strongest classical baseline here. Population-based, so it handles the
    discontinuities where a candidate simply fails to converge, and it does not
    need gradients that this objective does not have.
    """

    name = "de"

    def __init__(self, pop_size: int = 20, f: float = 0.6, cr: float = 0.9) -> None:
        self.pop_size = pop_size
        self.f = f
        self.cr = cr

    def run(self, ev: Evaluator, rng: np.random.Generator) -> None:
        n = self._dim(ev)
        pop_size = max(5, min(self.pop_size, max(5, ev.budget // 4)))
        try:
            pop = rng.random((pop_size, n))
            fit = np.array([self._score(ev, p) for p in pop])

            while not ev.exhausted:
                for i in range(pop_size):
                    if ev.exhausted:
                        break
                    idxs = [j for j in range(pop_size) if j != i]
                    a, b, c = pop[rng.choice(idxs, 3, replace=False)]
                    mutant = np.clip(a + self.f * (b - c), 0, 1)

                    cross = rng.random(n) < self.cr
                    if not cross.any():
                        cross[rng.integers(n)] = True
                    trial = np.where(cross, mutant, pop[i])

                    ft = self._score(ev, trial)
                    if ft <= fit[i]:
                        pop[i], fit[i] = trial, ft
        except BudgetExhausted:
            pass


REGISTRY: dict[str, type[Optimizer]] = {
    RandomSearch.name: RandomSearch,
    LatinHypercube.name: LatinHypercube,
    NelderMead.name: NelderMead,
    DifferentialEvolution.name: DifferentialEvolution,
}


def get(name: str, **kwargs) -> Optimizer:
    if name in REGISTRY:
        return REGISTRY[name](**kwargs)
    # The LLM agent lives in bias.agent so that the core benchmark has no
    # dependency on an API client. Imported lazily and only if asked for.
    if name in ("llm", "agent"):
        from .agent import LLMAgent

        return LLMAgent(**kwargs)
    raise KeyError(f"unknown optimizer {name!r}; have {sorted(REGISTRY) + ['llm']}")


def available() -> list[str]:
    return sorted(REGISTRY) + ["llm"]
