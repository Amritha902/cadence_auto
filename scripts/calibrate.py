"""Check every spec in the suite is reachable, and how hard it is.

A benchmark target nobody can hit is a bug, not a challenge. This runs the
strongest classical baseline at a generous budget on each spec and reports
whether it was solved and how many simulations it took. Use the output to
retune specs.py -- targets should be reachable but not trivial.

    .venv/bin/python scripts/calibrate.py [budget] [seeds] [pdk]
"""

from __future__ import annotations

import sys

import numpy as np

from bias import optimizers, pdk, specs, topology
from bias.evaluate import Evaluator


def main() -> int:
    budget = int(sys.argv[1]) if len(sys.argv) > 1 else 800
    n_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    pdk_name = sys.argv[3] if len(sys.argv) > 3 else "dev180"

    process = pdk.DEV180 if pdk_name == "dev180" else pdk.get(pdk_name)
    tb = topology.Testbench(cl=1e-12)

    print(f"calibrating with de, budget={budget}, seeds={n_seeds}, pdk={process.name}\n")
    header = f"{'spec':<14} {'solved':>7} {'median sims':>12} {'best score':>11}"
    print(header)
    print("-" * len(header))

    for name in specs.names():
        spec, topo_name = specs.get(name)
        topo = topology.get(topo_name)

        hits, best_scores = [], []
        for seed in range(n_seeds):
            ev = Evaluator(topo, process, spec, tb, budget=budget)
            optimizers.get("de").run(ev, np.random.default_rng(seed))
            hits.append(ev.sims_to_target())
            best_scores.append(ev.best.score if ev.best else float("inf"))

        solved = [h for h in hits if h is not None]
        rate = f"{len(solved)}/{n_seeds}"
        med = f"{int(np.median(solved))}" if solved else "-"
        print(f"{name:<14} {rate:>7} {med:>12} {min(best_scores):>11.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
