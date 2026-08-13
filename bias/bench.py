"""The benchmark harness.

Runs an optimizer against a spec over several seeds and records what happened,
in a form that can be diffed and published. The headline numbers are:

  success rate     -- fraction of seeds that met every hard constraint
  sims to target   -- median simulations spent before the first solution
  best score       -- how close the failures got

Results are keyed by (spec, optimizer, seed) and written as JSON so a run can
be re-analysed without re-simulating.
"""

from __future__ import annotations

import json
import platform
import statistics
import subprocess
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import optimizers as opt_registry
from . import pdk as pdk_registry
from . import specs as spec_registry
from . import topology as topo_registry
from .evaluate import Evaluator
from .topology import Testbench


@dataclass
class RunResult:
    spec: str
    optimizer: str
    seed: int
    solved: bool
    sims_to_target: int | None
    sims_used: int
    best_score: float
    best_values: dict[str, float]
    best_measured: dict[str, float]
    seconds: float
    failed_sims: int


@dataclass
class Aggregate:
    spec: str
    optimizer: str
    seeds: int
    success_rate: float
    median_sims_to_target: float | None
    mean_best_score: float
    seconds: float


@dataclass
class BenchReport:
    pdk: str
    calibrated: bool
    budget: int
    seeds: int
    cl: float
    created: str
    ngspice: str
    host: str
    runs: list[RunResult] = field(default_factory=list)
    aggregates: list[Aggregate] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(
            {
                "pdk": self.pdk,
                "calibrated": self.calibrated,
                "budget": self.budget,
                "seeds": self.seeds,
                "cl": self.cl,
                "created": self.created,
                "ngspice": self.ngspice,
                "host": self.host,
                "runs": [asdict(r) for r in self.runs],
                "aggregates": [asdict(a) for a in self.aggregates],
            },
            indent=2,
        )


def run_one(
    spec_name: str,
    optimizer_name: str,
    seed: int,
    *,
    pdk_name: str = "dev180",
    budget: int = 200,
    cl: float = 1e-12,
    optimizer_kwargs: dict | None = None,
) -> tuple[RunResult, Evaluator]:
    spec, topo_name = spec_registry.get(spec_name)
    topo = topo_registry.get(topo_name)
    process = (
        pdk_registry.DEV180
        if pdk_name == "dev180"
        else pdk_registry.get(pdk_name)
    )
    tb = Testbench(cl=cl)

    ev = Evaluator(topo, process, spec, tb, budget=budget)
    optimizer = opt_registry.get(optimizer_name, **(optimizer_kwargs or {}))

    t0 = time.perf_counter()
    optimizer.run(ev, np.random.default_rng(seed))
    elapsed = time.perf_counter() - t0

    best = ev.best
    result = RunResult(
        spec=spec_name,
        optimizer=optimizer_name,
        seed=seed,
        solved=ev.solved() is not None,
        sims_to_target=ev.sims_to_target(),
        sims_used=ev.used,
        best_score=best.score if best else float("inf"),
        best_values=best.values if best else {},
        best_measured=best.measured if best else {},
        seconds=elapsed,
        failed_sims=sum(1 for e in ev.history if not e.cached and not e.ok),
    )
    return result, ev


def run_matrix(
    spec_names: list[str],
    optimizer_names: list[str],
    *,
    seeds: int = 5,
    pdk_name: str = "dev180",
    budget: int = 200,
    cl: float = 1e-12,
    progress: bool = True,
) -> BenchReport:
    process = (
        pdk_registry.DEV180
        if pdk_name == "dev180"
        else pdk_registry.get(pdk_name)
    )
    report = BenchReport(
        pdk=process.name,
        calibrated=process.calibrated,
        budget=budget,
        seeds=seeds,
        cl=cl,
        created=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        ngspice=_ngspice_version(),
        host=f"{platform.system()} {platform.machine()} py{platform.python_version()}",
    )

    for spec_name in spec_names:
        for optimizer_name in optimizer_names:
            runs = []
            for seed in range(seeds):
                if progress:
                    print(
                        f"  {spec_name} / {optimizer_name} / seed {seed} ...",
                        end="",
                        flush=True,
                    )
                result, _ = run_one(
                    spec_name,
                    optimizer_name,
                    seed,
                    pdk_name=pdk_name,
                    budget=budget,
                    cl=cl,
                )
                runs.append(result)
                report.runs.append(result)
                if progress:
                    status = (
                        f"solved at {result.sims_to_target}"
                        if result.solved
                        else f"unsolved (score {result.best_score:.3f})"
                    )
                    print(f" {status}")

            hits = [r.sims_to_target for r in runs if r.sims_to_target is not None]
            report.aggregates.append(
                Aggregate(
                    spec=spec_name,
                    optimizer=optimizer_name,
                    seeds=seeds,
                    success_rate=len(hits) / seeds,
                    median_sims_to_target=(
                        float(statistics.median(hits)) if hits else None
                    ),
                    mean_best_score=float(
                        statistics.fmean(r.best_score for r in runs)
                    ),
                    seconds=sum(r.seconds for r in runs),
                )
            )

    return report


def format_table(report: BenchReport) -> str:
    lines = []
    if not report.calibrated:
        lines.append(
            "!! PDK "
            f"'{report.pdk}' is not a calibrated process. These numbers are for "
            "harness development only and must not be published as results.\n"
        )

    header = (
        f"{'spec':<14} {'optimizer':<14} {'solved':>8} "
        f"{'med sims':>9} {'mean score':>11}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    for a in report.aggregates:
        med = (
            f"{a.median_sims_to_target:.0f}"
            if a.median_sims_to_target is not None
            else "-"
        )
        lines.append(
            f"{a.spec:<14} {a.optimizer:<14} "
            f"{a.success_rate * a.seeds:>3.0f}/{a.seeds:<4d} "
            f"{med:>9} {a.mean_best_score:>11.4f}"
        )

    lines.append(
        f"\nbudget={report.budget} sims/run  seeds={report.seeds}  "
        f"pdk={report.pdk}  CL={report.cl:.3g}F"
    )
    lines.append(f"{report.ngspice} on {report.host}")
    return "\n".join(lines)


def save(report: BenchReport, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report.to_json())
    return path


def _ngspice_version() -> str:
    try:
        out = subprocess.run(
            ["ngspice", "-v"], capture_output=True, text=True, timeout=10
        ).stdout
        for line in out.splitlines():
            if "ngspice" in line.lower():
                return line.strip().lstrip("* ").strip()
    except Exception:
        pass
    return "ngspice (version unknown)"
