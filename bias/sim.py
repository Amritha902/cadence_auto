"""ngspice driver.

One job: take a complete deck, run it, and return the scalars it printed.
Every measurement the benchmark uses comes back through `SimResult.values`.

Decks talk to us through ngspice's `.control` block rather than `.measure`,
because the control language lets a topology compute derived quantities
(power from supply current, phase margin at the measured crossover) inside the
simulator instead of in Python. Everything printed as `name = value` is
collected; topologies agree to print exactly the metric names their Spec uses.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

NGSPICE = shutil.which("ngspice")

# ngspice prints scalars from `print` and `meas` in a few shapes:
#   gain = 4.512340e+01
#   gbw =  1.234e+07
#   a0 = 4.5123e+01 at = 1.0000e+01      <- meas ... find ... at ...
# Grab the first numeric token after the '=' following a bare identifier.
_SCALAR = re.compile(
    r"^\s*([A-Za-z_]\w*)\s*=\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)",
    re.MULTILINE,
)
# ngspice reports a failed measurement without aborting the run.
_FAILED_MEAS = re.compile(r"^\s*([A-Za-z_]\w*)\s*=\s*failed", re.MULTILINE | re.IGNORECASE)


class NgspiceNotFound(RuntimeError):
    pass


@dataclass
class SimResult:
    ok: bool
    values: dict[str, float] = field(default_factory=dict)
    stdout: str = ""
    stderr: str = ""
    returncode: int = 0
    # Set when the deck ran but produced no usable measurements.
    reason: str = ""

    def get(self, key: str) -> float:
        return self.values.get(key, float("nan"))


def run_deck(deck: str, timeout: float = 30.0, keep_dir: Path | None = None) -> SimResult:
    """Run one netlist in batch mode and harvest its printed scalars.

    A simulator crash, a convergence failure and a topology that simply misses
    its target all come back the same way: ok=False or missing values. Callers
    score them, they do not raise. An optimizer must be free to propose a
    design that does not converge.
    """
    if NGSPICE is None:
        raise NgspiceNotFound(
            "ngspice is not on PATH. Install it (brew install ngspice / "
            "apt install ngspice) -- the benchmark cannot run without it."
        )

    workdir = keep_dir or Path(tempfile.mkdtemp(prefix="bias-"))
    workdir.mkdir(parents=True, exist_ok=True)
    deck_path = workdir / "deck.sp"
    deck_path.write_text(deck)

    try:
        proc = subprocess.run(
            [NGSPICE, "-b", str(deck_path)],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=workdir,
        )
    except subprocess.TimeoutExpired:
        return SimResult(ok=False, reason=f"ngspice timed out after {timeout}s")
    finally:
        if keep_dir is None:
            shutil.rmtree(workdir, ignore_errors=True)

    out = proc.stdout + "\n" + proc.stderr
    failed = set(_FAILED_MEAS.findall(out))
    values: dict[str, float] = {}
    for name, raw in _SCALAR.findall(out):
        if name in failed:
            continue
        try:
            values[name] = float(raw)
        except ValueError:
            continue

    for name in failed:
        values.setdefault(name, float("nan"))

    ok = bool(values) and proc.returncode == 0
    reason = ""
    if not values:
        reason = "no scalars printed; deck likely failed to converge"
    elif proc.returncode != 0:
        reason = f"ngspice exited {proc.returncode}"

    return SimResult(
        ok=ok,
        values=values,
        stdout=proc.stdout,
        stderr=proc.stderr,
        returncode=proc.returncode,
        reason=reason,
    )
