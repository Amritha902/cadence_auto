"""The benchmark suite: fixed, named design targets.

These are the tasks optimizers are scored on. They are deliberately frozen --
changing a target changes the benchmark, so treat edits here the way you would
treat editing a test fixture that other people's published numbers depend on.

Each spec pairs with one topology. Difficulty tiers exist so the benchmark can
distinguish "can find any working point" from "can trade gain against power
under a real constraint".
"""

from __future__ import annotations

from .spec import Metric, Spec

# Every spec carries this. A railed output makes gain and GBW meaningless, so
# without it an optimizer can score well on a circuit that is not an amplifier.
_HEADROOM = Metric(
    "vout_margin", "max", 0.20, "",
    weight=2.0,
    # Weighted up: this is a validity condition, not a performance target.
)


# ---------------------------------------------------------------------------
# 5-transistor OTA
# ---------------------------------------------------------------------------

OTA5T_EASY = Spec(
    name="ota5t-easy",
    description="Modest single-stage OTA. Reachable from most starting points.",
    metrics=(
        Metric("gain", "max", 45.0, "dB"),
        Metric("gbw", "max", 10e6, "Hz"),
        Metric("pm", "max", 60.0, "deg"),
        Metric("pwr", "min", 100e-6, "W"),
        _HEADROOM,
    ),
)

OTA5T_BASE = Spec(
    name="ota5t-base",
    description=(
        "The reference single-stage task: gain and bandwidth together under a "
        "tight power ceiling. Requires trading input-pair size against current."
    ),
    metrics=(
        Metric("gain", "max", 50.0, "dB"),
        Metric("gbw", "max", 20e6, "Hz"),
        Metric("pm", "max", 60.0, "deg"),
        Metric("pwr", "min", 50e-6, "W"),
        _HEADROOM,
    ),
)

OTA5T_HARD = Spec(
    name="ota5t-hard",
    description=(
        "Aggressive bandwidth at low power, with area penalised. The easy "
        "directions conflict: current buys GBW but breaks the power budget, "
        "length buys gain but costs bandwidth and area."
    ),
    # Calibrated twice against the measured Pareto frontier, not guessed.
    #
    # v1 asked for 50MHz at 40uW. Into CL=1pF that needs gm=314uS from ~11uA
    # per branch -- gm/Id ~ 28, above the subthreshold limit. Unsatisfiable;
    # DE solved 0/3 at 800 sims.
    #
    # v2 asked for 52dB at 30MHz/60uW. Still 0/3, and the diagnosis was more
    # interesting: all three seeds converged to gain 49.1dB with phase margin
    # pinned at exactly 60deg. That is the topology's real ceiling here -- a
    # 5T OTA buys gain with channel length, length costs phase margin, and the
    # stability constraint binds before 52dB is reachable.
    #
    # The targets below sit just inside that measured frontier: demanding
    # enough that DE needs most of its budget, but physically achievable.
    metrics=(
        Metric("gain", "max", 48.0, "dB"),
        Metric("gbw", "max", 25e6, "Hz"),
        Metric("pm", "max", 60.0, "deg"),
        Metric("pwr", "min", 70e-6, "W"),
        Metric("area", "min", 500e-12, "m2", objective=True, weight=0.5),
        _HEADROOM,
    ),
)


# ---------------------------------------------------------------------------
# Two-stage Miller-compensated op-amp
# ---------------------------------------------------------------------------

MILLER_EASY = Spec(
    name="miller-easy",
    description=(
        "Get the two-stage amplifier biased and stable at all. The hard part "
        "is not performance -- it is making the second stage's current match "
        "so the output does not sit at a rail."
    ),
    metrics=(
        Metric("gain", "max", 55.0, "dB"),
        Metric("gbw", "max", 1e6, "Hz"),
        Metric("pm", "max", 55.0, "deg"),
        Metric("pwr", "min", 500e-6, "W"),
        _HEADROOM,
    ),
)

MILLER_BASE = Spec(
    name="miller-base",
    description=(
        "The reference two-stage task. High gain with a real stability "
        "constraint -- the compensation network has to track the sizing."
    ),
    metrics=(
        Metric("gain", "max", 70.0, "dB"),
        Metric("gbw", "max", 10e6, "Hz"),
        Metric("pm", "max", 60.0, "deg"),
        Metric("pwr", "min", 200e-6, "W"),
        _HEADROOM,
    ),
)

MILLER_HARD = Spec(
    name="miller-hard",
    description=(
        "High gain, high bandwidth, tight power, and area penalised. This is "
        "the task where naive parameter sweeps stop working."
    ),
    metrics=(
        Metric("gain", "max", 75.0, "dB"),
        Metric("gbw", "max", 25e6, "Hz"),
        Metric("pm", "max", 60.0, "deg"),
        Metric("pwr", "min", 150e-6, "W"),
        Metric("area", "min", 2000e-12, "m2", objective=True, weight=0.5),
        _HEADROOM,
    ),
)


# spec name -> (spec, topology name)
SUITE: dict[str, tuple[Spec, str]] = {
    OTA5T_EASY.name: (OTA5T_EASY, "ota5t"),
    OTA5T_BASE.name: (OTA5T_BASE, "ota5t"),
    OTA5T_HARD.name: (OTA5T_HARD, "ota5t"),
    MILLER_EASY.name: (MILLER_EASY, "miller"),
    MILLER_BASE.name: (MILLER_BASE, "miller"),
    MILLER_HARD.name: (MILLER_HARD, "miller"),
}


def get(name: str) -> tuple[Spec, str]:
    if name not in SUITE:
        raise KeyError(f"unknown spec {name!r}; have {sorted(SUITE)}")
    return SUITE[name]


def names() -> list[str]:
    return list(SUITE)
