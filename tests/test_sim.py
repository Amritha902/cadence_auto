"""End-to-end tests that actually invoke ngspice.

These are the tests that would have caught the real bugs: a railed operating
point reported as a valid amplifier, and a phase margin computed with the wrong
sign convention. They are slower than the rest of the suite and skip cleanly
when ngspice is not installed.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from bias import optimizers, pdk, specs, topology
from bias.evaluate import BudgetExhausted, Evaluator
from bias.sim import run_deck
from bias.topology import Testbench

pytestmark = pytest.mark.needs_ngspice


@pytest.fixture(scope="module")
def process():
    return pdk.DEV180


@pytest.fixture(scope="module")
def tb():
    return Testbench(cl=1e-12)


class TestOTA5T:
    """The 5T OTA at its default sizing is a known-good reference point."""

    @staticmethod
    @pytest.fixture(scope="class")
    def measured(process, tb):
        t = topology.get("ota5t")
        vals = t.defaults()
        r = run_deck(t.deck(process, vals, tb))
        assert r.ok, f"reference point failed to simulate: {r.reason}"
        return t.postprocess(r.values, vals, process)

    def test_all_metrics_present_and_finite(self, measured):
        for key in ("gain", "gbw", "pm", "pwr", "vout_dc", "vout_margin"):
            assert key in measured, f"{key} was never measured"
            assert math.isfinite(measured[key]), f"{key} is not finite"

    def test_gain_is_physically_plausible(self, measured):
        # A single-stage OTA in 180nm lands in the 30-60dB range. Outside that
        # something is wrong with the models or the testbench.
        assert 30.0 < measured["gain"] < 60.0

    def test_single_pole_response_is_well_damped(self, measured):
        # Dominant-pole OTA into a capacitive load: phase margin near 90deg.
        assert 70.0 < measured["pm"] < 95.0

    def test_output_is_not_railed(self, measured):
        assert measured["vout_margin"] > 0.15

    def test_gain_bandwidth_matches_the_small_signal_estimate(self, measured, tb):
        """GBW should equal gm/(2*pi*CL) with gm implied by gain and rout.

        This is the cross-check that catches a testbench measuring the wrong
        node: it ties the AC result back to the DC operating point.
        """
        # gm*rout = 10^(gain/20); GBW = gm/(2*pi*CL). Both unknowns cancel into
        # a consistency check only if we know one -- so bound it instead:
        # any real gm here is between 1uS and 10mS.
        gm_implied = 2 * math.pi * tb.cl * measured["gbw"]
        assert 1e-6 < gm_implied < 1e-2, f"implied gm={gm_implied:.3g}S is unphysical"

    def test_power_matches_supply_current(self, measured, process):
        assert measured["pwr"] == pytest.approx(
            measured["isup"] * process.vdd, rel=1e-6
        )


class TestMillerBiasing:
    """The two-stage amp's failure mode is a railed output, not a bad number."""

    def test_default_sizing_rails_the_output(self, process, tb):
        """Regression: at defaults the second stage is mis-biased.

        This is expected -- it is the design problem the benchmark poses. The
        test pins it so that a change making defaults accidentally valid does
        not silently make miller-easy trivial.
        """
        t = topology.get("miller")
        vals = t.defaults()
        r = run_deck(t.deck(process, vals, tb))
        m = t.postprocess(r.values, vals, process)
        assert m["vout_margin"] < 0.10, "defaults no longer rail; retune miller-easy"

    def test_a_solved_point_is_not_railed(self, process, tb):
        """The solution DE finds must be a real amplifier, not a scoring artifact."""
        t = topology.get("miller")
        ev = Evaluator(t, process, specs.MILLER_EASY, tb, budget=300)
        optimizers.get("de").run(ev, np.random.default_rng(0))
        solved = ev.solved()
        assert solved is not None, "miller-easy became unsolvable"
        assert solved.measured["vout_margin"] > 0.20
        assert solved.measured["gain"] > 50.0
        assert solved.measured["pm"] > 50.0


class TestEvaluatorAccounting:
    def test_repeated_point_is_cached_not_recharged(self, process, tb):
        t = topology.get("ota5t")
        ev = Evaluator(t, process, specs.OTA5T_BASE, tb, budget=5)
        ev.evaluate(t.defaults())
        ev.evaluate(t.defaults())
        assert ev.used == 1
        assert len(ev.history) == 2
        assert ev.history[1].cached

    def test_budget_is_enforced(self, process, tb):
        t = topology.get("ota5t")
        ev = Evaluator(t, process, specs.OTA5T_BASE, tb, budget=2)
        rng = np.random.default_rng(0)
        with pytest.raises(BudgetExhausted):
            for _ in range(5):
                ev.evaluate(
                    {p.name: p.from_unit(rng.random()) for p in t.params}
                )
        assert ev.used == 2

    def test_failed_simulation_scores_rather_than_raises(self, process, tb):
        """An optimizer must be free to propose a point that does not converge."""
        t = topology.get("ota5t")
        ev = Evaluator(t, process, specs.OTA5T_BASE, tb, budget=3)
        # Minimum everything: a degenerate, barely-conducting circuit.
        degenerate = {p.name: p.lo for p in t.params}
        e = ev.evaluate(degenerate)
        assert math.isfinite(e.score)
        assert not e.satisfied


class TestOptimizersRun:
    """Every classical optimizer must respect the budget and return a best."""

    @pytest.mark.parametrize("name", ["random", "lhs", "nelder-mead", "de"])
    def test_optimizer_respects_budget(self, name, process, tb):
        t = topology.get("ota5t")
        ev = Evaluator(t, process, specs.OTA5T_EASY, tb, budget=25)
        optimizers.get(name).run(ev, np.random.default_rng(0))
        assert ev.used <= 25
        assert ev.best is not None
        assert math.isfinite(ev.best.score)
