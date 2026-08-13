"""Scoring and phase-margin tests. No simulator required."""

from __future__ import annotations

import math

import pytest

from bias.spec import BAD_POINT_SHORTFALL, Metric, Spec
from bias.topology import phase_margin


class TestMetric:
    def test_max_metric_met(self):
        m = Metric("gain", "max", 50.0, "dB")
        assert m.satisfied(55.0)
        assert m.shortfall(55.0) == 0.0

    def test_max_metric_missed(self):
        m = Metric("gain", "max", 50.0, "dB")
        assert not m.satisfied(45.0)
        assert m.shortfall(45.0) == pytest.approx(0.1)

    def test_min_metric_is_a_ceiling(self):
        m = Metric("pwr", "min", 100e-6, "W")
        assert m.satisfied(50e-6)
        assert m.shortfall(50e-6) == 0.0
        assert not m.satisfied(200e-6)
        assert m.shortfall(200e-6) > 0

    def test_log_domain_is_symmetric_in_ratio(self):
        """A 2x shortfall costs the same at 1MHz as at 1GHz."""
        a = Metric("gbw", "max", 1e6, "Hz").shortfall(0.5e6)
        b = Metric("gbw", "max", 1e9, "Hz").shortfall(0.5e9)
        assert a == pytest.approx(b)
        assert a == pytest.approx(math.log10(2.0))

    def test_linear_domain_for_db_and_degrees(self):
        assert Metric("pm", "max", 60.0, "deg").shortfall(30.0) == pytest.approx(0.5)

    def test_nan_is_a_bounded_miss_not_an_exception(self):
        """A failed simulation must be scoreable, not fatal."""
        m = Metric("gain", "max", 50.0, "dB")
        assert not m.satisfied(float("nan"))
        assert m.shortfall(float("nan")) == BAD_POINT_SHORTFALL
        assert math.isfinite(m.shortfall(float("inf")))

    def test_objective_keeps_rewarding_past_target(self):
        obj = Metric("area", "min", 100.0, "m2", objective=True)
        constraint = Metric("area", "min", 100.0, "m2")
        # Beating the target: objective goes negative, constraint stops at 0.
        assert obj.shortfall(50.0) < 0
        assert constraint.shortfall(50.0) == 0.0


class TestSpec:
    def _spec(self):
        return Spec(
            name="t",
            metrics=(
                Metric("gain", "max", 50.0, "dB"),
                Metric("pwr", "min", 100e-6, "W"),
            ),
        )

    def test_all_met_scores_zero_and_satisfies(self):
        s = self._spec()
        m = {"gain": 55.0, "pwr": 50e-6}
        assert s.score(m) == 0.0
        assert s.satisfied(m)

    def test_missing_metric_counts_as_failure(self):
        s = self._spec()
        assert not s.satisfied({"gain": 55.0})
        assert s.score({"gain": 55.0}) >= BAD_POINT_SHORTFALL

    def test_objectives_do_not_gate_satisfaction(self):
        s = Spec(
            name="t",
            metrics=(
                Metric("gain", "max", 50.0, "dB"),
                Metric("area", "min", 1.0, "m2", objective=True),
            ),
        )
        # Area is way over its target but it is only an objective.
        assert s.satisfied({"gain": 55.0, "area": 1000.0})

    def test_report_marks_pass_and_fail(self):
        s = self._spec()
        text = s.report({"gain": 55.0, "pwr": 500e-6})
        assert "[PASS] gain" in text
        assert "[FAIL] pwr" in text


class TestPhaseMargin:
    """The bug this file exists to prevent: assuming every amp is non-inverting."""

    def test_non_inverting_single_pole(self):
        # 5T OTA: starts at 0 deg, lags 90 deg by crossover.
        assert phase_margin(0.0, -90.0) == pytest.approx(90.0)

    def test_non_inverting_measured_case(self):
        assert phase_margin(0.0, -97.48) == pytest.approx(82.52)

    def test_inverting_two_stage(self):
        # Miller amp: starts at 180 deg. Same 130 deg of lag.
        assert phase_margin(180.0, 50.85) == pytest.approx(50.85)

    def test_inverting_is_branch_independent(self):
        """ngspice may report the DC phase as +180 or -180; both must agree."""
        assert phase_margin(180.0, 50.85) == pytest.approx(
            phase_margin(-180.0, 50.85)
        )

    def test_unstable_amplifier_gives_negative_margin(self):
        # 200 deg of lag from DC -> margin is negative.
        assert phase_margin(0.0, -200.0) == pytest.approx(-20.0)

    def test_nan_propagates(self):
        assert math.isnan(phase_margin(float("nan"), -90.0))
        assert math.isnan(phase_margin(0.0, float("nan")))
