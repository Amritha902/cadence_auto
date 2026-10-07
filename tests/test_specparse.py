"""Natural-language specification parsing. No simulator."""

from __future__ import annotations

import pytest

from bias import design, specparse


class TestQuantities:
    def test_mega_and_milli_are_distinguished_by_case(self):
        """'10 MHz' and '10 mW' differ only by case; lowercasing loses six
        orders of magnitude."""
        mhz = specparse.quantities("10 MHz")[0]
        mw = specparse.quantities("10 mW")[0]
        assert mhz.value == pytest.approx(10e6)
        assert mw.value == pytest.approx(10e-3)

    @pytest.mark.parametrize("text,expected", [
        ("1pF", 1e-12), ("2.2 pF", 2.2e-12), ("100uW", 100e-6),
        ("100 µW", 100e-6), ("5 GHz", 5e9), ("1.5kHz", 1500.0),
        ("20 MHz", 20e6), ("3.3 V", 3.3),
    ])
    def test_engineering_prefixes(self, text, expected):
        assert specparse.quantities(text)[0].value == pytest.approx(expected)

    def test_db_and_degrees_ignore_prefixes(self):
        assert specparse.quantities("60 dB")[0].value == pytest.approx(60.0)
        assert specparse.quantities("55 degrees")[0].value == pytest.approx(55.0)

    def test_scientific_notation(self):
        assert specparse.quantities("1e6 Hz")[0].value == pytest.approx(1e6)


class TestParse:
    def test_basic_request(self):
        p = specparse.parse("an op-amp with 60dB gain and 10MHz bandwidth")
        assert p.found["gain"] == pytest.approx(60.0)
        assert p.found["gbw"] == pytest.approx(10e6)

    def test_reversed_word_order_parses_the_same(self):
        a = specparse.parse("60dB gain, 10MHz bandwidth")
        b = specparse.parse("gain of 60 dB and a bandwidth of 10 MHz")
        assert a.found == b.found

    def test_units_alone_are_enough(self):
        """'under 100uW' names no metric, but watts can only be power."""
        p = specparse.parse("45 dB gain, 20 MHz, under 100uW")
        assert p.found["pwr"] == pytest.approx(100e-6)
        assert not p.unmatched

    def test_load_is_a_condition_not_a_target(self):
        p = specparse.parse("50dB gain driving a 2pF load")
        assert p.cl == pytest.approx(2e-12)
        assert "cl" not in p.found
        assert all(m.name != "cl" for m in p.spec.metrics)

    def test_directions_are_right(self):
        p = specparse.parse("60dB gain, 10MHz, under 100uW")
        by_name = {m.name: m for m in p.spec.metrics}
        assert by_name["gain"].direction == "max"     # a floor
        assert by_name["gbw"].direction == "max"
        assert by_name["pwr"].direction == "min"      # a ceiling

    def test_unstated_stability_is_implied(self):
        """An amplifier with no stated phase margin still has to be stable."""
        p = specparse.parse("60dB gain, 10MHz bandwidth")
        names = {m.name for m in p.spec.metrics}
        assert "pm" in names and "vout_margin" in names
        assert "pm" in p.implied and "vout_margin" in p.implied

    def test_stated_phase_margin_overrides_the_implied_one(self):
        p = specparse.parse("60dB gain, 10MHz, phase margin 70 degrees")
        by_name = {m.name: m for m in p.spec.metrics}
        assert by_name["pm"].target == pytest.approx(70.0)
        assert "pm" not in p.implied

    def test_terse_comma_form(self):
        p = specparse.parse("op-amp, 60dB, 10MHz, 80uW, 1pF")
        assert p.found["gain"] == pytest.approx(60.0)
        assert p.found["gbw"] == pytest.approx(10e6)
        assert p.found["pwr"] == pytest.approx(80e-6)
        assert p.cl == pytest.approx(1e-12)

    def test_no_targets_raises_rather_than_returning_an_empty_spec(self):
        """An empty spec would let the optimizer 'succeed' against nothing."""
        with pytest.raises(ValueError, match="no design targets"):
            specparse.parse("build me something nice")

    def test_strict_mode_rejects_unplaced_values(self):
        with pytest.raises(ValueError):
            specparse.parse("60dB gain and 5 volts somewhere", strict=True)


class TestTopologyChoice:
    def test_low_gain_uses_the_single_stage(self):
        name, why = design.choose_topology(40.0)
        assert name == "ota5t"
        assert "ceiling" in why

    def test_high_gain_needs_two_stages(self):
        name, why = design.choose_topology(70.0)
        assert name == "miller"

    def test_threshold_is_the_measured_ceiling(self):
        """The boundary comes from calibration, not a rule of thumb."""
        ceiling = design.OTA5T_GAIN_CEILING_DB
        assert design.choose_topology(ceiling - 0.1)[0] == "ota5t"
        assert design.choose_topology(ceiling + 0.1)[0] == "miller"

    def test_no_gain_target_defaults_to_the_cheaper_topology(self):
        assert design.choose_topology(None)[0] == "ota5t"


@pytest.mark.needs_ngspice
class TestAnalogBuild:
    def test_easy_spec_is_sized_and_verified(self, tmp_path):
        from bias import pdk

        result = design.size(
            "an op-amp with 40dB gain and 5MHz bandwidth under 200uW "
            "driving a 1pF load",
            pdk.DEV180, budget=400, outdir=tmp_path,
        )
        assert result.solved, result.describe()
        assert result.parsed.spec.satisfied(result.measured)
        for artifact in ("testbench", "symbol", "report", "bode"):
            assert result.files[artifact].exists(), f"{artifact} not written"

    def test_stops_as_soon_as_the_spec_is_met(self):
        """Building should not spend the rest of the budget once it is done."""
        from bias import pdk

        result = design.size(
            "op-amp with 40dB gain, 2MHz bandwidth, under 500uW, 1pF load",
            pdk.DEV180, budget=400,
        )
        assert result.solved
        assert result.sims_used == result.sims_to_target

    def test_impossible_gain_is_refused_without_burning_the_budget(self):
        from bias import pdk

        result = design.size("op-amp with 140dB gain and 1MHz bandwidth",
                             pdk.DEV180, budget=400)
        assert not result.solved
        assert result.infeasible
        assert result.sims_used == 0

    def test_bode_plot_is_well_formed(self):
        from bias import pdk, plot, topology
        from bias.topology import Testbench

        topo = topology.get("ota5t")
        svg = plot.bode(topo, pdk.DEV180, topo.defaults(), Testbench())
        assert svg and svg.startswith("<svg") and svg.rstrip().endswith("</svg>")
        assert svg.count("<polyline") == 2      # magnitude and phase

    def test_sweep_returns_real_points(self):
        from bias import pdk, plot, topology
        from bias.topology import Testbench

        topo = topology.get("ota5t")
        rows = plot.sweep(topo, pdk.DEV180, topo.defaults(), Testbench())
        assert len(rows) > 100
        freqs = [r[0] for r in rows]
        assert freqs == sorted(freqs)
        # DC gain at the low end should match what the sizing loop measures.
        assert 30.0 < rows[0][1] < 60.0

    def test_unified_build_routes_digital_and_analog(self, tmp_path):
        from bias import pdk

        digital = design.build_any("half adder", pdk.DEV180, use_llm=False)
        assert digital.cell == "half_adder"

        analog = design.build_any(
            "op-amp with 40dB gain, 2MHz bandwidth, under 500uW, 1pF load",
            pdk.DEV180, use_llm=False, budget=300,
        )
        assert analog.topology_name in ("ota5t", "miller")

    def test_unresolvable_request_names_both_menus(self):
        from bias import pdk

        with pytest.raises(LookupError, match="half_adder"):
            design.build_any("build me a spaceship", pdk.DEV180, use_llm=False)


@pytest.mark.needs_ngspice
class TestTopologyFallback:
    """The single-stage threshold is a measured number from one process.

    It can be wrong on another, so size() escalates to two stages rather than
    reporting a near miss. These force that path: the normal thresholds happen
    to be right on both PDKs shipped here, so it would otherwise never run.
    """

    def test_falls_back_to_two_stages_when_the_threshold_is_wrong(
        self, monkeypatch
    ):
        from bias import pdk

        # Pretend the single stage can reach 200dB, so it is chosen for a
        # target it cannot possibly meet.
        monkeypatch.setattr(design, "OTA5T_GAIN_CEILING_DB", 200.0)

        result = design.size(
            "op-amp with 65dB gain, 2MHz bandwidth, under 400uW, 1pF load",
            pdk.DEV180, budget=250, attempts=1,
        )
        assert result.topology_name == "miller", (
            "should have escalated after the single stage fell short"
        )
        assert "reached only" in result.why_topology

    def test_no_fallback_when_the_single_stage_succeeds(self):
        from bias import pdk

        result = design.size(
            "op-amp with 40dB gain, 2MHz bandwidth, under 400uW, 1pF load",
            pdk.DEV180, budget=300, attempts=1,
        )
        assert result.topology_name == "ota5t"
        assert "reached only" not in result.why_topology


class TestRouting:
    """Which half of the tool a request goes to.

    This exists because of a real shipped bug. "and" was an alias for the AND
    gate, build_any resolved cell names before parsing specifications, and so
    "an op-amp with 40dB gain and 5MHz bandwidth" was built as a 6-transistor
    AND gate. The earlier collision test only checked digital phrasings, so it
    passed throughout.
    """

    @staticmethod
    def _route(text: str) -> str:
        from bias import specparse

        try:
            specparse.parse(text)
        except ValueError:
            return "digital"
        return "analog"

    @pytest.mark.parametrize("text", [
        "an op-amp with 40dB gain and 5MHz bandwidth under 200uW",
        "60dB gain and 10MHz bandwidth",
        "amplifier with 50dB gain and a 2pF load",
        "45dB gain or better, 10MHz bandwidth",
    ])
    def test_analog_prose_containing_and_or_stays_analog(self, text):
        assert self._route(text) == "analog"

    @pytest.mark.parametrize("text", [
        "a half adder", "an and gate", "a NAND gate", "an or gate",
        "a 2:1 multiplexer", "an inverter", "a 2 bit adder",
    ])
    def test_logic_requests_stay_digital(self, text):
        assert self._route(text) == "digital"

    def test_bare_and_or_are_not_aliases(self):
        """They are ordinary English; only the explicit gate names resolve."""
        assert "and" not in design.ALIASES
        assert "or" not in design.ALIASES
        assert design.ALIASES["and gate"] == "and2"
        assert design.ALIASES["or gate"] == "or2"

    @pytest.mark.needs_ngspice
    def test_build_any_routes_an_amplifier_to_the_analog_path(self):
        from bias import pdk

        result = design.build_any(
            "an op-amp with 40dB gain and 5MHz bandwidth under 300uW, 1pF load",
            pdk.DEV180, use_llm=False, budget=250, attempts=1,
        )
        assert hasattr(result, "topology_name"), "routed to the logic path"
        assert result.topology_name in ("ota5t", "miller")
