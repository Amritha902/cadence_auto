"""Real foundry models.

Everything else in the suite runs on dev180, which is a hand-written card and
not a real process. These tests are the ones that prove the benchmark works on
silicon models a fab would recognise, and they skip cleanly when the PDK has
not been fetched.
"""

from __future__ import annotations

import pytest

from bias import logic, pdk, sim, topology, verify
from bias.topology import Testbench

pytestmark = [pytest.mark.needs_ngspice, pytest.mark.needs_sky130]


@pytest.fixture(scope="module")
def sky():
    return pdk.get("sky130")


class TestInstantiation:
    def test_devices_are_subcircuits_not_models(self, sky):
        """sky130 primitives are .subckt wrappers; an M instance would be
        silently wrong rather than an error."""
        assert sky.device_style == "subckt"
        line = sky.mos_line("M1", "d", "g", "s", "b", "nmos", 1e-6, 0.15e-6)
        assert line.startswith("X1 ")
        assert "sky130_fd_pr__nfet_01v8" in line

    def test_parameters_are_lowercase_for_subckt(self, sky):
        """The .subckt declares `l`, `w`, `ad`...; the wrapper passes them on."""
        line = sky.mos_line("M1", "d", "g", "s", "b", "pmos", 2e-6, 0.2e-6)
        assert " w=" in line and " l=" in line
        assert " W=" not in line and " L=" not in line

    def test_diffusion_geometry_is_emitted(self, sky):
        line = sky.mos_line("M1", "d", "g", "s", "b", "nmos", 1e-6, 0.15e-6)
        for token in ("ad=", "as=", "pd=", "ps="):
            assert token in line

    def test_dev180_still_uses_model_instances(self):
        """The hand-written card must not be disturbed by the subckt path."""
        line = pdk.DEV180.mos_line("M1", "d", "g", "s", "b", "nmos",
                                   1e-6, 0.18e-6)
        assert line.startswith("M1 ") and " W=" in line


class TestAnalogOnSky130:
    @staticmethod
    @pytest.fixture(scope="class")
    def measured(sky):
        t = topology.get("ota5t")
        vals = t.defaults()
        r = sim.run_deck(t.deck(sky, vals, Testbench(cl=1e-12)), timeout=300)
        assert r.ok, f"sky130 reference point failed: {r.reason}"
        return t.postprocess(r.values, vals, sky)

    def test_reference_point_simulates(self, measured):
        for key in ("gain", "gbw", "pm", "pwr", "vout_margin"):
            assert key in measured

    def test_gain_is_plausible_for_130nm(self, measured):
        # A single-stage OTA on a real 130nm process sits well below the
        # 180nm figure -- short-channel output resistance is worse.
        assert 30.0 < measured["gain"] < 50.0

    def test_response_is_well_damped(self, measured):
        assert 70.0 < measured["pm"] < 95.0

    def test_output_is_near_mid_rail(self, measured):
        assert measured["vout_margin"] > 0.2

    def test_both_topologies_simulate(self, sky):
        for name in ("ota5t", "miller"):
            t = topology.get(name)
            r = sim.run_deck(t.deck(sky, t.defaults(), Testbench(cl=1e-12)),
                             timeout=300)
            assert r.ok, f"{name} failed on sky130: {r.reason}"


class TestLogicOnSky130:
    @pytest.mark.parametrize("name", sorted(logic.CELLS))
    def test_cell_matches_its_truth_table(self, name, sky):
        report = verify.verify(name, sky)
        assert report.passed, f"{name} on sky130:\n{report.table()}\n{report.reason}"

    def test_outputs_reach_their_rails(self, sky):
        report = verify.verify("half_adder", sky)
        assert report.worst_rail_error < 0.05


def test_sky130_is_marked_calibrated(sky):
    """Unlike dev180, results from this PDK are reportable."""
    assert sky.calibrated
