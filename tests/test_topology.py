"""Parameter mapping, deck generation and area accounting. No simulator."""

from __future__ import annotations

import re

import pytest

from bias import pdk, topology
from bias.topology import Param
from bias.topology import Testbench as Testbench  # noqa: PLC0414

# Renamed locally so pytest does not try to collect the dataclass as a test class.
Testbench.__test__ = False

_GEOM = re.compile(r"^(M\w+)\b.*?\bW=([\d.eE+-]+).*?\bL=([\d.eE+-]+)", re.MULTILINE)


def _device_geometry(deck: str) -> list[tuple[str, float, float]]:
    """Every MOSFET's (instance, W, L) as actually written into the netlist."""
    return [(m[0], float(m[1]), float(m[2])) for m in _GEOM.findall(deck)]


class TestParam:
    def test_unit_roundtrip_log(self):
        p = Param("w", 1e-6, 100e-6, "m", log=True)
        for u in (0.0, 0.25, 0.5, 0.75, 1.0):
            assert p.to_unit(p.from_unit(u)) == pytest.approx(u, abs=1e-9)

    def test_unit_roundtrip_linear(self):
        p = Param("rz", 10.0, 1000.0, "ohm", log=False)
        for u in (0.0, 0.3, 1.0):
            assert p.to_unit(p.from_unit(u)) == pytest.approx(u, abs=1e-9)

    def test_log_midpoint_is_geometric(self):
        p = Param("w", 1e-6, 100e-6, "m", log=True)
        assert p.from_unit(0.5) == pytest.approx(10e-6)

    def test_clip_and_out_of_range_unit(self):
        p = Param("w", 1e-6, 100e-6, "m")
        assert p.clip(1e-9) == 1e-6
        assert p.clip(1.0) == 100e-6
        assert p.from_unit(-5.0) == pytest.approx(p.lo)
        assert p.from_unit(5.0) == pytest.approx(p.hi)


@pytest.mark.parametrize("name", ["ota5t", "miller"])
class TestDecks:
    def test_defaults_are_inside_bounds(self, name):
        t = topology.get(name)
        for k, v in t.defaults().items():
            lo, hi = t.bounds()[k]
            assert lo <= v <= hi

    def test_deck_contains_required_pieces(self, name):
        t = topology.get(name)
        deck = t.deck(pdk.DEV180, t.defaults(), Testbench())
        assert ".control" in deck and ".endc" in deck
        # Every measurement the postprocessor needs must be requested.
        for token in ("ph_dc", "ph_fu", "gbw", "gain", "pwr", "vout_dc"):
            assert token in deck, f"{name} deck never measures {token}"

    def test_every_device_has_diffusion_geometry(self, name):
        """Missing AD/AS/PD/PS silently zeroes junction capacitance."""
        t = topology.get(name)
        deck = t.deck(pdk.DEV180, t.defaults(), Testbench())
        for line in deck.splitlines():
            if line.startswith("M"):
                for token in ("AD=", "AS=", "PD=", "PS="):
                    assert token in line, f"{line!r} missing {token}"

    def test_out_of_range_values_are_clipped_not_emitted(self, name):
        """An optimizer proposing an absurd size must not reach the netlist."""
        t = topology.get(name)
        wild = {p.name: p.hi * 1000 for p in t.params}
        deck = t.deck(pdk.DEV180, wild, Testbench())

        emitted = _device_geometry(deck)
        assert emitted, "no MOSFETs found in deck"
        wmax = max(p.hi for p in t.params if p.name.startswith("w"))
        lmax = max(p.hi for p in t.params if p.name.startswith("l"))
        for inst, w, l in emitted:
            assert w <= wmax * 1.001, f"{inst} emitted W={w:g} above bound {wmax:g}"
            assert l <= lmax * 1.001, f"{inst} emitted L={l:g} above bound {lmax:g}"

    def test_area_is_positive_and_grows_with_width(self, name):
        t = topology.get(name)
        base = t.defaults()
        assert t.area(base) > 0
        wider = dict(base)
        wider["w1"] = base["w1"] * 2
        assert t.area(wider) > t.area(base)


class TestPostprocess:
    def test_vout_margin_detects_railed_output(self):
        t = topology.get("miller")
        p = pdk.DEV180
        vals = t.defaults()
        railed = t.postprocess({"vout_dc": 1.74}, vals, p)
        centred = t.postprocess({"vout_dc": 0.9}, vals, p)
        assert railed["vout_margin"] < 0.05
        assert centred["vout_margin"] == pytest.approx(0.5)

    def test_miller_area_charges_for_the_compensation_cap(self):
        t = topology.get("miller")
        small = dict(t.defaults(), cc=10e-15)
        large = dict(t.defaults(), cc=10e-12)
        assert t.area(large) > t.area(small)


def test_registry_names_match_keys():
    for name, t in topology.REGISTRY.items():
        assert t.name == name


def test_unknown_topology_raises():
    with pytest.raises(KeyError):
        topology.get("does-not-exist")
