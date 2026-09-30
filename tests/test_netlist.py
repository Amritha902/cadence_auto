"""Circuit IR: validation and structural identity. No simulator."""

from __future__ import annotations

import pytest

from bias.netlist import GROUND, Circuit, Device, Kind


def _mos(name, d, g, s, b, kind=Kind.NMOS, w=1e-6, l=0.18e-6):
    return Device(name, kind, {"d": d, "g": g, "s": s, "b": b}, {"W": w, "L": l})


def _inverter() -> Circuit:
    return Circuit(
        "inv",
        [
            _mos("Mn1", "y", "a", GROUND, GROUND, Kind.NMOS),
            _mos("Mp1", "y", "a", "vdd", "vdd", Kind.PMOS),
        ],
        ports=["a", "y", "vdd"],
    )


class TestValidation:
    def test_good_circuit_is_valid(self):
        assert _inverter().is_valid()

    def test_missing_terminal_is_caught(self):
        c = _inverter()
        del c.devices[0].nets["b"]
        problems = [str(p) for p in c.validate()]
        assert any("missing terminal" in p for p in problems)

    def test_unknown_terminal_is_caught(self):
        c = _inverter()
        c.devices[0].nets["drain"] = "y"
        assert any("unknown terminal" in str(p) for p in c.validate())

    def test_transistor_without_geometry_is_caught(self):
        c = _inverter()
        del c.devices[0].params["W"]
        assert any("has no W" in str(p) for p in c.validate())

    def test_floating_net_is_caught(self):
        """A net touched once has nowhere for current to go."""
        c = _inverter()
        c.devices.append(_mos("Mn2", "dangle", "a", GROUND, GROUND))
        assert any("floating" in str(p) for p in c.validate())

    def test_duplicate_device_name_is_caught(self):
        c = _inverter()
        c.devices.append(_mos("Mn1", "y", "a", GROUND, GROUND))
        assert any("duplicate" in str(p) for p in c.validate())

    def test_shorted_passive_is_caught(self):
        c = _inverter()
        c.devices.append(Device("R1", Kind.RES, {"p": "y", "n": "y"}, {"value": 1e3}))
        assert any("shorted" in str(p) for p in c.validate())

    def test_drain_shorted_to_source_is_caught(self):
        c = _inverter()
        c.devices.append(_mos("Mn9", "y", "a", "y", GROUND))
        assert any("cannot conduct" in str(p) for p in c.validate())

    def test_missing_ground_is_caught(self):
        c = Circuit("x", [_mos("Mn1", "y", "a", "vs", "vs")], ports=["a", "y", "vs"])
        assert any("no ground" in str(p) for p in c.validate())

    def test_island_with_no_path_to_ground_is_caught(self):
        """Each node looks connected, but the group floats as a whole."""
        c = _inverter()
        c.devices.append(Device("C1", Kind.CAP, {"p": "iso1", "n": "iso2"},
                                {"value": 1e-12}))
        c.devices.append(Device("C2", Kind.CAP, {"p": "iso2", "n": "iso1"},
                                {"value": 1e-12}))
        problems = [str(p) for p in c.validate()]
        assert any("no path to ground" in p for p in problems)


class TestGraphHash:
    def test_identical_circuits_hash_equal(self):
        assert _inverter().graph_hash() == _inverter().graph_hash()

    def test_renaming_internal_nets_does_not_change_the_hash(self):
        a = Circuit("x", [
            _mos("Mn1", "mid", "in", GROUND, GROUND),
            _mos("Mn2", "out", "mid", GROUND, GROUND),
        ], ports=["in", "out"])
        b = Circuit("x", [
            _mos("Mn1", "zzz", "in", GROUND, GROUND),
            _mos("Mn2", "out", "zzz", GROUND, GROUND),
        ], ports=["in", "out"])
        assert a.graph_hash() == b.graph_hash()

    def test_renaming_devices_does_not_change_the_hash(self):
        a = _inverter()
        b = _inverter()
        b.devices[0].name = "Mn99"
        assert a.graph_hash() == b.graph_hash()

    def test_different_topologies_hash_differently(self):
        from bias import logic

        hashes = {
            name: logic.get(name)[0]().graph_hash() for name in logic.CELLS
        }
        assert len(set(hashes.values())) == len(hashes), (
            f"two cells collide: {hashes}"
        )

    def test_swapping_a_supply_for_a_signal_net_changes_the_hash(self):
        """VDD is not interchangeable with an internal node."""
        a = _inverter()
        b = _inverter()
        b.devices[1].nets["s"] = "y"
        assert a.graph_hash() != b.graph_hash()


class TestEmission:
    def test_mosfet_line_uses_pdk_model_names(self):
        spice = _inverter().to_spice({"nmos": "nch_lvt", "pmos": "pch_lvt"})
        assert "nch_lvt" in spice and "pch_lvt" in spice

    def test_terminal_order_is_d_g_s_b(self):
        spice = _inverter().to_spice({"nmos": "nch"})
        line = [l for l in spice.splitlines() if l.startswith("Mn1")][0]
        assert line.split()[1:5] == ["y", "a", GROUND, GROUND]

    def test_instance_prefix_is_applied_when_absent(self):
        d = _mos("1", "y", "a", GROUND, GROUND)
        assert d.instance == "M1"

    def test_passive_value_is_emitted(self):
        c = Circuit("x", [
            Device("R1", Kind.RES, {"p": "a", "n": GROUND}, {"value": 2.2e3}),
            Device("R2", Kind.RES, {"p": "a", "n": GROUND}, {"value": 1e3}),
        ], ports=["a"])
        assert "2200" in c.to_spice()


def test_unknown_device_lookup_raises():
    with pytest.raises(KeyError):
        _inverter().device("nope")
