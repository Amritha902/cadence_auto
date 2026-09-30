"""Circuit topologies and the decks they generate.

A Topology owns three things: the free parameters an optimizer may set, the
netlist it emits for a given set of values, and the measurements it promises to
print. Everything an optimizer is allowed to know about a circuit comes from
here -- which is what makes the LLM agent and the classical optimizers
comparable on identical footing.

Decks print their results through an ngspice `.control` block as `name = value`
lines. The names must match the Metric names in the Spec.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from .pdk import PDK


@dataclass(frozen=True)
class Param:
    """One free variable, with the range an optimizer may explore."""

    name: str
    lo: float
    hi: float
    unit: str = ""
    # Widths, currents and capacitances span decades; search them in log space.
    log: bool = True
    description: str = ""

    def clip(self, v: float) -> float:
        return min(self.hi, max(self.lo, v))

    def to_unit(self, v: float) -> float:
        """Map a value into [0, 1] for optimizers that want a normalized cube."""
        v = self.clip(v)
        if self.log:
            return (math.log10(v) - math.log10(self.lo)) / (
                math.log10(self.hi) - math.log10(self.lo)
            )
        return (v - self.lo) / (self.hi - self.lo)

    def from_unit(self, u: float) -> float:
        u = min(1.0, max(0.0, u))
        if self.log:
            lo, hi = math.log10(self.lo), math.log10(self.hi)
            return 10 ** (lo + u * (hi - lo))
        return self.lo + u * (self.hi - self.lo)


@dataclass(frozen=True)
class Testbench:
    """Load and bias conditions held fixed across every candidate."""

    cl: float = 1e-12
    # Input common mode as a fraction of VDD. 0.5 is mid-rail.
    vcm_frac: float = 0.5
    temp: float = 27.0
    fstart: float = 1.0
    fstop: float = 1e10
    pts_per_dec: int = 50


class Topology(ABC):
    name: str = ""
    description: str = ""
    params: tuple[Param, ...] = ()
    # Measurement names this topology guarantees to print.
    provides: tuple[str, ...] = ()

    def bounds(self) -> dict[str, tuple[float, float]]:
        return {p.name: (p.lo, p.hi) for p in self.params}

    def param(self, name: str) -> Param:
        for p in self.params:
            if p.name == name:
                return p
        raise KeyError(name)

    def defaults(self) -> dict[str, float]:
        """Geometric midpoint of every range -- the neutral starting point."""
        return {p.name: p.from_unit(0.5) for p in self.params}

    @abstractmethod
    def deck(self, pdk: PDK, values: dict[str, float], tb: Testbench) -> str: ...

    def postprocess(
        self, raw: dict[str, float], values: dict[str, float], pdk: PDK
    ) -> dict[str, float]:
        """Derive metrics SPICE should not be asked to compute.

        Anything requiring branch logic or unwrapping belongs here rather than
        in the `.control` block, where a wrong expression fails silently.
        """
        out = dict(raw)
        out["pm"] = phase_margin(
            raw.get("ph_dc", float("nan")), raw.get("ph_fu", float("nan"))
        )
        out["area"] = self.area(values)
        # Distance from the nearer supply rail, as a fraction of VDD. A railed
        # output makes gain and GBW meaningless, so specs constrain this to
        # reject degenerate operating points outright.
        vout = raw.get("vout_dc", float("nan"))
        if math.isfinite(vout) and pdk.vdd > 0:
            out["vout_margin"] = min(vout, pdk.vdd - vout) / pdk.vdd
        else:
            out["vout_margin"] = float("nan")
        return out

    def area(self, values: dict[str, float]) -> float:
        """Total gate area in m^2. Used as a cost proxy; override if a topology
        has passives whose area matters."""
        total = 0.0
        for p in self.params:
            if p.name.startswith("w"):
                idx = p.name[1:]
                lname = f"l{idx}"
                if any(q.name == lname for q in self.params):
                    total += values[p.name] * values[lname]
        return total


# ---------------------------------------------------------------------------


def _ac_control(vdd: float, tb: Testbench) -> str:
    """Shared control block: operating point, then open-loop AC.

    This measures raw quantities only. Phase margin is *not* computed here:
    some topologies are non-inverting at DC (the 5T OTA) and some are inverting
    (the two-stage Miller), so a single in-SPICE expression like `180 + phase`
    is wrong for half of them. The deck reports the DC phase and the phase at
    unity gain, and `phase_margin()` derives the margin with proper unwrapping.
    """
    return f"""
.control
set units=degrees
set noaskquit
option temp={tb.temp}

op
let isup = abs(i(vsup))
let pwr = isup * {vdd}
let vout_dc = v(vout)
print pwr
print vout_dc
print isup

ac dec {tb.pts_per_dec} {tb.fstart:g} {tb.fstop:g}
meas ac gain find vdb(vout) at={tb.fstart:g}
meas ac ph_dc find vp(vout) at={tb.fstart:g}
meas ac gbw when vdb(vout)=0 fall=1
meas ac ph_fu find vp(vout) when vdb(vout)=0 fall=1

quit
.endc
"""


def phase_margin(ph_dc: float, ph_fu: float) -> float:
    """Phase margin from the DC phase and the phase at unity gain.

    Measures the phase *lag accumulated* between DC and crossover, unwrapped
    into [0, 360), then subtracts it from 180. This is correct whether the
    amplifier starts at 0 deg (non-inverting) or +/-180 deg (inverting), and it
    does not care which branch ngspice reports the wrapped phase on.
    """
    if not (math.isfinite(ph_dc) and math.isfinite(ph_fu)):
        return float("nan")
    lag = ph_dc - ph_fu
    lag %= 360.0
    return 180.0 - lag


def _header(pdk: PDK, name: str) -> str:
    return f"* bias :: {name} :: pdk={pdk.name}\n{pdk.preamble()}\n"


def _mos(
    inst: str, d: str, g: str, s: str, b: str, model: str,
    w: float, l: float, pdk: PDK,
) -> str:
    """Emit one MOSFET with estimated source/drain diffusion geometry.

    Without AD/AS/PD/PS the junction capacitances are zero and every topology
    looks faster than it is -- ngspice warns about exactly this. The diffusion
    is estimated as a contacted source/drain of 2.5*Lmin, which is the usual
    rule of thumb and is what a layout would actually give you.
    """
    kind = "nmos" if model == pdk.nmos else "pmos"
    return pdk.mos_line(inst, d, g, s, b, kind, w, l)


def _drive(vdd: float, tb: Testbench) -> str:
    """Supply, common-mode bias, and a 1V differential AC stimulus.

    The differential drive is built from a single AC source split by two VCVS
    so that the common mode is held exactly at vcm while vinp/vinn move
    antiphase -- the standard open-loop OTA measurement.
    """
    vcm = vdd * tb.vcm_frac
    return f"""
Vsup vdd 0 DC {vdd:g}
Vcm  vcm 0 DC {vcm:g}
Vind vind 0 DC 0 AC 1
Einp vinp vcm vind 0 0.5
Einn vinn vcm vind 0 -0.5
CL   vout 0 {tb.cl:g}
"""


class OTA5T(Topology):
    """Five-transistor OTA: NMOS input pair, PMOS mirror load, mirrored tail.

    The simplest circuit that still has a real design trade-off -- gain against
    bandwidth against power, set by the input pair's operating point. Seven free
    parameters.
    """

    name = "ota5t"
    description = (
        "5-transistor OTA (NMOS input pair, PMOS current-mirror load, "
        "NMOS mirrored tail source)"
    )
    provides = (
        "gain", "gbw", "pm", "pwr", "vout_dc", "vout_margin", "isup", "area",
    )

    params = (
        Param("w1", 0.5e-6, 200e-6, "m", description="input pair width (M1/M2)"),
        Param("l1", 0.18e-6, 4e-6, "m", description="input pair length"),
        Param("w3", 0.5e-6, 200e-6, "m", description="PMOS load width (M3/M4)"),
        Param("l3", 0.18e-6, 4e-6, "m", description="PMOS load length"),
        Param("w5", 0.5e-6, 200e-6, "m", description="tail device width (M5/M6)"),
        Param("l5", 0.18e-6, 4e-6, "m", description="tail device length"),
        Param("ibias", 100e-9, 500e-6, "A", description="bias current into the tail mirror"),
    )

    def deck(self, pdk: PDK, values: dict[str, float], tb: Testbench) -> str:
        v = {p.name: p.clip(values[p.name]) for p in self.params}
        return (
            _header(pdk, self.name)
            + _drive(pdk.vdd, tb)
            + "\n* --- input pair ---\n"
            + _mos("M1", "vd1", "vinp", "vtail", "0", pdk.nmos, v["w1"], v["l1"], pdk) + "\n"
            + _mos("M2", "vout", "vinn", "vtail", "0", pdk.nmos, v["w1"], v["l1"], pdk) + "\n"
            + "\n* --- PMOS current-mirror load ---\n"
            + _mos("M3", "vd1", "vd1", "vdd", "vdd", pdk.pmos, v["w3"], v["l3"], pdk) + "\n"
            + _mos("M4", "vout", "vd1", "vdd", "vdd", pdk.pmos, v["w3"], v["l3"], pdk) + "\n"
            + "\n* --- tail source and its bias diode ---\n"
            + _mos("M5", "vtail", "vbias", "0", "0", pdk.nmos, v["w5"], v["l5"], pdk) + "\n"
            + _mos("M6", "vbias", "vbias", "0", "0", pdk.nmos, v["w5"], v["l5"], pdk) + "\n"
            + f"Ibias vdd vbias DC {v['ibias']:g}\n"
            + _ac_control(pdk.vdd, tb)
        )


class MillerOTA(Topology):
    """Two-stage Miller-compensated op-amp.

    Stage one is a PMOS-input differential pair with NMOS mirror load; stage two
    is a common-source NMOS driver with a PMOS current-source load, split by a
    Miller cap and a nulling resistor. Ten free parameters, and a genuine
    stability problem -- the compensation network has to track the sizing.
    """

    name = "miller"
    description = (
        "Two-stage Miller-compensated op-amp (PMOS input pair, NMOS mirror "
        "load, common-source second stage, Rz-Cc compensation)"
    )
    provides = (
        "gain", "gbw", "pm", "pwr", "vout_dc", "vout_margin", "isup", "area",
    )

    params = (
        Param("w1", 0.5e-6, 300e-6, "m", description="input pair width (M1/M2)"),
        Param("l1", 0.18e-6, 4e-6, "m", description="input pair length"),
        Param("w3", 0.5e-6, 300e-6, "m", description="NMOS mirror load width (M3/M4)"),
        Param("l3", 0.18e-6, 4e-6, "m", description="NMOS mirror load length"),
        Param("w5", 0.5e-6, 300e-6, "m", description="tail PMOS width (M5/M8)"),
        Param("l5", 0.18e-6, 4e-6, "m", description="tail PMOS length"),
        Param("w6", 0.5e-6, 600e-6, "m", description="second-stage NMOS driver width (M6)"),
        Param("l6", 0.18e-6, 4e-6, "m", description="second-stage NMOS driver length"),
        Param("w7", 0.5e-6, 600e-6, "m", description="second-stage PMOS load width (M7)"),
        Param("l7", 0.18e-6, 4e-6, "m", description="second-stage PMOS load length"),
        Param("ibias", 100e-9, 500e-6, "A", description="bias current into the tail mirror"),
        Param("cc", 10e-15, 20e-12, "F", description="Miller compensation capacitor"),
        Param("rz", 10.0, 200e3, "ohm", description="nulling resistor in series with Cc"),
    )

    def deck(self, pdk: PDK, values: dict[str, float], tb: Testbench) -> str:
        v = {p.name: p.clip(values[p.name]) for p in self.params}
        return (
            _header(pdk, self.name)
            + _drive(pdk.vdd, tb)
            + "\n* --- stage 1: PMOS input pair, NMOS mirror load ---\n"
            + _mos("M1", "vd1", "vinp", "vtail", "vdd", pdk.pmos, v["w1"], v["l1"], pdk) + "\n"
            + _mos("M2", "vd2", "vinn", "vtail", "vdd", pdk.pmos, v["w1"], v["l1"], pdk) + "\n"
            + _mos("M3", "vd1", "vd1", "0", "0", pdk.nmos, v["w3"], v["l3"], pdk) + "\n"
            + _mos("M4", "vd2", "vd1", "0", "0", pdk.nmos, v["w3"], v["l3"], pdk) + "\n"
            + "\n* --- tail PMOS mirrored from the bias diode ---\n"
            + _mos("M5", "vtail", "vbias", "vdd", "vdd", pdk.pmos, v["w5"], v["l5"], pdk) + "\n"
            + _mos("M8", "vbias", "vbias", "vdd", "vdd", pdk.pmos, v["w5"], v["l5"], pdk) + "\n"
            + f"Ibias vbias 0 DC {v['ibias']:g}\n"
            + "\n* --- stage 2: common-source driver with current-source load ---\n"
            + _mos("M6", "vout", "vd2", "0", "0", pdk.nmos, v["w6"], v["l6"], pdk) + "\n"
            + _mos("M7", "vout", "vbias", "vdd", "vdd", pdk.pmos, v["w7"], v["l7"], pdk) + "\n"
            + "\n* --- Miller compensation with nulling resistor ---\n"
            + f"Rz  vd2 vz  {v['rz']:g}\n"
            + f"Cc  vz  vout {v['cc']:g}\n"
            + _ac_control(pdk.vdd, tb)
        )

    def area(self, values: dict[str, float]) -> float:
        gate = super().area(values)
        # Charge the compensation cap at roughly 2 fF/um^2 for a MIM cap, so an
        # optimizer cannot buy stability with an arbitrarily large capacitor.
        cap_area = values.get("cc", 0.0) / 2e-15 * 1e-12
        return gate + cap_area


REGISTRY: dict[str, Topology] = {}


def register(t: Topology) -> Topology:
    REGISTRY[t.name] = t
    return t


register(OTA5T())
register(MillerOTA())


def get(name: str) -> Topology:
    if name not in REGISTRY:
        raise KeyError(f"unknown topology {name!r}; have {sorted(REGISTRY)}")
    return REGISTRY[name]
