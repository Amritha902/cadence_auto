"""PDK abstraction.

A PDK here is the minimum an optimizer needs to know: where the model cards
live, what the devices are called, what the supply is, and what ranges the
sizing parameters may take. Swapping SKY130 for IHP SG13G2 -- or for a
proprietary PDK behind a company firewall -- should be a config change and
nothing else.

Deliberately, no PDK content is vendored into this repo. `dev180` is a small
hand-written Level-3 card used only to bring the harness up; it is NOT a
calibrated process and must never be used for reported results. Real PDKs are
fetched by scripts/fetch_pdk.sh.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PDK_DIR = REPO_ROOT / "pdks"
_SKY_CELLS = PDK_DIR / "sky130" / "cells"


@dataclass(frozen=True)
class PDK:
    name: str
    # SPICE .lib / .include line inserted at the top of every deck.
    include: str
    nmos: str
    pmos: str
    vdd: float
    # Minimum drawn length; used to build default parameter bounds.
    lmin: float
    wmin: float
    # How devices are instantiated. Hand-written .model cards take an "M"
    # instance naming the model; foundry PDKs almost always wrap each device in
    # a .subckt, which needs an "X" instance and lowercase parameters. Getting
    # this wrong produces a netlist ngspice parses and silently mis-simulates,
    # so it belongs to the PDK rather than to the topology.
    device_style: str = "model"      # "model" | "subckt"
    # True only for processes with real, foundry-calibrated models. The bench
    # refuses to publish results from a PDK where this is False.
    calibrated: bool = True
    notes: str = ""
    extra: dict[str, str] = field(default_factory=dict)

    def preamble(self) -> str:
        return self.include

    def model_for(self, kind: str) -> str:
        return self.nmos if kind.lower() == "nmos" else self.pmos

    def mos_line(
        self, inst: str, d: str, g: str, s: str, b: str, kind: str,
        w: float, l: float, *, diffusion: bool = True,
    ) -> str:
        """One MOSFET instance, in whichever form this process expects.

        Source/drain diffusion is estimated as a contacted 2.5*Lmin region.
        Without it the junction capacitances are zero and every circuit looks
        faster than it is.
        """
        bare = inst[1:] if inst[:1].upper() in ("M", "X") else inst
        model = self.model_for(kind)
        hdif = 2.5 * self.lmin
        ad = w * hdif
        pd = 2 * (w + hdif)

        if self.device_style == "subckt":
            line = f"X{bare} {d} {g} {s} {b} {model} w={w:g} l={l:g}"
            if diffusion:
                line += (f" ad={ad:g} as={ad:g} pd={pd:g} ps={pd:g}")
            return line

        line = f"M{bare} {d} {g} {s} {b} {model} W={w:g} L={l:g}"
        if diffusion:
            line += f" AD={ad:g} AS={ad:g} PD={pd:g} PS={pd:g}"
        return line

    def default_bounds(self) -> dict[str, tuple[float, float]]:
        """Sizing ranges a designer would actually consider.

        Widths run to 100x minimum, lengths to 20x -- wide enough to contain
        good solutions for the topologies in this benchmark without letting an
        optimizer wander into 1mm devices.
        """
        return {
            "W": (self.wmin, self.wmin * 1000),
            "L": (self.lmin, self.lmin * 20),
        }


# ---------------------------------------------------------------------------
# Bring-up PDK. Not physical. Not for results.
# ---------------------------------------------------------------------------

DEV180 = PDK(
    name="dev180",
    include=f".include {PDK_DIR / 'dev180' / 'dev180.lib'}",
    nmos="nch",
    pmos="pch",
    vdd=1.8,
    lmin=0.18e-6,
    wmin=0.22e-6,
    calibrated=False,
    notes=(
        "Hand-written BSIM3v3 card for harness bring-up only. Parameters are "
        "plausible for a generic 180nm bulk CMOS process but are NOT calibrated "
        "against any foundry. Never report benchmark numbers from this PDK."
    ),
)

SKY130 = PDK(
    name="sky130",
    # Only the two devices these topologies use, rather than the whole `tt`
    # section. That section also pulls in the 5V and ESD models, several of
    # which are written with a bare `include` that ngspice parses as a current
    # source and dies on -- the reason the usual advice is to build sky130
    # through open_pdks first. Including just the primitives in use sidesteps
    # that entirely, parses faster, and is honest: a circuit should declare the
    # devices it actually instantiates.
    # Mismatch files first: the tt models reference *_slope_spectre
    # parameters defined there, and a corner file includes its own .pm3 at the
    # end, so the definitions have to already be in scope.
    include="\n".join((
        f'.include {PDK_DIR / "sky130_nominal.spice"}',
        f'.include {_SKY_CELLS / "nfet_01v8" / "sky130_fd_pr__nfet_01v8__mismatch.corner.spice"}',
        f'.include {_SKY_CELLS / "pfet_01v8" / "sky130_fd_pr__pfet_01v8__mismatch.corner.spice"}',
        f'.include {_SKY_CELLS / "nfet_01v8" / "sky130_fd_pr__nfet_01v8__tt.corner.spice"}',
        f'.include {_SKY_CELLS / "pfet_01v8" / "sky130_fd_pr__pfet_01v8__tt.corner.spice"}',
    )),
    nmos="sky130_fd_pr__nfet_01v8",
    pmos="sky130_fd_pr__pfet_01v8",
    vdd=1.8,
    lmin=0.15e-6,
    wmin=0.42e-6,
    device_style="subckt",
    calibrated=True,
    notes=(
        "SkyWater 130nm open PDK, BSIM4. Primitives are .subckt wrappers, so "
        "they instantiate as X devices. Fetch with scripts/fetch_pdk.sh sky130."
    ),
)

IHP_SG13G2 = PDK(
    name="ihp-sg13g2",
    include=f'.lib {PDK_DIR / "ihp-sg13g2" / "models" / "cornerMOSlv.lib"} mos_tt',
    nmos="sg13_lv_nmos",
    pmos="sg13_lv_pmos",
    vdd=1.5,
    lmin=0.13e-6,
    wmin=0.15e-6,
    device_style="subckt",
    calibrated=True,
    notes=(
        "IHP SG13G2 130nm BiCMOS. Devices are PSP 103.6, which ngspice can "
        "only load as a compiled OSDI shared object -- and the upstream repo "
        "ships no osdi/ directory, so the binaries must be built with OpenVAF "
        "for your platform first. Until then this PDK is registered but not "
        "usable; sky130 is BSIM4 and needs no compiled models."
    ),
)

REGISTRY: dict[str, PDK] = {p.name: p for p in (DEV180, SKY130, IHP_SG13G2)}


def get(name: str) -> PDK:
    if name not in REGISTRY:
        raise KeyError(f"unknown PDK {name!r}; have {sorted(REGISTRY)}")
    pdk = REGISTRY[name]
    if not _model_files_present(pdk):
        raise FileNotFoundError(
            f"PDK {name!r} is registered but its model files are missing.\n"
            f"Expected under {PDK_DIR / name}\n"
            f"Run: scripts/fetch_pdk.sh {name}"
        )
    return pdk


def available() -> list[str]:
    return [n for n, p in REGISTRY.items() if _model_files_present(p)]


def _model_files_present(pdk: PDK) -> bool:
    # The include line ends with a path (possibly followed by a corner name).
    for token in pdk.include.split():
        if "/" in token and Path(token).exists():
            return True
    return False
