"""HTTP front end for the circuit engine.

Thin on purpose. Every route hands off to bias.design and returns what the
simulator actually measured -- no caching of plausible-looking answers, no
precomputed demos. A request that fails to meet its spec says so, because a
tool that quietly returns a near miss is worse than one that admits it.

Run it with:  web/run.sh
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from bias import design, logic, pdk, specparse, symbol

STATIC = Path(__file__).parent / "static"

app = FastAPI(title="cadence_auto", docs_url="/api/docs")


class BuildRequest(BaseModel):
    request: str = Field(min_length=1, max_length=400)
    pdk: str = "dev180"
    # Kept modest so a browser request stays responsive. The CLI has no cap.
    budget: int = Field(default=300, ge=20, le=1200)
    attempts: int = Field(default=2, ge=1, le=4)


def _resolve_pdk(name: str):
    if name == "dev180":
        return pdk.DEV180
    try:
        return pdk.get(name)
    except (KeyError, FileNotFoundError) as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get("/api/catalogue")
def catalogue() -> dict:
    """What this instance can build, and on which processes."""
    cells = []
    for name in sorted(logic.CELLS):
        builder, ins, outs, _ = logic.get(name)
        circuit = builder()
        cells.append({
            "name": name,
            "transistors": len(circuit.transistors()),
            "inputs": ins,
            "outputs": outs,
        })

    processes = []
    ready = set(pdk.available())
    for name, p in pdk.REGISTRY.items():
        processes.append({
            "name": name,
            "ready": name in ready,
            "calibrated": p.calibrated,
            "vdd": p.vdd,
            "blocker": p.blocker if not p.usable else "",
        })

    return {"cells": cells, "processes": processes, "examples": EXAMPLES}


EXAMPLES = [
    {"label": "Half adder", "request": "build me a half adder"},
    {"label": "Full adder", "request": "build me a full adder"},
    {"label": "2:1 mux", "request": "a 2:1 multiplexer"},
    {"label": "2-to-4 decoder", "request": "a 2 to 4 decoder"},
    {"label": "Comparator", "request": "a magnitude comparator"},
    {"label": "2-bit adder", "request": "a 2 bit adder"},
    {"label": "Op-amp", "request":
     "an op-amp with 40dB gain and 5MHz bandwidth under 200uW driving a 1pF load"},
    {"label": "Two-stage op-amp", "request":
     "op-amp with 65dB gain, 8MHz bandwidth, under 200uW, 1pF load"},
]


@app.post("/api/build")
async def build(req: BuildRequest) -> dict:
    """Build whatever was asked for, and report what the simulator said."""
    process = _resolve_pdk(req.pdk)
    started = time.perf_counter()

    # Simulation is blocking and can take seconds; keep the event loop free.
    try:
        result = await asyncio.to_thread(
            design.build_any,
            req.request,
            process,
            use_llm=False,
            budget=req.budget,
            attempts=req.attempts,
        )
    except LookupError as exc:
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:                      # surface, do not swallow
        raise HTTPException(500, f"{type(exc).__name__}: {exc}") from exc

    elapsed = time.perf_counter() - started

    if hasattr(result, "report"):
        return _digital_payload(result, process, elapsed)
    return _analog_payload(result, process, elapsed)


def _digital_payload(result, process, elapsed: float) -> dict:
    builder, ins, outs, _ = logic.get(result.cell)
    circuit = result.circuit
    sym = symbol.Symbol.from_circuit(
        circuit, ins, outs,
        subtitle=f"{len(circuit.transistors())} transistors, static CMOS",
    )

    rows = [
        {
            "inputs": r.inputs,
            "expected": r.expected,
            "actual": r.actual,
            "volts": {k: round(v, 4) for k, v in r.measured_v.items()},
            "passed": r.passed,
        }
        for r in result.report.rows
    ]

    return {
        "kind": "digital",
        "request": result.request,
        "resolved": result.cell,
        "how": result.how,
        "ok": result.ok,
        "pdk": process.name,
        "calibrated": process.calibrated,
        "seconds": round(elapsed, 2),
        "transistors": len(circuit.transistors()),
        "summary": circuit.summary(),
        "inputs": ins,
        "outputs": outs,
        "rows": rows,
        "worst_rail_error": round(result.report.worst_rail_error, 5),
        "symbol_svg": sym.render(),
        "netlist": circuit.to_spice(pdk=process),
        "reason": result.report.reason,
    }


def _analog_payload(result, process, elapsed: float) -> dict:
    from bias import plot, topology as topo_registry
    from bias.topology import Testbench

    targets = [
        {
            "name": m.name,
            "relation": ">=" if m.direction == "max" else "<=",
            "target": m.target,
            "unit": m.unit,
            "implied": m.name in result.parsed.implied,
            "measured": result.measured.get(m.name),
            "passed": m.satisfied(result.measured.get(m.name, float("nan"))),
        }
        for m in result.parsed.spec.metrics
    ]

    payload = {
        "kind": "analog",
        "request": result.request,
        "resolved": result.topology_name,
        "how": result.why_topology,
        "ok": result.solved,
        "pdk": process.name,
        "calibrated": process.calibrated,
        "seconds": round(elapsed, 2),
        "sims_used": result.sims_used,
        "sims_to_target": result.sims_to_target,
        "infeasible": result.infeasible,
        "load": result.parsed.cl,
        "targets": targets,
        "sizing": {k: v for k, v in result.values.items()},
        "symbol_svg": "",
        "bode_svg": "",
        "netlist": "",
    }

    if result.values and not result.infeasible:
        topo = topo_registry.get(result.topology_name)
        tb = Testbench(cl=result.parsed.cl or 1e-12)
        payload["netlist"] = topo.deck(process, result.values, tb)
        payload["symbol_svg"] = symbol.Symbol(
            name=result.topology_name,
            inputs=["vinp", "vinn"], outputs=["vout"],
            power=["vdd"], ground=["0"],
            subtitle=f"{len(topo.params)} sized parameters",
        ).render()
        payload["bode_svg"] = plot.bode(
            topo, process, result.values, tb, result.measured
        ) or ""

    return payload


@app.post("/api/parse")
async def parse(req: BuildRequest) -> dict:
    """Show how a request is understood, without simulating anything.

    Useful on its own: it lets someone see which constraints were supplied on
    their behalf before spending a minute of simulation on them.
    """
    try:
        cell, how = design.resolve(req.request, use_llm=False)
        return {"kind": "digital", "resolved": cell, "how": how}
    except LookupError:
        pass

    try:
        parsed = specparse.parse(req.request)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc

    gain = parsed.found.get("gain")
    topo, why = design.choose_topology(gain)
    return {
        "kind": "analog",
        "resolved": topo,
        "how": why,
        "load": parsed.cl,
        "targets": [
            {
                "name": m.name,
                "relation": ">=" if m.direction == "max" else "<=",
                "target": m.target,
                "unit": m.unit,
                "implied": m.name in parsed.implied,
            }
            for m in parsed.spec.metrics
        ],
    }


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")
