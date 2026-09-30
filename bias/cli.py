"""Command line interface.

    bias list                          what topologies, specs, optimizers exist
    bias sim ota5t                     simulate one point and print measurements
    bias solve ota5t-base --with de    run one optimizer against one spec
    bias bench --seeds 5               the full matrix
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from . import bench, optimizers, pdk, specs, topology
from .evaluate import Evaluator
from .sim import NgspiceNotFound
from .topology import Testbench


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="bias",
        description="Benchmark analog circuit sizing: LLM agent vs classical optimizers.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="show available topologies, specs and optimizers")

    p_sim = sub.add_parser("sim", help="simulate one sizing and print what it measures")
    p_sim.add_argument("topology", help="topology name")
    p_sim.add_argument("--pdk", default="dev180")
    p_sim.add_argument("--cl", type=float, default=1e-12, help="load capacitance (F)")
    p_sim.add_argument(
        "--set", action="append", default=[], metavar="NAME=VALUE",
        help="override one parameter; repeatable",
    )
    p_sim.add_argument("--deck", action="store_true", help="print the netlist and exit")

    p_solve = sub.add_parser("solve", help="run one optimizer against one spec")
    p_solve.add_argument("spec", help="spec name")
    p_solve.add_argument("--with", dest="optimizer", default="de")
    p_solve.add_argument("--pdk", default="dev180")
    p_solve.add_argument("--budget", type=int, default=200)
    p_solve.add_argument("--seed", type=int, default=0)
    p_solve.add_argument("--cl", type=float, default=1e-12)
    p_solve.add_argument(
        "--model", default=None, help="model id, for --with llm"
    )
    p_solve.add_argument(
        "--trace", action="store_true", help="print each candidate as it is tried"
    )

    p_build = sub.add_parser(
        "build", help="describe a circuit in words; get a verified netlist and symbol"
    )
    p_build.add_argument("request", help='e.g. "build me a half adder symbol"')
    p_build.add_argument("--pdk", default="dev180")
    p_build.add_argument("--out", default=None,
                         help="directory for netlist, symbol, testbench, truth table")
    p_build.add_argument("--wn", type=float, default=1.0e-6,
                         help="NMOS width for synthesised logic (m)")
    p_build.add_argument("--beta", type=float, default=2.5,
                         help="PMOS/NMOS width ratio")
    p_build.add_argument("--no-llm", action="store_true",
                         help="deterministic matching only; never call a model")
    p_build.add_argument("--budget", type=int, default=400,
                         help="simulations per attempt, for analog requests")
    p_build.add_argument("--attempts", type=int, default=3,
                         help="restarts before giving up, for analog requests")
    p_build.add_argument("--with", dest="optimizer", default="de",
                         help="optimizer for analog sizing (de, nelder-mead, llm, ...)")

    p_cells = sub.add_parser("cells", help="list the logic cells that can be built")
    p_cells.add_argument("--truth", metavar="CELL", default=None,
                         help="print one cell's reference truth table")

    p_bench = sub.add_parser("bench", help="run the full matrix and write a report")
    p_bench.add_argument("--specs", nargs="*", default=None)
    p_bench.add_argument(
        "--optimizers", nargs="*", default=["random", "lhs", "nelder-mead", "de"]
    )
    p_bench.add_argument("--pdk", default="dev180")
    p_bench.add_argument("--budget", type=int, default=200)
    p_bench.add_argument("--seeds", type=int, default=5)
    p_bench.add_argument("--cl", type=float, default=1e-12)
    p_bench.add_argument("--out", default="results/bench.json")

    args = parser.parse_args(argv)

    try:
        return _dispatch(args)
    except NgspiceNotFound as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except (KeyError, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _dispatch(args) -> int:
    if args.command == "list":
        return _cmd_list()
    if args.command == "sim":
        return _cmd_sim(args)
    if args.command == "solve":
        return _cmd_solve(args)
    if args.command == "bench":
        return _cmd_bench(args)
    if args.command == "build":
        return _cmd_build(args)
    if args.command == "cells":
        return _cmd_cells(args)
    return 1


def _cmd_list() -> int:
    print("TOPOLOGIES")
    for name, t in topology.REGISTRY.items():
        print(f"  {name:<10} {t.description}")
        print(f"{'':<12} {len(t.params)} parameters: "
              f"{', '.join(p.name for p in t.params)}")

    print("\nSPECS")
    for name in specs.names():
        spec, topo = specs.get(name)
        print(f"  {name:<14} [{topo}] {spec.description}")

    print("\nOPTIMIZERS")
    for name in optimizers.available():
        note = "  (needs ANTHROPIC_API_KEY)" if name == "llm" else ""
        print(f"  {name}{note}")

    print("\nPDKS")
    have = set(pdk.available())
    for name, p in pdk.REGISTRY.items():
        if name in have:
            status = "ready"
        elif not p.usable:
            status = "unusable"
        else:
            status = "not fetched"
        cal = "" if p.calibrated else "  [NOT CALIBRATED -- dev only]"
        print(f"  {name:<14} {status}{cal}")
        if not p.usable and p.blocker:
            print(f"{'':<16} {p.blocker}")
    return 0


def _resolve_pdk(name: str):
    return pdk.DEV180 if name == "dev180" else pdk.get(name)


def _cmd_sim(args) -> int:
    topo = topology.get(args.topology)
    process = _resolve_pdk(args.pdk)
    tb = Testbench(cl=args.cl)

    values = topo.defaults()
    for override in args.set:
        if "=" not in override:
            raise ValueError(f"--set expects NAME=VALUE, got {override!r}")
        key, raw = override.split("=", 1)
        key = key.strip()
        if key not in values:
            raise KeyError(
                f"{key!r} is not a parameter of {topo.name}; "
                f"have {', '.join(values)}"
            )
        values[key] = float(raw)

    if args.deck:
        print(topo.deck(process, values, tb))
        return 0

    from .sim import run_deck

    result = run_deck(topo.deck(process, values, tb))
    measured = topo.postprocess(result.values, values, process)

    print(f"{topo.name} on {process.name}, CL={args.cl:.3g}F")
    print("  sizing: " + " ".join(f"{k}={v:.4g}" for k, v in values.items()))
    if not result.ok:
        print(f"  SIMULATION FAILED: {result.reason}")
    print("  measured:")
    for key in topo.provides:
        print(f"    {key:<12} {measured.get(key, float('nan')):.6g}")
    return 0 if result.ok else 1


def _cmd_solve(args) -> int:
    spec, topo_name = specs.get(args.spec)
    topo = topology.get(topo_name)
    process = _resolve_pdk(args.pdk)

    kwargs = {}
    if args.optimizer in ("llm", "agent") and args.model:
        kwargs["model"] = args.model

    ev = Evaluator(topo, process, spec, Testbench(cl=args.cl), budget=args.budget)
    optimizer = optimizers.get(args.optimizer, **kwargs)

    print(f"{args.spec} [{topo.name}] with '{args.optimizer}' "
          f"on {process.name}, budget={args.budget}, seed={args.seed}")
    if not process.calibrated:
        print("  note: dev PDK -- results are not publishable\n")

    optimizer.run(ev, np.random.default_rng(args.seed))

    if args.trace:
        turns = {t.index: t.reasoning for t in getattr(optimizer, "turns", [])}
        for e in ev.history:
            if e.cached:
                continue
            print(f"\n#{e.index} score={e.score:.4f}"
                  f"{'  SOLVED' if e.satisfied else ''}")
            if e.index in turns and turns[e.index]:
                print(f"  agent: {turns[e.index]}")
            print("  " + " ".join(f"{k}={v:.4g}" for k, v in e.values.items()))

    best = ev.best
    print(f"\nsimulations used: {ev.used}")
    if best is None:
        print("no candidate was evaluated")
        return 1

    if ev.solved():
        print(f"SOLVED at simulation {ev.sims_to_target()}")
    else:
        print(f"not solved; best score {best.score:.4f}")
    print("\nbest sizing:")
    for k, v in best.values.items():
        print(f"  {k:<8} {v:.6g}")
    print("\nmeasured:")
    print(spec.report(best.measured))
    return 0 if ev.solved() else 1


def _cmd_build(args) -> int:
    from . import design, logic

    process = _resolve_pdk(args.pdk)
    sizing = logic.Sizing(wn=args.wn, beta=args.beta)

    try:
        result = design.build_any(
            args.request,
            process,
            outdir=args.out,
            use_llm=not args.no_llm,
            sizing=sizing,
            budget=args.budget,
            attempts=args.attempts,
            optimizer=args.optimizer,
        )
    except LookupError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(result.describe())
    return 0 if result.ok else 1


def _cmd_cells(args) -> int:
    from . import logic

    if args.truth:
        rows = logic.truth_table(args.truth)
        ins = list(rows[0][0])
        outs = list(rows[0][1])
        print("  " + " ".join(f"{i:>3}" for i in ins) + "  |  "
              + "  ".join(f"{o:>6}" for o in outs))
        for inputs, expected in rows:
            print("  " + " ".join(f"{inputs[i]:>3}" for i in ins) + "  |  "
                  + "  ".join(f"{expected[o]:>6}" for o in outs))
        return 0

    print("LOGIC CELLS")
    for name in sorted(logic.CELLS):
        builder, ins, outs, _ = logic.get(name)
        circuit = builder()
        print(f"  {name:<12} {len(circuit.transistors()):>3}T   "
              f"in: {','.join(ins):<12} out: {','.join(outs)}")
    print("\nBuild one with:  bias build \"half adder\" --out ./out")
    return 0


def _cmd_bench(args) -> int:
    spec_names = args.specs or specs.names()
    print(f"benchmarking {len(spec_names)} spec(s) x {len(args.optimizers)} "
          f"optimizer(s) x {args.seeds} seed(s)\n")

    report = bench.run_matrix(
        spec_names,
        args.optimizers,
        seeds=args.seeds,
        pdk_name=args.pdk,
        budget=args.budget,
        cl=args.cl,
    )

    print("\n" + bench.format_table(report))
    out = bench.save(report, Path(args.out))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
