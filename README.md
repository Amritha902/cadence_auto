# bias

**Describe a circuit, get a verified one. And an honest benchmark for whether
language models can size analog circuits at all.**

Two halves of the same problem, sharing one structured circuit representation.

**Building.** Say "build me a half adder symbol" and get a transistor-level
netlist, a schematic symbol, a testbench, and a truth table *proved by
transient simulation* — not asserted. Static CMOS logic is synthesised
deterministically, so the transistors are correct by construction.

**Benchmarking.** A reproducible suite for LLM-driven analog sizing. Several
papers show LLM sizing working — [EEsizer](https://arxiv.org/abs/2509.25510),
[LEDRO](https://arxiv.org/html/2411.12930),
[AutoSizer](https://arxiv.org/abs/2602.02849) — but each uses its own
topologies, PDK and success criterion, so no two results can be compared and
most cannot be rerun. This is the shared runnable baseline, with classical
optimizers implemented to win.

Everything runs on open PDKs with ngspice. No commercial licence, no NDA, no
cloud.

---

## Quickstart

```bash
git clone https://github.com/Amritha902/cadence_auto.git && cd cadence_auto
python3 -m venv .venv && .venv/bin/pip install -e '.[llm,dev]'
brew install ngspice      # or: apt install ngspice
```

Build a circuit from a description:

```bash
.venv/bin/bias build "build me a half adder" --out ./out
```

Simulate one analog sizing:

```bash
.venv/bin/bias sim ota5t
```

Run a classical optimizer against a spec:

```bash
.venv/bin/bias solve ota5t-base --with de --budget 400
```

Run the language model, with its reasoning shown per step:

```bash
export ANTHROPIC_API_KEY=...
.venv/bin/bias solve ota5t-base --with llm --budget 40 --trace
```

The full matrix:

```bash
.venv/bin/bias bench --seeds 5 --budget 200
```

---

## Building circuits from a description

```bash
bias cells                                    # what can be built
bias build "build me a half adder" --out ./out
```

```
request : "build me a half adder"
resolved: half_adder  (matched cell name)
built   : half_adder: 9 nmos, 9 pmos; 13 nets
          18 transistors, structurally valid
verified: VERIFIED against the truth table by transient simulation

    a   b  |    sum   carry   |   sum(V)  carry(V)
  ---------------------------------------------------
    0   0  |      0       0   |    0.000     0.000   ok
    0   1  |      1       0   |    1.800     0.000   ok
    1   0  |      1       0   |    1.800     0.000   ok
    1   1  |      0       1   |    0.000     1.800   ok

worst output rail error: 0.00% of VDD
```

Four artifacts land in `./out`: the netlist, an SVG schematic symbol, the
transient testbench, and the truth table.

### Why it does not hallucinate

The usual way to build this is to ask a language model for SPICE. It returns
text that looks right and does not simulate — floating nodes, a MOSFET with
three terminals, a net referenced once. Nothing notices until ngspice fails,
and the error is about line 14 rather than about the circuit.

So the model never writes a netlist here. The split is:

| step | who does it |
| --- | --- |
| decide *what* to build | model (or a lookup, which is tried first) |
| build the transistors | deterministic synthesis in `logic.py` |
| check it is a circuit | `netlist.py` validator |
| prove it works | ngspice transient vs. the truth table |

A wrong answer is therefore a wrong *choice*, not a broken circuit — and the
last step catches even that.

Resolution is deterministic first and reaches for a model only on requests the
alias table does not cover. For the circuits people actually ask for, a lookup
is exact, instant, free, and cannot hallucinate.

### The validator

`Circuit.validate()` catches what breaks generated netlists, and reports it in
electrical terms rather than SPICE terms:

- a MOSFET missing its bulk terminal, or carrying an invented one
- a transistor with no W or L
- a net connected to exactly one terminal — floating, which ngspice reports as
  a singular matrix
- a device with both ends on the same net
- an island of nets with no conductive path to ground
- duplicate instance names, which SPICE silently overwrites

### Cells

| cell | transistors | notes |
| --- | --- | --- |
| `inverter` | 2 | |
| `nand2` | 4 | series NMOS widened by stack depth |
| `nor2` | 4 | series PMOS widened by stack depth |
| `xor2` | 16 | four NAND2, every node actively driven |
| `half_adder` | 18 | carry reuses the XOR's first NAND |
| `full_adder` | 50 | two half adders and an OR |

The half adder is 18 transistors rather than 20 because `NAND(a,b)` is already
computed inside the XOR, so the carry costs one inverter instead of a whole AND
gate.

### Circuit identity

`Circuit.graph_hash()` fingerprints topology via Weisfeiler-Lehman refinement
over the device/net graph. Two circuits hash equal when they are the same
topology under any renaming of internal nets and devices — but supply and
ground nets are labelled distinctly, because swapping VDD for an internal node
is a different circuit even when the graph shape is identical.


---

## What is measured

Three numbers, per (spec, optimizer):

| metric | meaning |
| --- | --- |
| **success rate** | fraction of seeds that met every hard constraint |
| **sims to target** | median simulations spent before the first solution |
| **best score** | how close the failures got (0 = all specs met) |

`sims to target` is the headline. Simulation is the expensive resource in real
analog design — an industrial corner run is minutes, not milliseconds — so the
question is not whether an optimizer *can* find a point, but how few
simulations it needs. A method that wins on wall-clock by burning 10,000
simulations has not solved the problem practitioners have.

Every optimizer reaches ngspice through the same `Evaluator`, which counts and
caches on the exact parameter vector. One definition of "one simulation", for
everybody.

---

## The suite

Two topologies, three difficulty tiers each.

**`ota5t`** — five-transistor OTA (NMOS input pair, PMOS mirror load, mirrored
tail). Seven free parameters. The simplest circuit with a real trade-off: gain
against bandwidth against power, set by the input pair's operating point.

**`miller`** — two-stage Miller-compensated op-amp with a nulling resistor.
Thirteen free parameters. Harder in a specific way: the second stage's load
current must match what its driver sinks, or the output sits at a rail and the
gain collapses. An optimizer has to fix the bias before it can chase
performance.

| spec | gain | GBW | PM | power | notes |
| --- | --- | --- | --- | --- | --- |
| `ota5t-easy` | ≥45 dB | ≥10 MHz | ≥60° | ≤100 µW | smoke test |
| `ota5t-base` | ≥50 dB | ≥20 MHz | ≥60° | ≤50 µW | reference single-stage task |
| `ota5t-hard` | ≥48 dB | ≥25 MHz | ≥60° | ≤70 µW | + area penalty |
| `miller-easy` | ≥55 dB | ≥1 MHz | ≥55° | ≤500 µW | mostly a biasing problem |
| `miller-base` | ≥70 dB | ≥10 MHz | ≥60° | ≤200 µW | reference two-stage task |
| `miller-hard` | ≥75 dB | ≥25 MHz | ≥60° | ≤150 µW | + area penalty |

Every spec also requires `vout_margin ≥ 0.20` — the DC output must sit at least
20% of VDD from either rail. Without it an optimizer can score well on a
circuit that is not an amplifier.

**Targets are calibrated, not guessed.** `scripts/calibrate.py` runs the
strongest classical baseline at a generous budget against every spec and
reports what it solved. Current state, DE at 800 sims × 3 seeds:

| spec | solved | median sims to target |
| --- | --- | --- |
| `ota5t-easy` | 3/3 | 6 |
| `ota5t-base` | 3/3 | 315 |
| `ota5t-hard` | 3/3 | 238 |
| `miller-easy` | 3/3 | 85 |
| `miller-base` | 3/3 | 384 |
| `miller-hard` | 1/3 | 283 |

`ota5t-hard` took two rounds of this to get right, and the failures were
instructive. v1 asked for 50 MHz at 40 µW: into a 1 pF load that needs
gm = 314 µS from ~11 µA per branch, i.e. gm/Id ≈ 28, above the subthreshold
limit — simply not a circuit. v2 asked for 52 dB and still solved 0/3, but all
three seeds converged on gain = 49.1 dB with phase margin pinned at exactly
60°. That is the topology's real ceiling: a 5T OTA buys gain with channel
length, length costs phase margin, and stability binds before 52 dB. Neither
wall was visible from intuition; both showed up in an hour of simulation.

Rerun calibration after touching `specs.py`.

---

## Baseline results

Classical optimizers, 200-simulation budget, 5 seeds, `dev180`. **These are
harness-development numbers, not results** — `dev180` is not a calibrated
process (see [PDKs](#pdks)). They are shown because the *shape* is what matters
for benchmark design.

| spec | random | lhs | nelder-mead | de |
| --- | --- | --- | --- | --- |
| `ota5t-easy` | 5/5 | 5/5 | 5/5 | 5/5 |
| `ota5t-base` | 0/5 | 0/5 | **3/5** | 0/5 |
| `ota5t-hard` | 0/5 | 1/5 | **4/5** | 1/5 |
| `miller-easy` | 2/5 | 3/5 | 2/5 | **4/5** |
| `miller-base` | 0/5 | 0/5 | 0/5 | **1/5** |
| `miller-hard` | 0/5 | 0/5 | 0/5 | 0/5 |

Three things worth noting.

**No classical method dominates.** Nelder-Mead wins the single-stage tasks;
differential evolution wins the two-stage ones. A benchmark where one baseline
sweeps everything is measuring one thing; this one is measuring at least two.

**The budget is the whole game.** DE solves `ota5t-base` 3/3 at an 800-sim
budget and 0/5 at 200. Population methods spend their first hundred
simulations just filling a population. Any claim about an optimizer here is
meaningless without stating the budget, which is why the harness counts it in
one place.

**There is real headroom.** `miller-base` and `miller-hard` are essentially
unsolved at this budget. If a language model can use circuit knowledge to reach
them in 200 simulations, that is a result worth reporting — and if it cannot,
that is worth reporting too.

The LLM arm has not been run yet: it needs `ANTHROPIC_API_KEY`, and the numbers
above should be regenerated on a calibrated PDK before anything is published.

## Baselines

Implemented in pure numpy — no scipy, no solver-version drift, reproducible
from a seed.

- **`random`** — uniform sampling. The floor any method must clear.
- **`lhs`** — Latin hypercube. Stratified, so it cannot clump the way uniform
  draws do in 13 dimensions. A genuinely better random baseline.
- **`nelder-mead`** — multi-start simplex. Restarts matter: the objective is
  full of flat regions where the simulator fails to converge.
- **`de`** — differential evolution. The strongest classical baseline here.

These are implemented to win, not to lose. A benchmark whose baselines are
strawmen proves nothing.

---

## The agent

`bias/agent.py`. The model sees exactly what a designer sees: the topology, the
parameters it may set with their ranges, the spec, and the measured result of
every candidate so far. It does **not** see the netlist, the model cards, or
the scoring function's internals — only the same pass/fail table a human reads.

That constraint is the interesting part. The agent has to reason about circuit
behaviour ("gain is short and power is under budget, so lengthen the input pair
rather than raise the current") rather than pattern-match on an objective it
can differentiate.

It is also the shape a real deployment needs. Everything crossing the boundary
is scrubbed topology intent and numbers — no PDK content, no model parameters,
no file paths ever reach the model. That is the same separation
[SABLE](https://arxiv.org/pdf/2607.03701) argues industrial analog flows
require, and it is why this design would survive contact with a foundry NDA.

---

## PDKs

No PDK content is vendored here.

```bash
scripts/fetch_pdk.sh ihp-sg13g2    # IHP SG13G2 130nm BiCMOS
scripts/fetch_pdk.sh sky130        # SkyWater 130nm
```

`dev180` is a small hand-written BSIM3v3 card used to bring the harness up and
to give CI something deterministic. **It is not a calibrated process** — the
parameters are plausible for generic 180 nm bulk CMOS but correspond to no real
fab. `bench` refuses to present its output as publishable, and you should
treat any number from it as a smoke test.

Adding a PDK is a config entry: model file, device names, VDD, minimum
geometry. Pointing this at a proprietary PDK behind a company firewall is the
same change.

---

## Limitations

Stated plainly, because a benchmark that oversells itself is worse than none.

- **Topology is fixed.** This sizes circuits; it does not invent them. Topology
  synthesis is a different and harder problem.
- **Nominal corner only.** One temperature, typical models, no process corners,
  no Monte Carlo mismatch. Real sizing must survive all of these, and a point
  that passes here is not a tapeout-ready design.
- **Schematic only.** No layout, no parasitic extraction. The gap between
  schematic and post-layout is where much of analog design actually lives.
- **Two topologies.** Enough to show a method works or does not; not enough to
  claim generality.
- **The dev PDK is not physical.** See above.

Each of these is a direction, not an excuse. Corners and mismatch are the
natural next addition, because they change which sizings are actually good.

---

## Layout

```
bias/
  spec.py         specifications and scoring    (no SPICE knowledge)
  pdk.py          process abstraction
  topology.py     circuits and netlist emission
  sim.py          ngspice driver
  evaluate.py     the single evaluation path -- budget counting lives here
  optimizers.py   classical baselines
  agent.py        the LLM agent
  netlist.py      circuit IR: validation, emission, graph identity
  logic.py        deterministic static-CMOS synthesis
  verify.py       truth-table proof by transient simulation
  symbol.py       SVG schematic symbols
  design.py       natural language -> verified circuit
  specs.py        the frozen benchmark suite
  bench.py        the harness
scripts/
  calibrate.py    check every spec is reachable
  fetch_pdk.sh    download open PDKs
```

## Tests

```bash
.venv/bin/python -m pytest
```

The tests that matter are in `tests/test_scoring.py::TestPhaseMargin`. Phase
margin was the first real bug: the 5T OTA is non-inverting at DC and the
two-stage Miller is inverting, so the obvious `180 + phase(f_unity)` is correct
for one and silently wrong for the other — it reported a 231° phase margin on
an amplifier whose output was railed. SPICE now measures raw phase and Python
derives the margin with proper unwrapping.

## Licence

MIT for the code. PDKs carry their own licences (both supported open PDKs are
Apache-2.0) and are not redistributed here.
