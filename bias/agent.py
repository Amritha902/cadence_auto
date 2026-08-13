"""The LLM sizing agent.

The agent sees exactly what a designer would: the topology, the parameters it
may set with their ranges, the spec, and the measured result of every candidate
it has tried. It does not see the netlist, the models, or the score function's
internals -- only the same pass/fail table a human reads.

That constraint is deliberate and it is the interesting part of this benchmark.
The agent has to reason about circuit behaviour ("gain is short and power is
under budget, so lengthen the input pair rather than raise the current") rather
than pattern-match on an objective it can differentiate.

It is also the shape a real deployment needs: everything crossing the boundary
here is scrubbed topology intent and numbers. No PDK content, no model cards,
no file paths ever reach the model -- the same separation the SABLE work argues
industrial flows require.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field

import numpy as np

from .evaluate import BudgetExhausted, Evaluator, Evaluation
from .optimizers import Optimizer
from .spec import Spec
from .topology import Topology

DEFAULT_MODEL = "claude-opus-4-5"

SYSTEM = """You are an expert analog IC designer sizing a circuit against a \
specification. You work the way a designer works at a terminal: propose a \
sizing, read the simulated result, and reason about what to change next.

You will be given a topology, the parameters you may set with their legal \
ranges, the target specification, and the measured result of every candidate \
tried so far.

Rules:
- Reply with ONE JSON object and nothing else: {"reasoning": "...", "values": \
{"param": number, ...}}.
- "values" must contain every parameter, in SI base units (metres, amps, \
farads, ohms). 2.5e-6 means 2.5 microns.
- "reasoning" is one or two sentences on the circuit-level change you are \
making and why. Reference the measured numbers.
- Stay inside the stated ranges. Values outside them are clipped.

Design guidance that matters here:
- Gain rises with channel length (higher output resistance) and with gm/Id \
(lower overdrive), and falls with current.
- GBW is roughly gm1 / (2*pi*CL) for a single stage, or gm1 / (2*pi*Cc) for a \
Miller-compensated two-stage. Raising current or width raises gm.
- Power is set almost entirely by the bias current.
- In a two-stage amplifier the second stage's load current must match what its \
driver sinks, or the output sits at a rail and gain collapses. If vout_margin \
is near zero, fix the bias before chasing performance.
- For Miller compensation, phase margin needs Cc large enough relative to CL, \
and the nulling resistor near 1/gm of the second stage.

Do not repeat a candidate you have already tried. If a direction is not \
working after a few attempts, change strategy rather than taking smaller steps \
in the same direction."""


@dataclass
class Turn:
    """One agent proposal and what it measured. Kept for the transcript."""

    index: int
    reasoning: str
    values: dict[str, float]
    score: float
    satisfied: bool


class LLMAgent(Optimizer):
    name = "llm"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        max_tokens: int = 1200,
        history_window: int = 12,
        seed_with_defaults: bool = True,
        client=None,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.history_window = history_window
        self.seed_with_defaults = seed_with_defaults
        self._client = client
        self.turns: list[Turn] = []

    # -- client ----------------------------------------------------------

    def _get_client(self):
        if self._client is not None:
            return self._client
        try:
            import anthropic
        except ImportError as exc:
            raise RuntimeError(
                "the llm optimizer needs the anthropic package: pip install anthropic"
            ) from exc
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set; the llm optimizer cannot run"
            )
        self._client = anthropic.Anthropic()
        return self._client

    # -- prompt construction ---------------------------------------------

    def _describe_problem(self, topo: Topology, spec: Spec, ev: Evaluator) -> str:
        lines = [
            f"TOPOLOGY: {topo.name} -- {topo.description}",
            "",
            "PARAMETERS YOU MAY SET:",
        ]
        for p in topo.params:
            lines.append(
                f"  {p.name}: {p.description} "
                f"[{p.lo:.3g} .. {p.hi:.3g}] {p.unit}"
            )
        lines += [
            "",
            f"LOAD: CL = {ev.testbench.cl:.3g} F, "
            f"VDD = {ev.pdk.vdd:g} V, T = {ev.testbench.temp:g} C",
            "",
            f"TARGET SPEC ({spec.name}): {spec.description}",
        ]
        for m in spec.metrics:
            rel = ">=" if m.direction == "max" else "<="
            kind = " [objective, keep improving]" if m.objective else " [must meet]"
            lines.append(f"  {m.name} {rel} {m.target:.4g} {m.unit}{kind}")
        lines += [
            "",
            "vout_margin is the DC output's distance from the nearer supply "
            "rail as a fraction of VDD. Below ~0.1 the amplifier is railed and "
            "its gain and GBW numbers are meaningless.",
        ]
        return "\n".join(lines)

    def _describe_history(self, ev: Evaluator, spec: Spec) -> str:
        real = [e for e in ev.history if not e.cached]
        if not real:
            return "No candidates tried yet. Propose a sensible starting point."

        shown = real[-self.history_window :]
        best = min(real, key=lambda e: e.score)

        out = [f"TRIED {len(real)} candidate(s). Lower score is better; 0 means all specs met."]
        if best not in shown:
            out.append("\nBEST SO FAR:")
            out.append(self._format_eval(best, spec))
        out.append(f"\nMOST RECENT {len(shown)}:")
        for e in shown:
            out.append(self._format_eval(e, spec))
        return "\n".join(out)

    def _format_eval(self, e: Evaluation, spec: Spec) -> str:
        vals = " ".join(f"{k}={v:.4g}" for k, v in e.values.items())
        head = f"\n  #{e.index} score={e.score:.4f}"
        if not e.ok:
            head += f"  SIMULATION FAILED ({e.reason or 'no convergence'})"
        return f"{head}\n    sizing: {vals}\n{spec.report(e.measured)}"

    # -- response parsing -------------------------------------------------

    def _parse(self, text: str, topo: Topology) -> tuple[dict[str, float], str]:
        """Pull the JSON proposal out of the reply.

        Models sometimes wrap JSON in prose or a code fence; take the outermost
        brace-balanced object rather than trusting the whole string to parse.
        """
        blob = _extract_json(text)
        if blob is None:
            raise ValueError(f"no JSON object in reply: {text[:200]!r}")
        data = json.loads(blob)
        raw = data.get("values", data)
        if not isinstance(raw, dict):
            raise ValueError("'values' is not an object")

        values: dict[str, float] = {}
        for p in topo.params:
            if p.name not in raw:
                raise ValueError(f"missing parameter {p.name!r}")
            try:
                values[p.name] = float(raw[p.name])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"parameter {p.name!r} is not a number") from exc
        return values, str(data.get("reasoning", "")).strip()

    # -- main loop --------------------------------------------------------

    def run(self, ev: Evaluator, rng: np.random.Generator) -> None:
        client = self._get_client()
        topo, spec = ev.topology, ev.spec
        problem = self._describe_problem(topo, spec, ev)

        # Seed with the neutral midpoint so the model's first proposal is
        # informed by a real measurement rather than a guess. Classical
        # optimizers get their first sample for free too, so this costs the
        # agent one simulation from the same budget.
        if self.seed_with_defaults and not ev.exhausted:
            try:
                seed_eval = ev.evaluate(topo.defaults())
            except BudgetExhausted:
                return
            # An easy spec can already be met at the neutral midpoint. Stop
            # here rather than spending API calls re-solving a solved problem.
            if seed_eval.satisfied:
                return

        consecutive_bad_replies = 0

        while not ev.exhausted:
            prompt = (
                f"{problem}\n\n{'=' * 60}\n\n{self._describe_history(ev, spec)}\n\n"
                f"Simulations remaining: {ev.budget - ev.used}.\n"
                "Propose the next sizing as a single JSON object."
            )

            try:
                reply = client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    system=SYSTEM,
                    messages=[{"role": "user", "content": prompt}],
                )
                text = "".join(
                    block.text for block in reply.content if block.type == "text"
                )
                values, reasoning = self._parse(text, topo)
                consecutive_bad_replies = 0
            except (ValueError, json.JSONDecodeError):
                # A malformed reply costs a turn, not the run. Three in a row
                # means something is structurally wrong -- stop and let the
                # bench record however far it got.
                consecutive_bad_replies += 1
                if consecutive_bad_replies >= 3:
                    return
                continue

            try:
                e = ev.evaluate(values)
            except BudgetExhausted:
                return

            self.turns.append(
                Turn(e.index, reasoning, e.values, e.score, e.satisfied)
            )
            if e.satisfied:
                return


def _extract_json(text: str) -> str | None:
    """Outermost balanced {...}, ignoring braces inside strings."""
    start = text.find("{")
    if start < 0:
        return None
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None
