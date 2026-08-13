"""Agent loop tests using a fake client. No API key, no network, no simulator.

These cover the parts that break in practice: a model that wraps its JSON in
prose, a model that emits a malformed reply, a model that repeats itself, and
the budget accounting that makes the benchmark comparable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pytest

from bias import pdk, specs, topology
from bias.agent import LLMAgent, _extract_json
from bias.evaluate import Evaluator
from bias.spec import Metric, Spec
from bias.topology import Testbench


# --- fake anthropic client -------------------------------------------------


@dataclass
class _Block:
    text: str
    type: str = "text"


@dataclass
class _Reply:
    content: list


class FakeClient:
    """Replays a scripted list of reply strings."""

    def __init__(self, replies: list[str]):
        self.replies = list(replies)
        self.prompts: list[str] = []

        class _Messages:
            def __init__(self, outer):
                self._outer = outer

            def create(self, **kwargs):
                self._outer.prompts.append(kwargs["messages"][0]["content"])
                if not self._outer.replies:
                    raise AssertionError("agent asked for more replies than scripted")
                return _Reply(content=[_Block(self._outer.replies.pop(0))])

        self.messages = _Messages(self)


def _proposal(topo, **overrides) -> str:
    values = dict(topo.defaults(), **overrides)
    return json.dumps({"reasoning": "test move", "values": values})


def _evaluator(budget: int = 5, spec: Spec | None = None) -> Evaluator:
    topo = topology.get("ota5t")
    return Evaluator(
        topo,
        pdk.DEV180,
        spec or specs.OTA5T_EASY,
        Testbench(cl=1e-12),
        budget=budget,
    )


# --- JSON extraction -------------------------------------------------------


class TestExtractJson:
    def test_bare_object(self):
        assert _extract_json('{"a": 1}') == '{"a": 1}'

    def test_wrapped_in_prose_and_fence(self):
        text = 'Here is my proposal:\n```json\n{"a": 1, "b": 2}\n```\nDone.'
        assert json.loads(_extract_json(text)) == {"a": 1, "b": 2}

    def test_nested_objects(self):
        text = 'x {"values": {"w1": 1e-6}, "r": "y"} z'
        assert json.loads(_extract_json(text))["values"] == {"w1": 1e-6}

    def test_braces_inside_strings_do_not_confuse_it(self):
        text = '{"reasoning": "use {this} shape", "values": {"w1": 1e-6}}'
        assert json.loads(_extract_json(text))["values"] == {"w1": 1e-6}

    def test_no_json_returns_none(self):
        assert _extract_json("no object here") is None


# --- parsing ---------------------------------------------------------------


class TestParsing:
    def test_missing_parameter_is_rejected(self):
        topo = topology.get("ota5t")
        agent = LLMAgent(client=FakeClient([]))
        partial = json.dumps({"values": {"w1": 1e-6}})
        with pytest.raises(ValueError, match="missing parameter"):
            agent._parse(partial, topo)

    def test_non_numeric_parameter_is_rejected(self):
        topo = topology.get("ota5t")
        agent = LLMAgent(client=FakeClient([]))
        bad = json.dumps({"values": dict(topo.defaults(), w1="wide")})
        with pytest.raises(ValueError, match="not a number"):
            agent._parse(bad, topo)

    def test_reasoning_is_captured(self):
        topo = topology.get("ota5t")
        agent = LLMAgent(client=FakeClient([]))
        _, reasoning = agent._parse(_proposal(topo), topo)
        assert reasoning == "test move"


# --- the loop --------------------------------------------------------------


class TestAgentLoop:
    def test_stops_as_soon_as_the_spec_is_met(self):
        """An easy spec the seed point already satisfies costs one simulation."""
        always_met = Spec(
            name="trivial", metrics=(Metric("gain", "max", 1.0, "dB"),)
        )
        ev = _evaluator(budget=10, spec=always_met)
        client = FakeClient([_proposal(ev.topology)])
        agent = LLMAgent(client=client)
        agent.run(ev, np.random.default_rng(0))

        # Seeded with defaults, which already pass -- so the model is never
        # asked and the run stops.
        assert ev.used == 1
        assert ev.solved() is not None
        assert client.prompts == []

    def test_respects_the_budget(self):
        impossible = Spec(
            name="impossible", metrics=(Metric("gain", "max", 1e9, "dB"),)
        )
        ev = _evaluator(budget=4, spec=impossible)
        topo = ev.topology
        # More replies than the budget allows; the agent must stop at 4.
        replies = [
            _proposal(topo, w1=w) for w in (1e-6, 2e-6, 3e-6, 4e-6, 5e-6, 6e-6)
        ]
        agent = LLMAgent(client=FakeClient(replies))
        agent.run(ev, np.random.default_rng(0))
        assert ev.used == 4

    def test_malformed_replies_cost_a_turn_not_the_run(self):
        impossible = Spec(
            name="impossible", metrics=(Metric("gain", "max", 1e9, "dB"),)
        )
        ev = _evaluator(budget=3, spec=impossible)
        topo = ev.topology
        agent = LLMAgent(
            client=FakeClient(
                [
                    "not json at all",
                    _proposal(topo, w1=2e-6),
                    "broken {again",
                    _proposal(topo, w1=3e-6),
                ]
            )
        )
        agent.run(ev, np.random.default_rng(0))
        # Seed + two valid proposals landed despite the two bad replies
        # interleaved between them.
        assert ev.used == 3

    def test_three_consecutive_bad_replies_ends_the_run(self):
        impossible = Spec(
            name="impossible", metrics=(Metric("gain", "max", 1e9, "dB"),)
        )
        ev = _evaluator(budget=20, spec=impossible)
        agent = LLMAgent(client=FakeClient(["junk", "junk", "junk"]))
        agent.run(ev, np.random.default_rng(0))
        assert ev.used == 1  # only the seed

    def test_repeated_proposal_is_cached_and_not_recharged(self):
        impossible = Spec(
            name="impossible", metrics=(Metric("gain", "max", 1e9, "dB"),)
        )
        ev = _evaluator(budget=3, spec=impossible)
        topo = ev.topology
        same = _proposal(topo, w1=5e-6)
        agent = LLMAgent(client=FakeClient([same, same, _proposal(topo, w1=9e-6)]))
        agent.run(ev, np.random.default_rng(0))
        # seed + two distinct points = 3 charged; the repeat was free.
        assert ev.used == 3
        assert sum(1 for e in ev.history if e.cached) == 1

    def test_prompt_never_leaks_pdk_internals(self):
        """The boundary that makes this deployable behind an NDA."""
        impossible = Spec(
            name="impossible", metrics=(Metric("gain", "max", 1e9, "dB"),)
        )
        ev = _evaluator(budget=3, spec=impossible)
        topo = ev.topology
        agent = LLMAgent(
            client=FakeClient([_proposal(topo, w1=2e-6), _proposal(topo, w1=3e-6)])
        )
        agent.run(ev, np.random.default_rng(0))

        assert agent._client.prompts, "expected at least one prompt"
        for prompt in agent._client.prompts:
            assert ".lib" not in prompt
            assert ".include" not in prompt
            assert "pdks/" not in prompt
            assert "vth0" not in prompt.lower()
            assert "bsim" not in prompt.lower()
            assert pdk.DEV180.nmos not in prompt
            assert pdk.DEV180.pmos not in prompt

    def test_turns_record_reasoning(self):
        impossible = Spec(
            name="impossible", metrics=(Metric("gain", "max", 1e9, "dB"),)
        )
        ev = _evaluator(budget=2, spec=impossible)
        agent = LLMAgent(client=FakeClient([_proposal(ev.topology, w1=2e-6)]))
        agent.run(ev, np.random.default_rng(0))
        assert agent.turns
        assert agent.turns[0].reasoning == "test move"
