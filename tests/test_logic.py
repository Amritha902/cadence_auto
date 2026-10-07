"""Logic synthesis, symbols, and request resolution.

The truth-table-by-simulation tests live here too and need ngspice; they are
the ones that actually prove a synthesised gate works.
"""

from __future__ import annotations

import re

import pytest

from bias import design, logic, pdk, symbol, verify


class TestSynthesis:
    @pytest.mark.parametrize("name", sorted(logic.CELLS))
    def test_every_cell_is_structurally_valid(self, name):
        circuit = logic.get(name)[0]()
        assert circuit.is_valid(), circuit.explain_problems()

    @pytest.mark.parametrize("name", sorted(logic.CELLS))
    def test_every_cell_declares_its_ports(self, name):
        builder, ins, outs, _ = logic.get(name)
        circuit = builder()
        for port in ins + outs:
            assert port in circuit.nets(), f"{port} is declared but never wired"

    def test_transistor_counts_match_the_design(self):
        counts = {
            name: len(logic.get(name)[0]().transistors()) for name in logic.CELLS
        }
        assert counts["inverter"] == 2
        assert counts["nand2"] == 4
        assert counts["nor2"] == 4
        assert counts["xor2"] == 16          # four NAND2
        # 18, not 20: carry reuses the XOR's first NAND and needs only an
        # inverter rather than a whole AND gate.
        assert counts["half_adder"] == 18

    def test_nand_series_stack_is_widened_by_depth(self):
        """Series NMOS must be wider so pull-down drive matches an inverter."""
        inv = logic.get("inverter")[0]()
        nand = logic.get("nand2")[0]()
        inv_wn = [d.params["W"] for d in inv.transistors()
                  if d.kind.value == "nmos"][0]
        nand_wn = [d.params["W"] for d in nand.transistors()
                   if d.kind.value == "nmos"][0]
        assert nand_wn == pytest.approx(2 * inv_wn)

    def test_nor_pmos_stack_is_widened_by_depth(self):
        nor = logic.get("nor2")[0]()
        pmos_w = [d.params["W"] for d in nor.transistors()
                  if d.kind.value == "pmos"]
        sizing = logic.Sizing()
        assert all(w == pytest.approx(2 * sizing.wp) for w in pmos_w)

    def test_pmos_is_wider_than_nmos(self):
        inv = logic.get("inverter")[0]()
        wn = [d.params["W"] for d in inv.transistors() if d.kind.value == "nmos"][0]
        wp = [d.params["W"] for d in inv.transistors() if d.kind.value == "pmos"][0]
        assert wp > wn

    def test_sizing_is_honoured(self):
        circuit = logic.inverter_cell(logic.Sizing(wn=4e-6, beta=3.0))
        widths = {d.kind.value: d.params["W"] for d in circuit.transistors()}
        assert widths["nmos"] == pytest.approx(4e-6)
        assert widths["pmos"] == pytest.approx(12e-6)


class TestTruthTables:
    def test_half_adder_reference_is_correct(self):
        rows = dict(
            (tuple(i.values()), o) for i, o in logic.truth_table("half_adder")
        )
        assert rows[(0, 0)] == {"sum": 0, "carry": 0}
        assert rows[(0, 1)] == {"sum": 1, "carry": 0}
        assert rows[(1, 0)] == {"sum": 1, "carry": 0}
        assert rows[(1, 1)] == {"sum": 0, "carry": 1}

    def test_full_adder_reference_counts_ones(self):
        for inputs, out in logic.truth_table("full_adder"):
            total = sum(inputs.values())
            assert out["sum"] == total % 2
            assert out["cout"] == (1 if total >= 2 else 0)

    def test_table_covers_every_combination(self):
        assert len(logic.truth_table("full_adder")) == 8
        assert len(logic.truth_table("half_adder")) == 4


class TestResolution:
    @pytest.mark.parametrize("request_text,expected", [
        ("build me a half adder symbol", "half_adder"),
        ("half-adder", "half_adder"),
        ("make a full adder", "full_adder"),
        ("I need an XOR gate", "xor2"),
        ("nand gate please", "nand2"),
        ("a not gate", "inverter"),
        ("inverter", "inverter"),
    ])
    def test_deterministic_matches(self, request_text, expected):
        cell, _how = design.resolve(request_text, use_llm=False)
        assert cell == expected

    def test_longest_alias_wins(self):
        """'full adder' must beat the shorter 'adder'."""
        assert design.resolve("full adder", use_llm=False)[0] == "full_adder"

    def test_unresolvable_request_raises_with_the_menu(self):
        with pytest.raises(LookupError, match="half_adder"):
            design.resolve("a 500MHz phase locked loop", use_llm=False)


class TestSymbol:
    def test_pins_are_inside_the_canvas(self):
        """Pin dots on the exact edge render clipped in half."""
        builder, ins, outs, _ = logic.get("half_adder")
        svg = symbol.Symbol.from_circuit(builder(), ins, outs).render()
        width = float(re.search(r'width="(\d+)"', svg).group(1))
        for x in re.findall(r'<circle class="dot" cx="([\d.]+)"', svg):
            assert 0 < float(x) < width

    def test_roles_are_inferred_from_ports(self):
        builder, ins, outs, _ = logic.get("half_adder")
        sym = symbol.Symbol.from_circuit(builder(), ins, outs)
        assert sym.inputs == ["a", "b"]
        assert sym.outputs == ["sum", "carry"]
        assert sym.power == ["vdd"]

    def test_svg_is_well_formed_and_labelled(self):
        builder, ins, outs, _ = logic.get("xor2")
        svg = symbol.Symbol.from_circuit(builder(), ins, outs).render()
        assert svg.startswith("<svg") and svg.rstrip().endswith("</svg>")
        assert svg.count("<svg") == 1 and svg.count("</svg>") == 1
        assert "xor2" in svg

    def test_dark_mode_is_defined(self):
        svg = symbol.Symbol("x", ["a"], ["y"], ["vdd"]).render()
        assert "prefers-color-scheme: dark" in svg


@pytest.mark.needs_ngspice
class TestSimulatedTruthTables:
    """The tests that prove the transistors are wired correctly."""

    @pytest.mark.parametrize("name", sorted(logic.CELLS))
    def test_cell_matches_its_truth_table(self, name):
        report = verify.verify(name, pdk.DEV180)
        assert report.passed, f"{name} failed:\n{report.table()}\n{report.reason}"

    @pytest.mark.parametrize("name", ["inverter", "nand2", "half_adder"])
    def test_outputs_reach_their_rails(self, name):
        """A gate settling mid-rail passes a threshold test but is broken."""
        report = verify.verify(name, pdk.DEV180)
        assert report.worst_rail_error < 0.05, (
            f"{name} worst rail error {report.worst_rail_error:.3f} of VDD"
        )

    def test_end_to_end_build_from_a_sentence(self, tmp_path):
        result = design.build(
            "build me a half adder symbol", pdk.DEV180,
            outdir=tmp_path, use_llm=False,
        )
        assert result.ok
        assert result.cell == "half_adder"
        for artifact in ("symbol", "netlist", "testbench", "truth"):
            assert result.files[artifact].exists()
        assert result.files["symbol"].read_text().startswith("<svg")


class TestExpandedLibrary:
    """The cells added to make "ask for it and get it" actually reach."""

    def test_reference_functions_are_right(self):
        ref = {n: dict((tuple(i.values()), o)
                       for i, o in logic.truth_table(n)) for n in logic.CELLS}

        assert ref["mux2"][(0, 1, 1)] == {"y": 1}     # s=1 selects b
        assert ref["mux2"][(0, 1, 0)] == {"y": 0}     # s=0 selects a
        assert ref["majority3"][(1, 1, 0)] == {"y": 1}
        assert ref["majority3"][(1, 0, 0)] == {"y": 0}
        assert ref["parity4"][(1, 1, 1, 0)] == {"y": 1}
        assert ref["parity4"][(1, 1, 0, 0)] == {"y": 0}
        assert ref["comparator1"][(1, 0)] == {"gt": 1, "eq": 0, "lt": 0}
        assert ref["comparator1"][(1, 1)] == {"gt": 0, "eq": 1, "lt": 0}
        assert ref["half_subtractor"][(0, 1)] == {"diff": 1, "borrow": 1}
        assert ref["half_subtractor"][(1, 1)] == {"diff": 0, "borrow": 0}

    def test_decoder_is_one_hot(self):
        for inputs, out in logic.truth_table("decoder2to4"):
            assert sum(out.values()) == 1, f"{inputs} is not one-hot: {out}"

    def test_full_subtractor_matches_arithmetic(self):
        for inputs, out in logic.truth_table("full_subtractor"):
            a, b, bi = inputs["a"], inputs["b"], inputs["bin"]
            expected = a - b - bi
            value = out["diff"] - 2 * out["borrow"]
            assert value == expected, f"{inputs} gave {out}"

    def test_adder2_matches_binary_addition(self):
        for inputs, out in logic.truth_table("adder2"):
            a = inputs["a1"] << 1 | inputs["a0"]
            b = inputs["b1"] << 1 | inputs["b0"]
            total = out["cout"] << 2 | out["s1"] << 1 | out["s0"]
            assert total == a + b + inputs["cin"], f"{inputs} gave {out}"

    def test_three_input_stacks_are_widened_by_depth(self):
        """A 3-high series stack needs 3x the width to keep drive constant.

        The complementary devices stay at nominal: in a NAND the PMOS are in
        parallel, in a NOR the NMOS are, and parallel devices need no widening.
        """
        base = logic.Sizing()

        def widths(cell: str, kind: str) -> list[float]:
            circuit = logic.get(cell)[0]()
            return sorted({d.params["W"] for d in circuit.transistors()
                           if d.kind.value == kind})

        assert widths("nand3", "nmos") == [pytest.approx(3 * base.wn)]
        assert widths("nand3", "pmos") == [pytest.approx(base.wp)]
        assert widths("nor3", "pmos") == [pytest.approx(3 * base.wp)]
        assert widths("nor3", "nmos") == [pytest.approx(base.wn)]

    @pytest.mark.parametrize("request_text,expected", [
        ("build me a mux", "mux2"),
        ("I need a 2:1 multiplexer", "mux2"),
        ("a 2 to 4 decoder", "decoder2to4"),
        ("make a full subtractor", "full_subtractor"),
        ("a magnitude comparator", "comparator1"),
        ("majority voter", "majority3"),
        ("parity checker", "parity4"),
        ("a 2 bit adder", "adder2"),
        ("ripple carry adder", "adder2"),
        ("3 input nand", "nand3"),
        ("an xnor gate", "xnor2"),
        ("a buffer", "buffer"),
        ("and gate", "and2"),
        ("or gate", "or2"),
    ])
    def test_new_aliases_resolve(self, request_text, expected):
        assert design.resolve(request_text, use_llm=False)[0] == expected

    def test_common_english_words_do_not_misroute(self):
        """'and' and 'or' are ordinary words; longer aliases must win."""
        assert design.resolve("build me a NAND gate", use_llm=False)[0] == "nand2"
        assert design.resolve("a NOR gate", use_llm=False)[0] == "nor2"
        assert design.resolve("a half adder", use_llm=False)[0] == "half_adder"

    def test_specific_beats_general(self):
        assert design.resolve("2 bit adder", use_llm=False)[0] == "adder2"
        assert design.resolve("full adder", use_llm=False)[0] == "full_adder"
        assert design.resolve("half subtractor", use_llm=False)[0] == "half_subtractor"
