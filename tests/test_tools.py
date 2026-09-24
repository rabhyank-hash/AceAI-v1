import copy
import json

import pytest

from aceai.config import DATA_RAW
from aceai.ingest import load_all
from aceai.ingest.ground_truth import extract_ground_truth, ground_truth_to_output
from aceai.schemas import LearningObjective, Module, SequencerOutput
from aceai.tools import (
    build_module_graph,
    check_cycles,
    check_module_order,
    check_provenance,
    topo_sort_modules,
    validate_lo,
    validate_output,
)


def lo(
    id, deps=(), bloom="C2", scope="atomic", parent=None, sources=None, **kw
) -> LearningObjective:
    return LearningObjective(
        id=id,
        raw_text=f"Text of {id}.",
        source_ids=sources if sources is not None else [f"r-{id}"],
        verb="explain",
        bloom_level=bloom,
        track="conceptual",
        target_concept=id,
        scope=scope,
        parent_id=parent,
        depends_on=list(deps),
        **kw,
    )


def mod(id, order, lo_ids, deps=(), head=None) -> Module:
    return Module(
        id=id, title=id, order=order, lo_ids=lo_ids, depends_on=list(deps), aggregate_lo_id=head
    )


def output(modules, los) -> SequencerOutput:
    return SequencerOutput(
        modules=modules,
        los=los,
        provenance=[{"raw_id": s, "lo_id": x.id} for x in los for s in x.source_ids],
    )


def codes(result, kind="errors") -> list[str]:
    return [i.code for i in getattr(result, kind)]


@pytest.fixture
def good() -> SequencerOutput:
    los = [
        lo("A", scope="aggregate", bloom="C3"),
        lo("a1", parent="A", bloom="C1"),
        lo("a2", parent="A", deps=["a1"], bloom="C3"),
        lo("b1", deps=["a2"], bloom="C2"),
    ]
    return output([mod("m1", 1, ["a1", "a2"], head="A"), mod("m2", 2, ["b1"], deps=["m1"])], los)


# --- results are LLM-readable -------------------------------------------------------------------


def test_results_serialize_to_json(good):
    r = check_cycles(good.los)
    data = json.loads(r.model_dump_json())
    assert set(data) >= {"ok", "errors", "warnings", "cycles"}


# --- validate -----------------------------------------------------------------------------------


def test_valid_output_passes(good):
    r = validate_output(good)
    assert r.ok, r.errors
    assert r.warnings == []


def test_validate_lo_reports_schema_errors_instead_of_raising():
    data = lo("x").model_dump() | {"bloom_level": "C9", "track": "practical"}
    r = validate_lo(data)
    assert not r.ok
    assert codes(r) == ["schema", "schema"]
    assert any("bloom_level" in e.message for e in r.errors)


def test_validate_lo_depth_floor_above_level():
    r = validate_lo(lo("x", bloom="C2", depth_floor="C4"))
    assert codes(r) == ["depth_floor_above_level"]
    assert validate_lo(lo("x", bloom="C4", depth_floor="C2")).ok


def test_validate_output_from_invalid_dict():
    r = validate_output({"modules": [], "los": [{"id": "x"}], "provenance": []})
    assert not r.ok and set(codes(r)) == {"schema"}


def test_aggregate_without_children():
    out = output([mod("m1", 1, ["a1"], head="A")], [lo("A", scope="aggregate"), lo("a1")])
    r = validate_output(out)
    assert codes(r) == ["aggregate_without_children"]
    assert r.errors[0].ids == ["A"]


def test_reference_errors():
    los = [
        lo("A", scope="aggregate"),
        lo("a1", parent="A", deps=["ghost"]),
        lo("a2", parent="a1"),  # parent is atomic
        lo("a3", parent="nobody"),
        lo("loose"),  # atomic, in no module
    ]
    mods = [
        mod("m1", 1, ["a1", "a2", "a3", "missing"], deps=["m9"], head="A"),
        mod("m2", 2, ["a1"]),  # a1 in two modules
    ]
    r = validate_output(output(mods, los))
    assert set(codes(r)) == {
        "unknown_dependency",
        "parent_not_aggregate",
        "unknown_parent",
        "atomic_lo_unplaced",
        "unknown_module_member",
        "unknown_module_dependency",
        "lo_in_many_modules",
    }


def test_own_ancestor_loop_reported_once():
    los = [
        lo("A", scope="aggregate", parent="B"),
        lo("B", scope="aggregate", parent="A"),
        lo("a1", parent="A"),
    ]
    r = validate_output(output([mod("m1", 1, ["a1", "A", "B"])], los))
    assert codes(r) == ["own_ancestor"]
    assert sorted(r.errors[0].ids) == ["A", "B"]


def test_duplicate_ids_and_unsorted_modules():
    los = [lo("x"), lo("x", sources=["r-x2"])]
    r = validate_output(output([mod("m2", 2, ["x"]), mod("m1", 1, [])], los))
    assert "duplicate_lo_id" in codes(r)
    assert "modules_not_sorted" in codes(r)
    r = validate_output(output([mod("m1", 1, ["x"]), mod("m2", 1, [])], [lo("x")]))
    assert codes(r) == ["duplicate_module_order"]


def test_head_children_elsewhere_is_a_warning():
    los = [lo("A", scope="aggregate"), lo("a1", parent="A"), lo("a2", parent="A")]
    r = validate_output(output([mod("m1", 1, ["a1"], head="A"), mod("m2", 2, ["a2"])], los))
    assert r.ok
    assert codes(r, "warnings") == ["head_children_elsewhere"]


# --- provenance ---------------------------------------------------------------------------------


def test_provenance_ok(good):
    raw = [s for x in good.los for s in x.source_ids]
    assert check_provenance(raw, good).ok


def test_provenance_missing_duplicated_unknown():
    los = [lo("x", sources=["r1", "r2"]), lo("y", sources=["r2"])]
    out = SequencerOutput(
        modules=[mod("m1", 1, ["x", "y"])],
        los=los,
        provenance=[
            {"raw_id": "r1", "lo_id": "x"},
            {"raw_id": "r2", "lo_id": "x"},
            {"raw_id": "r2", "lo_id": "y"},  # duplicated mapping
            {"raw_id": "rZ", "lo_id": "gone"},  # unknown raw + unknown LO
        ],
    )
    r = check_provenance(["r1", "r2", "r3"], out)
    assert not r.ok
    assert r.missing == ["r3"]
    assert r.duplicated == {"r2": ["x", "y"]}
    assert r.unknown_raw == ["rZ"]
    assert r.unknown_lo == ["gone"]
    assert set(codes(r)) == {"missing", "duplicated", "unknown_raw", "unknown_lo"}


def test_provenance_source_ids_mismatch():
    out = SequencerOutput(
        modules=[mod("m1", 1, ["x", "y"])],
        los=[lo("x", sources=["r1"]), lo("y", sources=["r2"])],
        provenance=[{"raw_id": "r1", "lo_id": "y"}, {"raw_id": "r2", "lo_id": "y"}],
    )
    r = check_provenance(["r1", "r2"], out)
    assert codes(r) == ["source_ids_mismatch", "source_ids_mismatch"]


def test_provenance_accepts_models_with_ids(good):
    from aceai.ingest.agent1_input import InputLO

    raw = [InputLO(id=s, text="t") for x in good.los for s in x.source_ids]
    assert check_provenance(raw, good).ok


# --- cycles -------------------------------------------------------------------------------------


def test_no_cycles(good):
    r = check_cycles(good.los)
    assert r.ok and r.cycles == []


def test_cycle_reported_with_path():
    los = [lo("c", deps=["a"]), lo("a", deps=["b"]), lo("b", deps=["c"]), lo("d", deps=["a"])]
    r = check_cycles(los)
    assert not r.ok
    assert r.cycles == [["a", "b", "c"]]  # a depends on b depends on c depends on a
    assert "a depends on b depends on c depends on a" in r.errors[0].message


def test_two_cycles_sorted_and_unknown_deps_ignored():
    los = [
        lo("x", deps=["y", "ghost"]),
        lo("y", deps=["x"]),
        lo("p", deps=["q"]),
        lo("q", deps=["p"]),
    ]
    r = check_cycles(los)
    assert r.cycles == [["p", "q"], ["x", "y"]]


# --- module order -------------------------------------------------------------------------------


def test_module_order_ok(good):
    r = check_module_order(good.modules, good.los)
    assert r.ok and r.warnings == []


def test_module_depends_on_later_module():
    mods = [mod("m1", 1, ["a"], deps=["m2"]), mod("m2", 2, ["b"])]
    r = check_module_order(mods, [lo("a"), lo("b")])
    assert codes(r) == ["module_depends_on_later_module"]
    assert r.errors[0].ids == ["m1", "m2"]


def test_lo_depends_on_later_module_and_later_lo():
    los = [lo("a", deps=["b"]), lo("a2", deps=["a3"]), lo("a3"), lo("b")]
    mods = [mod("m1", 1, ["a", "a2", "a3"]), mod("m2", 2, ["b"])]
    r = check_module_order(mods, los)
    assert sorted(codes(r)) == ["lo_depends_on_later_lo", "lo_depends_on_later_module"]


def test_order_uses_order_field_not_list_position():
    mods = [mod("m2", 2, ["b"]), mod("m1", 1, ["a"], deps=["m2"])]
    r = check_module_order(mods, [lo("a"), lo("b")])
    assert codes(r) == ["module_depends_on_later_module"]


def test_bloom_decrease_is_warning():
    los = [lo("a", bloom="C3"), lo("b", bloom="C1"), lo("c", bloom="C4")]
    r = check_module_order([mod("m1", 1, ["a", "b", "c"])], los)
    assert r.ok
    assert codes(r, "warnings") == ["bloom_decreases_in_module"]
    assert "a (C3) -> b (C1)" in r.warnings[0].message


# --- module graph -------------------------------------------------------------------------------


def test_build_module_graph_aggregates_lo_edges():
    los = [
        lo("a1"),
        lo("a2"),
        lo("b1", deps=["a1", "a2"]),
        lo("c1", deps=["b1"]),
        lo("x", deps=["c1"]),
    ]
    mods = [
        mod("m1", 1, ["a1", "a2"]),
        mod("m2", 2, ["b1"], deps=["m1"]),
        mod("m3", 3, ["c1"], deps=["m1"]),  # declared m1 without evidence; m2 undeclared
    ]
    r = build_module_graph(mods, los)
    edges = {(e.from_module, e.to_module): e for e in r.edges}
    assert set(edges) == {("m2", "m1"), ("m3", "m1"), ("m3", "m2")}
    assert edges[("m2", "m1")].lo_edges == [("b1", "a1"), ("b1", "a2")]
    assert edges[("m2", "m1")].declared
    assert edges[("m3", "m1")].lo_edges == [] and edges[("m3", "m1")].declared
    assert not edges[("m3", "m2")].declared
    assert r.ok
    assert sorted(codes(r, "warnings")) == [
        "declared_without_lo_evidence",
        "dependency_outside_modules",  # x is in no module
        "undeclared_module_dependency",
    ]


def test_intra_module_edges_are_not_module_edges():
    r = build_module_graph([mod("m1", 1, ["a", "b"])], [lo("a"), lo("b", deps=["a"])])
    assert r.edges == []


# --- topological sort ---------------------------------------------------------------------------


def test_topo_sort_unique_order():
    mods = [mod("m1", 1, [], deps=["m3"]), mod("m2", 2, [], deps=["m1"]), mod("m3", 3, [])]
    r = topo_sort_modules(mods)
    assert r.ok
    assert r.order == ["m3", "m1", "m2"]
    assert r.unique and r.ties == []
    assert not r.matches_current_order


def test_topo_sort_reports_ties():
    #   m1 <- m2, m1 <- m3, {m2, m3} <- m4 : m2 and m3 can go either way
    mods = [
        mod("m4", 4, [], deps=["m2", "m3"]),
        mod("m3", 3, [], deps=["m1"]),
        mod("m2", 2, [], deps=["m1"]),
        mod("m1", 1, []),
    ]
    r = topo_sort_modules(mods)
    assert r.order == ["m1", "m2", "m3", "m4"]  # tie broken by current order
    assert not r.unique
    assert [(t.position, t.candidates, t.chosen) for t in r.ties] == [(1, ["m2", "m3"], "m2")]
    assert r.matches_current_order
    # Deterministic regardless of input list order.
    assert topo_sort_modules(list(reversed(mods))) == r


def test_topo_sort_with_lo_edges():
    mods = [mod("m1", 1, ["a"]), mod("m2", 2, ["b"])]
    los = [lo("a", deps=["b"]), lo("b")]
    assert topo_sort_modules(mods).order == ["m1", "m2"]
    assert topo_sort_modules(mods, los).order == ["m2", "m1"]


def test_topo_sort_cycle():
    mods = [mod("m1", 1, [], deps=["m2"]), mod("m2", 2, [], deps=["m1"]), mod("m0", 0, [])]
    r = topo_sort_modules(mods)
    assert not r.ok
    assert r.order == ["m0"]
    assert codes(r) == ["module_cycle"]
    assert sorted(r.errors[0].ids) == ["m1", "m2"]


# --- tools never mutate inputs ------------------------------------------------------------------


def test_tools_do_not_mutate_inputs(good):
    before = copy.deepcopy(good.model_dump())
    raw = [s for x in good.los for s in x.source_ids]
    validate_output(good)
    check_provenance(raw, good)
    check_cycles(good.los)
    check_module_order(good.modules, good.los)
    build_module_graph(good.modules, good.los)
    topo_sort_modules(good.modules, good.los)
    assert good.model_dump() == before


# --- smoke test on the PPP ground truth ---------------------------------------------------------


def test_smoke_ppp_ground_truth():
    data = load_all(DATA_RAW)
    if "PPP" not in data:
        pytest.skip("PPP CSV not present in data/raw/")
    los = data["PPP"]
    out = ground_truth_to_output(extract_ground_truth("PPP", los), los)

    v = validate_output(out)
    # The 22 syllabus broad LOs are aggregate but the CSV says nothing about their children.
    assert set(codes(v)) == {"aggregate_without_children"}
    assert len(v.errors) == 22

    assert check_provenance(los, out).ok
    assert check_cycles(out.los).ok
    assert check_module_order(out.modules, out.los).ok
    assert build_module_graph(out.modules, out.los).edges == []
    t = topo_sort_modules(out.modules, out.los)
    assert t.ok and t.matches_current_order
    assert len(t.order) == 33 and len(t.ties) == 32  # no edges yet: every position is a tie
