import copy
import json

import pytest

from aceai.config import DATA_RAW
from aceai.ingest import load_all
from aceai.ingest.ground_truth import extract_ground_truth, ground_truth_to_output
from aceai.schemas import LearningObjective, Module, SequencerOutput
from aceai.tools import check_provenance, validate_lo, validate_output


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
    r = check_provenance([s for x in good.los for s in x.source_ids], good)
    data = json.loads(r.model_dump_json())
    assert set(data) >= {"ok", "errors", "warnings", "missing"}


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


# --- tools never mutate inputs ------------------------------------------------------------------


def test_tools_do_not_mutate_inputs(good):
    before = copy.deepcopy(good.model_dump())
    raw = [s for x in good.los for s in x.source_ids]
    validate_output(good)
    check_provenance(raw, good)
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


def test_schema_error_names_the_item_id():
    data = {
        "modules": [],
        "provenance": [],
        "los": [{"id": "A", "raw_text": "t", "source_ids": ["A"]}],
    }
    r = validate_output(data)
    assert not r.ok
    assert any(
        e.code == "schema" and "item A: verb" in e.message and e.ids == ["A"] for e in r.errors
    )


def test_dependency_on_merged_lo_names_the_survivor():
    out = SequencerOutput(
        modules=[mod("m1", 1, ["a", "b"])],
        los=[lo("a", sources=["r-a", "x"]), lo("b", deps=["x"])],
        provenance=[],
    )
    r = validate_output(out)
    errs = [e for e in r.errors if e.code == "dependency_on_merged_lo"]
    assert len(errs) == 1 and "merged into a" in errs[0].message
    assert not any(e.code == "unknown_dependency" for e in r.errors)
