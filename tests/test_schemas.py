import json

import pytest
from pydantic import ValidationError

from aceai.schemas import (
    BloomLevel,
    CsvSource,
    LearningObjective,
    Module,
    ModuleType,
    RawLO,
    SequencerOutput,
    SyllabusLevel,
    SyllabusSource,
)


def lo(**overrides) -> dict:
    base = {
        "id": "lo-1",
        "raw_text": "Explain the role of the pandas library.",
        "source_ids": ["dataeng-u00-concept-pandas-lo01"],
        "verb": "explain",
        "bloom_level": "C2",
        "track": "conceptual",
        "target_concept": "pandas",
        "scope": "atomic",
    }
    return base | overrides


# --- BloomLevel ---------------------------------------------------------------------------------


def test_bloom_orders_by_number():
    assert BloomLevel.C1 < BloomLevel.C2 < BloomLevel.C6
    assert BloomLevel.C6 > BloomLevel.C3
    assert BloomLevel.C4 <= BloomLevel.C4 and BloomLevel.C4 >= BloomLevel.C4
    assert sorted([BloomLevel.C5, BloomLevel.C1, BloomLevel.C3]) == [
        BloomLevel.C1,
        BloomLevel.C3,
        BloomLevel.C5,
    ]
    assert max(BloomLevel) is BloomLevel.C6


def test_bloom_number_label_roundtrip():
    assert BloomLevel.C3.number == 3
    assert BloomLevel.C3.label == "Apply"
    assert [b.label for b in BloomLevel] == [
        "Remember",
        "Understand",
        "Apply",
        "Analyze",
        "Evaluate",
        "Create",
    ]
    assert all(BloomLevel.from_number(b.number) is b for b in BloomLevel)
    with pytest.raises(ValueError):
        BloomLevel.from_number(7)


def test_bloom_does_not_order_against_plain_strings():
    with pytest.raises(TypeError):
        _ = BloomLevel.C2 < "C3"
    with pytest.raises(TypeError):
        _ = "C3" > BloomLevel.C2
    assert BloomLevel.C2 == "C2"  # equality with the serialized value still holds


# --- LearningObjective --------------------------------------------------------------------------


def test_valid_lo_and_json_roundtrip():
    obj = LearningObjective(**lo(depends_on=["lo-0"], priority=1, depth_floor="C1"))
    assert obj.bloom_level is BloomLevel.C2
    dumped = json.loads(obj.model_dump_json())
    assert dumped["bloom_level"] == "C2"
    assert LearningObjective.model_validate(dumped) == obj


@pytest.mark.parametrize(
    "overrides",
    [
        {"bloom_level": "C7"},
        {"bloom_level": "Apply"},
        {"track": "practical"},
        {"scope": "composite"},
        {"priority": 0},
        {"priority": 6},
        {"depth_floor": "C0"},
        {"id": ""},
        {"parent_id": "lo-1"},  # own parent
        {"depends_on": ["lo-1"]},  # depends on itself
        {"depends_on": ["lo-2", "lo-2"]},
        {"source_ids": ["a", "a"]},
        {"unexpected_field": 1},
    ],
)
def test_invalid_lo(overrides):
    with pytest.raises(ValidationError):
        LearningObjective(**lo(**overrides))


def test_lo_missing_required_field():
    data = lo()
    del data["verb"]
    with pytest.raises(ValidationError):
        LearningObjective(**data)


# --- RawLO --------------------------------------------------------------------------------------


def test_raw_lo_csv_and_syllabus_sources():
    csv_lo = RawLO(
        raw_id="ppp-u03-concept-data-structures-lo02",
        course="PPP",
        text="Describe lists.",
        original_text=" Describe  lists. ",
        source={
            "kind": "csv",
            "unit_no": 3,
            "unit_name": "Data Structures ",
            "module_type": "CONCEPT",
            "module_name": "Data Structures",
            "lo_no": 2,
        },
    )
    assert isinstance(csv_lo.source, CsvSource)
    assert csv_lo.source.module_type is ModuleType.CONCEPT

    syl = RawLO(
        raw_id="ppp-syllabus-course-goal-lo01",
        course="PPP",
        text="Explain Python.",
        original_text="Explain Python.",
        source={"kind": "syllabus", "level": "course_goal", "lo_no": 1},
    )
    assert isinstance(syl.source, SyllabusSource)
    assert syl.source.level is SyllabusLevel.COURSE_GOAL


def test_raw_lo_rejects_bad_module_type_and_stray_fields():
    base = {
        "raw_id": "x",
        "course": "PPP",
        "text": "t",
        "original_text": "t",
        "source": {
            "kind": "csv",
            "unit_no": 0,
            "unit_name": "u",
            "module_type": "LAB",
            "module_name": "m",
            "lo_no": 1,
        },
    }
    with pytest.raises(ValidationError):
        RawLO(**base)
    # Ground-truth fields may only live under `source`.
    base["source"]["module_type"] = "PRIMER"
    with pytest.raises(ValidationError):
        RawLO(**base, unit_no=0)


# --- Module / SequencerOutput -------------------------------------------------------------------


def test_invalid_modules():
    with pytest.raises(ValidationError):
        Module(id="m1", title="T", order=0, lo_ids=["a"], depends_on=["m1"])
    with pytest.raises(ValidationError):
        Module(id="m1", title="T", order=0, lo_ids=["a", "a"])
    with pytest.raises(ValidationError):
        Module(id="m1", title="T", order=-1, lo_ids=["a"])


def test_sequencer_output_keeps_duplicate_provenance_visible():
    out = SequencerOutput(
        modules=[Module(id="m1", title="T", order=0, lo_ids=["lo-1"])],
        los=[LearningObjective(**lo())],
        provenance=[
            {"raw_id": "r1", "lo_id": "lo-1"},
            {"raw_id": "r1", "lo_id": "lo-2"},
        ],
    )
    assert out.provenance_map() == {"r1": ["lo-1", "lo-2"]}
