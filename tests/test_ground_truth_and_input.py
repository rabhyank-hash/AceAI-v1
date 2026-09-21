import json
import re

import pytest

from aceai.config import DATA_RAW
from aceai.ingest import IngestError, load_all
from aceai.ingest.agent1_input import Agent1Input, input_id, make_agent1_input, strip_lo_marker
from aceai.ingest.ground_truth import extract_ground_truth
from aceai.schemas import CsvSource, RawLO, SyllabusLevel, SyllabusSource


def csv_lo(course, unit, mtype, module, no, text, unit_name="Unit") -> RawLO:
    slug = module.lower().replace(" ", "-")
    return RawLO(
        raw_id=f"{course.lower()}-u{unit:02d}-{mtype.lower()}-{slug}-lo{no:02d}",
        course=course,
        text=text,
        original_text=text,
        source=CsvSource(
            unit_no=unit, unit_name=unit_name, module_type=mtype, module_name=module, lo_no=no
        ),
    )


def broad_lo(course, level, no, text) -> RawLO:
    return RawLO(
        raw_id=f"{course.lower()}-syllabus-{level.replace('_', '-')}-lo{no:02d}",
        course=course,
        text=text,
        original_text=text,
        source=SyllabusSource(level=level, lo_no=no),
    )


@pytest.fixture
def small_course() -> list[RawLO]:
    return [
        csv_lo("P", 0, "CONCEPT", "Intro", 1, "Explain what a program is.", "getting started"),
        csv_lo("P", 0, "CONCEPT", "Intro", 2, "Define a variable. (LO2)", "getting started"),
        csv_lo("P", 0, "PROJECT", "Intro", 1, "Write a script.", "getting started"),
        csv_lo("P", 3, "PRIMER", "Data Structures", 1, "Describe lists.", "data structures"),
        broad_lo("P", "course_goal", 1, "Use Python to solve problems."),
    ]


# --- Ground truth -------------------------------------------------------------------------------


def test_ground_truth_structure(small_course):
    gt = extract_ground_truth("P", small_course)
    assert [(u.unit_no, u.unit_name) for u in gt.units] == [
        (0, "getting started"),
        (3, "data structures"),
    ]
    # Same module name with a different type is a different module.
    assert [(m.module_type.value, m.module_name) for m in gt.units[0].modules] == [
        ("CONCEPT", "Intro"),
        ("PROJECT", "Intro"),
    ]
    assert gt.units[0].modules[0].lo_ids == ["p-u00-concept-intro-lo01", "p-u00-concept-intro-lo02"]
    assert gt.broad_lo_ids == {SyllabusLevel.COURSE_GOAL: ["p-syllabus-course-goal-lo01"]}
    assert sorted(gt.all_lo_ids()) == sorted(lo.raw_id for lo in small_course)


def test_ground_truth_rejects_non_contiguous_module(small_course):
    shuffled = [small_course[0], small_course[2], small_course[1]]
    with pytest.raises(IngestError, match="not contiguous"):
        extract_ground_truth("P", shuffled)


def test_ground_truth_rejects_other_course(small_course):
    with pytest.raises(IngestError, match="belongs to"):
        extract_ground_truth("Q", small_course)


# --- Agent 1 input ------------------------------------------------------------------------------


def test_strip_lo_marker():
    assert strip_lo_marker("Add an element to a list. (LO3)") == "Add an element to a list."
    assert strip_lo_marker("Add an element (LO 12)") == "Add an element"
    assert strip_lo_marker("Explain (LO3) in the middle.") == "Explain (LO3) in the middle."


def test_input_contains_only_ids_and_texts(small_course):
    prep = make_agent1_input("P", seed=1, los=small_course)
    dumped = json.loads(prep.payload.model_dump_json())
    assert set(dumped) == {"course", "los"}
    assert all(set(item) == {"id", "text"} for item in dumped["los"])
    # Every LO is present once, broad LOs mixed in.
    assert sorted(prep.id_map.values()) == sorted(lo.raw_id for lo in small_course)
    assert len(dumped["los"]) == len(small_course)


def assert_no_leak(payload: Agent1Input, los: list[RawLO]) -> None:
    blob = payload.model_dump_json()
    for lo in los:
        assert lo.raw_id not in blob
        assert re.fullmatch(r"LO-[0-9a-f]{6}", input_id(lo.raw_id))
    for field in ("unit_no", "unit_name", "module_type", "module_name", "lo_no", "level", "kind"):
        assert f'"{field}"' not in blob
    for item in payload.los:
        assert not re.search(r"\(LO ?\d+\)\s*$", item.text)


def test_no_ground_truth_leak(small_course):
    prep = make_agent1_input("P", seed=1, los=small_course)
    assert_no_leak(prep.payload, small_course)
    [edit] = prep.edits
    assert edit.raw_id == "p-u00-concept-intro-lo02"
    assert edit.after == "Define a variable."


def test_seeded_shuffle_is_deterministic(small_course):
    a = make_agent1_input("P", seed=7, los=small_course).payload
    b = make_agent1_input("P", seed=7, los=list(reversed(small_course))).payload
    assert a == b  # independent of input order
    orders = {
        tuple(x.id for x in make_agent1_input("P", s, los=small_course).payload.los)
        for s in range(10)
    }
    assert len(orders) > 1


def test_only_requested_course(small_course):
    other = csv_lo("Q", 0, "CONCEPT", "Intro", 1, "Other course LO.")
    prep = make_agent1_input("P", seed=0, los=small_course + [other])
    assert other.raw_id not in prep.id_map.values()
    with pytest.raises(IngestError):
        make_agent1_input("Z", seed=0, los=small_course)


# --- Real data ----------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_data():
    data = load_all(DATA_RAW)
    if len(data) < 6:
        pytest.skip("course CSVs not present in data/raw/")
    return data


def test_real_ground_truth_covers_every_lo(real_data):
    for course, los in real_data.items():
        gt = extract_ground_truth(course, los)
        assert sorted(gt.all_lo_ids()) == sorted(lo.raw_id for lo in los)


def test_real_agent1_inputs(real_data):
    for course, los in real_data.items():
        prep = make_agent1_input(course, seed=0, los=los)
        assert len(prep.payload.los) == len(los)
        assert_no_leak(prep.payload, los)
        broad = sum(isinstance(lo.source, SyllabusSource) for lo in los)
        assert broad == (22 if course == "PPP" else 0)
        assert len(prep.edits) == (49 if course == "PPP" else 0)
        # Shuffled: not in file order.
        assert [prep.id_map[x.id] for x in prep.payload.los] != [lo.raw_id for lo in los]
