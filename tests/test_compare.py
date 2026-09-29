import pytest

from aceai.agents.sequencer_v3 import build_output
from aceai.eval.compare import adjusted_rand_index, bcubed, compare, sequence_agreement
from aceai.ingest.ground_truth import GTLO, GroundTruth, GTModule, GTUnit
from aceai.schemas import ModuleType

# A small course: unit 0 has modules X (a, b) and Y (c); unit 1 has module Z (d, e).
GT = GroundTruth(
    course="T",
    units=[
        GTUnit(
            unit_no=0,
            unit_name="u0",
            modules=[
                GTModule(
                    module_type=ModuleType.CONCEPT,
                    module_name="X",
                    los=[GTLO(id="ra", text="a"), GTLO(id="rb", text="b")],
                ),
                GTModule(
                    module_type=ModuleType.PROJECT, module_name="Y", los=[GTLO(id="rc", text="c")]
                ),
            ],
        ),
        GTUnit(
            unit_no=1,
            unit_name="u1",
            modules=[
                GTModule(
                    module_type=ModuleType.CONCEPT,
                    module_name="Z",
                    los=[GTLO(id="rd", text="d"), GTLO(id="re", text="e")],
                ),
            ],
        ),
    ],
    broad_los={},
)
ID_MAP = {"a": "ra", "b": "rb", "c": "rc", "d": "rd", "e": "re"}
TEXTS = {k: k for k in ID_MAP}


def out(modules, merged=None):
    return build_output(modules, TEXTS, merged or {})


def test_identical_structure_scores_one():
    c = compare(out([["a", "b"], ["c"], ["d", "e"]]), ID_MAP, GT)
    assert c["coverage"] == {"placed": 5, "total": 5, "complete": True}
    g = c["grouping"]["module"]
    assert g["f1"] == g["ari"] == g["bcubed"]["f1"] == 1.0
    assert c["order"]["module"]["agreement"] == c["sequence"]["agreement"] == 1.0


def test_merging_a_units_modules_is_right_at_unit_level():
    c = compare(out([["a", "b", "c"], ["d", "e"]]), ID_MAP, GT)
    assert c["grouping"]["unit"]["f1"] == 1.0
    assert c["grouping"]["module"]["precision"] < 1.0 and c["grouping"]["module"]["recall"] == 1.0


def test_reversed_order_and_sequence():
    c = compare(out([["d", "e"], ["c"], ["a", "b"]]), ID_MAP, GT)
    assert c["order"]["module"]["agreement"] == 0.0
    assert c["sequence"]["agreement"] < 0.5


def test_unplaced_los_lower_recall_and_break_coverage():
    c = compare(out([["a", "b"], ["d", "e"]]), ID_MAP, GT)  # c missing
    assert c["coverage"] == {"placed": 4, "total": 5, "complete": False}
    assert c["unplaced"][0]["raw_id"] == "rc"


def test_exact_duplicate_merge_counts_as_placed():
    c = compare(out([["a", "c"], ["d", "e"]], merged={"a": ["b"]}), ID_MAP, GT)
    assert c["coverage"]["complete"]
    assert c["merges"][0]["raw_ids"] == ["ra", "rb"]


@pytest.mark.parametrize(
    "a,b,expected",
    [
        ([0, 0, 1, 1], [5, 5, 7, 7], 1.0),  # same partition, different names
        ([0, 0, 1, 1], [0, 1, 0, 1], -0.5),  # maximally crossed
        ([0, 0, 0, 1, 1, 1], [0, 0, 1, 1, 2, 2], 0.242),
    ],
)
def test_adjusted_rand_index(a, b, expected):
    assert adjusted_rand_index(a, b) == expected


def test_bcubed():
    assert bcubed([0, 0, 1, 1], [5, 5, 7, 7]) == {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    assert bcubed([0, 0, 1, 1], [0, 0, 0, 0])["precision"] == 0.5  # all in one module
    assert bcubed([0, 0, 1, 1], [0, 1, 2, 3])["recall"] == 0.5  # all singletons


def test_sequence_agreement():
    assert sequence_agreement({"a": 0, "b": 1, "c": 2}, {"a": 0, "b": 1, "c": 2}) == 1.0
    assert sequence_agreement({"a": 0, "b": 1, "c": 2}, {"a": 2, "b": 1, "c": 0}) == 0.0
    assert sequence_agreement({"a": 0, "b": 0}, {"a": 0, "b": 1}) is None  # ties skipped
