import json

import pytest

from aceai.agents.sequencer import assemble, sequence
from aceai.config import DATA_RAW, ProviderConfig
from aceai.eval.compare import adjusted_rand_index, compare
from aceai.ingest import load_all
from aceai.ingest.agent1_input import make_agent1_input
from aceai.ingest.ground_truth import extract_ground_truth
from aceai.llm.client import LLMClient
from aceai.schemas import CsvSource
from fakes import FakeSDK, json_validate_error

PROVIDER = ProviderConfig("fake", "http://fake/v1", "FAKE_KEY", "m", {})


@pytest.fixture(scope="module")
def sample():
    """PPP units 1-2: detailed LOs only."""
    if not any(DATA_RAW.glob("PPP_learning_objectives_*.csv")):
        pytest.skip("PPP CSV not present in data/raw/")
    los = [
        lo
        for lo in load_all(DATA_RAW)["PPP"]
        if isinstance(lo.source, CsvSource) and lo.source.unit_no in (1, 2)
    ]
    prepared = make_agent1_input("PPP", seed=0, los=los)
    return prepared, extract_ground_truth("PPP", los)


def perfect_reply(prepared, gt):
    """What a model reproducing the CSV structure exactly would send."""
    to_input = {raw: iid for iid, raw in prepared.id_map.items()}
    los, modules = [], []
    for u in gt.units:
        for m in u.modules:
            ids = [to_input[r] for r in m.lo_ids]
            modules.append({"id": f"M{len(modules) + 1}", "title": m.module_name, "lo_ids": ids})
            for iid in ids:
                los.append(
                    {
                        "id": iid,
                        "verb": "explain",
                        "bloom_level": "C2",
                        "track": "conceptual",
                        "target_concept": "x",
                        "scope": "atomic",
                    }
                )
    return {"los": los, "modules": modules}


def client_for(*replies):
    script = [r if isinstance(r, str | Exception) else json.dumps(r) for r in replies]
    sdk = FakeSDK(script)
    return LLMClient(PROVIDER, cache_dir=None, sdk=sdk), sdk


def test_perfect_reply_passes_and_scores_one(sample):
    prepared, gt = sample
    client, sdk = client_for(perfect_reply(prepared, gt))
    run = sequence(client, prepared.payload)
    assert run.ok and run.stopped_because == "all checks passed"
    assert len(sdk.calls) == 1
    assert [m.order for m in run.output.modules] == list(range(1, len(run.output.modules) + 1))
    cmp = compare(run.output, prepared.id_map, gt)
    assert cmp["grouping"]["module"]["f1"] == 1.0 and cmp["grouping"]["module"]["ari"] == 1.0
    # CSV modules are finer than units: never mixing units (precision 1), but recall < 1.
    assert cmp["grouping"]["unit"]["precision"] == 1.0
    assert cmp["grouping"]["unit"]["recall"] < 1.0
    assert cmp["order"]["module"]["agreement"] == cmp["order"]["unit"]["agreement"] == 1.0
    assert cmp["n_detailed_placed"] == cmp["n_detailed"] and not cmp["merges"]


def test_model_never_sees_raw_ids(sample):
    prepared, gt = sample
    client, sdk = client_for(perfect_reply(prepared, gt))
    sequence(client, prepared.payload)
    sent = json.dumps(sdk.calls[0]["messages"])
    assert not any(raw in sent for raw in prepared.id_map.values())


def test_errors_are_sent_back_and_repaired(sample):
    prepared, gt = sample
    good = perfect_reply(prepared, gt)
    bad = json.loads(json.dumps(good))
    dropped = bad["los"].pop()["id"]
    bad["modules"][-1]["lo_ids"].remove(dropped)
    first, second = bad["los"][0]["id"], bad["los"][1]["id"]
    bad["los"][0]["depends_on"] = [second]
    bad["los"][1]["depends_on"] = [first]

    client, sdk = client_for(bad, good)
    run = sequence(client, prepared.payload, max_repairs=2)
    assert run.ok and len(run.attempts) == 2
    tools_with_errors = {k for k, v in run.attempts[0].checks.items() if v.errors}
    assert {"check_provenance", "check_cycles"} <= tools_with_errors
    repair = sdk.calls[1]["messages"][-1]["content"]
    assert dropped in repair and "missing" in repair and "cycle" in repair.lower()
    assert sdk.calls[1]["messages"][-2]["role"] == "assistant"


def test_invalid_json_then_gives_up(sample):
    prepared, _ = sample
    client, sdk = client_for("not json", "still not json")
    run = sequence(client, prepared.payload, max_repairs=1)
    assert not run.ok and run.output is None and len(sdk.calls) == 2
    assert "not valid JSON" in sdk.calls[1]["messages"][-1]["content"]
    assert run.stopped_because == "errors remain after 1 repair round(s)"


def test_provider_json_rejection_goes_to_repair(sample):
    prepared, gt = sample
    client, sdk = client_for(json_validate_error("{broken"), perfect_reply(prepared, gt))
    run = sequence(client, prepared.payload, max_repairs=1)
    assert run.ok and len(run.attempts) == 2
    assert run.attempts[0].reply_text == "{broken" and "invalid JSON" in run.attempts[0].error
    assert sdk.calls[1]["messages"][-2] == {"role": "assistant", "content": "{broken"}


def test_assemble_merges_become_provenance(sample):
    prepared, _ = sample
    a, b = prepared.payload.los[0], prepared.payload.los[1]
    reply = {
        "los": [
            {
                "id": a.id,
                "merged_ids": [b.id],
                "canonical_text": "both",
                "verb": "use",
                "bloom_level": "C3",
                "track": "applied",
                "target_concept": "x",
                "scope": "atomic",
            }
        ],
        "modules": [{"id": "M1", "title": "t", "lo_ids": [a.id]}],
    }
    data = assemble(reply, prepared.payload).data
    assert data["los"][0]["source_ids"] == [a.id, b.id]
    assert data["los"][0]["raw_text"] == a.text
    assert data["provenance"] == [
        {"raw_id": a.id, "lo_id": a.id},
        {"raw_id": b.id, "lo_id": a.id},
    ]
    assert data["modules"][0]["order"] == 1


def test_compare_counts_swapped_modules_and_merges(sample):
    prepared, gt = sample
    reply = perfect_reply(prepared, gt)
    reply["modules"][0], reply["modules"][1] = reply["modules"][1], reply["modules"][0]
    client, _ = client_for(reply)
    run = sequence(client, prepared.payload)
    cmp = compare(run.output, prepared.id_map, gt)
    assert cmp["grouping"]["module"]["f1"] == 1.0
    assert cmp["order"]["module"]["agreement"] < 1.0
    # Modules 0 and 1 are both in unit 1, so unit-level order is unaffected.
    assert cmp["order"]["unit"]["agreement"] == 1.0


def test_merging_a_units_modules_is_right_at_unit_level(sample):
    """One module per CSV unit: wrong at module level, perfect at unit level."""
    prepared, gt = sample
    reply = perfect_reply(prepared, gt)
    per_unit, start = [], 0
    for u in gt.units:
        n = len(u.modules)
        ids = [i for m in reply["modules"][start : start + n] for i in m["lo_ids"]]
        per_unit.append({"id": f"U{u.unit_no}", "title": u.unit_name, "lo_ids": ids})
        start += n
    reply["modules"] = per_unit
    client, _ = client_for(reply)
    cmp = compare(sequence(client, prepared.payload).output, prepared.id_map, gt)
    assert cmp["grouping"]["unit"]["f1"] == 1.0 and cmp["grouping"]["unit"]["ari"] == 1.0
    assert cmp["grouping"]["module"]["precision"] < 1.0
    assert cmp["grouping"]["module"]["recall"] == 1.0


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
