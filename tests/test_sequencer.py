import json

import pytest

from aceai.agents.sequencer import apply_patch, assemble, sequence
from aceai.config import DATA_RAW, ProviderConfig
from aceai.eval.compare import adjusted_rand_index, bcubed, compare
from aceai.eval.draw import graph_markdown, layers
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
            mid = f"M{len(modules) + 1}"
            modules.append({"id": mid, "title": m.module_name})
            for rid in m.lo_ids:
                los.append(
                    {
                        "id": to_input[rid],
                        "module": mid,
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
    assert cmp["coverage"] == {"placed": 49, "total": 49, "complete": True}
    assert not cmp["merges"]


def test_model_never_sees_raw_ids(sample):
    prepared, gt = sample
    client, sdk = client_for(perfect_reply(prepared, gt))
    sequence(client, prepared.payload)
    sent = json.dumps(sdk.calls[0]["messages"])
    assert not any(raw in sent for raw in prepared.id_map.values())


def test_errors_are_sent_back_and_repaired_by_patch(sample):
    prepared, gt = sample
    good = perfect_reply(prepared, gt)
    bad = json.loads(json.dumps(good))
    dropped = bad["los"].pop()
    first, second = bad["los"][0], bad["los"][1]
    first["depends_on"], second["depends_on"] = [second["id"]], [first["id"]]
    patch = {
        "los": [
            {"id": first["id"], "depends_on": None},  # partial update: remove the field
            {"id": second["id"], "depends_on": None},
            dropped,
        ]
    }

    client, sdk = client_for(bad, patch)
    run = sequence(client, prepared.payload, max_repairs=2)
    assert run.ok and len(run.attempts) == 2
    tools_with_errors = {k for k, v in run.attempts[0].checks.items() if v.errors}
    assert {"check_provenance", "check_cycles"} <= tools_with_errors
    repair = sdk.calls[1]["messages"][-1]["content"]
    assert dropped["id"] in repair and "missing" in repair and "cycle" in repair.lower()
    assert "ONLY what changes" in repair
    # The model sees a compact current plan, not the transcript.
    assert len(sdk.calls[1]["messages"]) == 3
    assert f"{first['id']} | {first['module']} | C2 | conceptual | atomic | - | {second['id']}" in (
        repair
    )
    # The patched plan equals the good one (dropped LO appended at the end, as in `good`).
    assert run.attempts[1].reply == good


def test_repair_requests_do_not_grow(sample):
    prepared, gt = sample
    bad = perfect_reply(prepared, gt)
    bad["los"].pop()
    client, sdk = client_for(bad, {"los": []}, {"los": []})
    run = sequence(client, prepared.payload, max_repairs=2)
    assert not run.ok and len(sdk.calls) == 3
    assert len(sdk.calls[1]["messages"]) == len(sdk.calls[2]["messages"]) == 3


def test_apply_patch_replaces_removes_and_appends():
    current = {
        "los": [{"id": "a", "v": 1}, {"id": "b", "v": 1}, {"id": "c", "v": 1}],
        "modules": [{"id": "M1"}],
    }
    patch = {
        "los": [{"id": "a", "v": None, "w": 3}, {"id": "b", "v": 2}, {"id": "d", "v": 2}],
        "remove_ids": ["c"],
    }
    out = apply_patch(current, patch)
    assert out == {
        "los": [{"id": "a", "w": 3}, {"id": "b", "v": 2}, {"id": "d", "v": 2}],
        "modules": [{"id": "M1"}],
    }
    assert current["los"][1] == {"id": "b", "v": 1}  # input not mutated


def test_apply_patch_merges_modules_by_id():
    """A patch listing one changed module must not wipe the others (seen on CloudNative)."""
    current = {
        "los": [],
        "modules": [{"id": "M1", "title": "a"}, {"id": "M2", "title": "b"}, {"id": "M3"}],
    }
    out = apply_patch(current, {"modules": [{"id": "M2", "title": "B"}]})
    assert out["modules"] == [{"id": "M1", "title": "a"}, {"id": "M2", "title": "B"}, {"id": "M3"}]
    out = apply_patch(
        current,
        {
            "modules": [{"id": "M4", "title": "d"}],
            "remove_module_ids": ["M1"],
            "module_order": ["M4", "M3"],
        },
    )
    assert [m["id"] for m in out["modules"]] == ["M4", "M3", "M2"]


def test_output_falls_back_to_last_usable_attempt(sample):
    prepared, gt = sample
    bad = perfect_reply(prepared, gt)
    bad["los"][0]["depends_on"] = [bad["los"][1]["id"]]
    bad["los"][1]["depends_on"] = [bad["los"][0]["id"]]  # cycle: parses, fails checks
    client, _ = client_for(bad, "not json")
    run = sequence(client, prepared.payload, max_repairs=1)
    assert not run.ok and run.output is not None and run.output_attempt == 0


def test_unplaced_los_make_coverage_incomplete_and_lower_recall(sample):
    prepared, gt = sample
    reply = perfect_reply(prepared, gt)
    for lo in reply["los"][:5]:
        del lo["module"]
    client, _ = client_for(reply)
    run = sequence(client, prepared.payload, max_repairs=0)
    cmp = compare(run.output, prepared.id_map, gt)
    assert cmp["coverage"] == {"placed": 44, "total": 49, "complete": False}
    assert cmp["grouping"]["module"]["recall"] < 1.0
    assert cmp["grouping"]["module"]["precision"] == 1.0
    assert cmp["order"]["module"]["skipped_unplaced_pairs"] > 0
    codes = {e.code for e in run.attempts[0].checks["validate_output"].errors}
    assert "atomic_lo_unplaced" in codes


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
    assert "complete JSON" in sdk.calls[1]["messages"][-1]["content"]


def test_assemble_merges_become_provenance(sample):
    prepared, _ = sample
    a, b = prepared.payload.los[0], prepared.payload.los[1]
    reply = {
        "los": [
            {
                "id": a.id,
                "module": "M1",
                "merged_ids": [b.id],
                "canonical_text": "both",
                "verb": "use",
                "bloom_level": "C3",
                "track": "applied",
                "target_concept": "x",
                "scope": "atomic",
            }
        ],
        "modules": [{"id": "M1", "title": "t"}],
    }
    data = assemble(reply, prepared.payload).data
    assert data["los"][0]["source_ids"] == [a.id, b.id]
    assert data["los"][0]["raw_text"] == a.text
    assert "module" not in data["los"][0]
    assert data["provenance"] == [
        {"raw_id": a.id, "lo_id": a.id},
        {"raw_id": b.id, "lo_id": a.id},
    ]
    assert data["modules"][0]["order"] == 1
    assert data["modules"][0]["lo_ids"] == [a.id]


def test_assemble_builds_module_lists_in_lo_order(sample):
    prepared, _ = sample
    ids = [lo.id for lo in prepared.payload.los[:4]]
    reply = {
        "los": [
            {"id": ids[0], "module": "M2"},
            {"id": ids[1], "module": "M1"},
            {"id": ids[2], "module": "M2"},
            {"id": ids[3], "module": "M9"},
        ],
        "modules": [{"id": "M1", "title": "a", "lo_ids": ["ignored"]}, {"id": "M2", "title": "b"}],
    }
    got = assemble(reply, prepared.payload)
    assert [m["lo_ids"] for m in got.data["modules"]] == [[ids[1]], [ids[0], ids[2]]]
    assert any("unknown module 'M9'" in n for n in got.notes)
    assert any("lo_ids ignored" in n for n in got.notes)


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
    unit_of_module, idx = {}, 0
    for u in gt.units:
        for _ in u.modules:
            idx += 1
            unit_of_module[f"M{idx}"] = f"U{u.unit_no}"
    for lo in reply["los"]:
        lo["module"] = unit_of_module[lo["module"]]
    reply["modules"] = [{"id": f"U{u.unit_no}", "title": u.unit_name} for u in gt.units]
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


def test_bcubed():
    assert bcubed([0, 0, 1, 1], [5, 5, 7, 7]) == {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    # Everything in one module: recall 1, precision = mean share of own group in the module.
    assert bcubed([0, 0, 1, 1], [0, 0, 0, 0])["precision"] == 0.5
    # All singletons: precision 1, recall = 1 / group size.
    assert bcubed([0, 0, 1, 1], [0, 1, 2, 3])["recall"] == 0.5


def test_graph_layers_and_mermaid(sample):
    prepared, gt = sample
    reply = perfect_reply(prepared, gt)
    m1_lo = next(lo for lo in reply["los"] if lo["module"] == "M1")
    m2_lo = next(lo for lo in reply["los"] if lo["module"] == "M2")
    m2_lo["depends_on"] = [m1_lo["id"]]
    reply["modules"][1]["depends_on"] = ["M1"]
    client, _ = client_for(reply)
    run = sequence(client, prepared.payload)
    steps, acyclic = layers(run.output)
    assert acyclic and steps[0][0] == "M1" and "M2" in steps[1]
    md = graph_markdown(run.output, prepared.id_map, gt)
    assert "```mermaid" in md and 'n_M1 -->|"1 LO link"| n_M2' in md
    assert "classDef u1" in md and "Step 2" in md


def test_prerequisites_set_module_and_lo_order(sample):
    """The model lists M1 first, but an M1 LO needs an M2 LO: code puts M2 first (plan step 6).
    Inside M1, an LO listed first but depending on the second is moved after it."""
    prepared, _ = sample
    a, b, c = (lo.id for lo in prepared.payload.los[:3])
    base = {
        "verb": "explain",
        "bloom_level": "C2",
        "track": "conceptual",
        "target_concept": "x",
        "scope": "atomic",
    }
    reply = {
        "los": [
            {**base, "id": a, "module": "M1", "depends_on": [b]},
            {**base, "id": b, "module": "M1", "depends_on": [c]},
            {**base, "id": c, "module": "M2"},
        ],
        "modules": [
            {"id": "M1", "title": "one", "depends_on": ["M9"]},
            {"id": "M2", "title": "two"},
        ],
        "order_rationale": "why",
    }
    got = assemble(reply, prepared.payload)
    mods = got.data["modules"]
    assert [m["id"] for m in mods] == ["M2", "M1"] and [m["order"] for m in mods] == [1, 2]
    assert mods[1]["depends_on"] == ["M2"]  # derived; the model's "M9" is ignored
    assert mods[1]["lo_ids"] == [b, a]
    assert got.order_rationale == "why"
    assert any("model-supplied depends_on ignored" in n for n in got.notes)
    assert any("module order set by prerequisites" in n for n in got.notes)


def test_model_order_breaks_ties(sample):
    prepared, _ = sample
    a, b = (lo.id for lo in prepared.payload.los[:2])
    base = {
        "verb": "explain",
        "bloom_level": "C2",
        "track": "conceptual",
        "target_concept": "x",
        "scope": "atomic",
    }
    reply = {
        "los": [{**base, "id": a, "module": "M2"}, {**base, "id": b, "module": "M1"}],
        "modules": [{"id": "M2", "title": "two"}, {"id": "M1", "title": "one"}],
    }
    got = assemble(reply, prepared.payload)
    assert [m["id"] for m in got.data["modules"]] == ["M2", "M1"]
    assert not any("order set" in n for n in got.notes)


def test_self_reference_in_merged_ids_is_dropped(sample):
    prepared, _ = sample
    a, b = (lo.id for lo in prepared.payload.los[:2])
    base = {
        "verb": "explain",
        "bloom_level": "C2",
        "track": "conceptual",
        "target_concept": "x",
        "scope": "atomic",
        "module": "M1",
    }
    reply = {
        "los": [{**base, "id": a, "merged_ids": [a, b, b]}],
        "modules": [{"id": "M1", "title": "t"}],
    }
    got = assemble(reply, prepared.payload)
    assert got.data["los"][0]["source_ids"] == [a, b]
    assert any("self-references" in n for n in got.notes)
