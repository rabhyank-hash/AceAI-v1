import json
from collections import Counter

from aceai.agents.sequencer import run_checks
from aceai.agents.sequencer_v2 import (
    Ask,
    ModuleVotes,
    aggregate,
    ask_messages,
    break_cycles,
    consensus_clusters,
    majority_edges,
    order_modules,
    order_within,
    parse_ask,
    sequence_v2,
)
from aceai.config import ProviderConfig
from aceai.ingest.agent1_input import Agent1Input, InputLO
from aceai.llm.client import LLMClient
from aceai.schemas import SequencerOutput
from fakes import FakeSDK

PROVIDER = ProviderConfig("fake", "http://fake/v1", "FAKE_KEY", "m", {})


def run_output(groups: list[list[str]], deps: dict[str, list[str]] | None = None, track=None):
    """A minimal v1-style output with the given module membership."""
    deps = deps or {}
    los, modules = [], []
    for i, g in enumerate(groups, 1):
        modules.append({"id": f"M{i}", "title": "t", "order": i, "lo_ids": g})
        for lid in g:
            los.append(
                {
                    "id": lid,
                    "raw_text": lid,
                    "source_ids": [lid],
                    "verb": "explain",
                    "bloom_level": "C2",
                    "track": (track or {}).get(lid, "conceptual"),
                    "target_concept": "x",
                    "scope": "atomic",
                    "depends_on": deps.get(lid, []),
                }
            )
    ids = [lid for g in groups for lid in g]
    return SequencerOutput.model_validate(
        {"modules": modules, "los": los, "provenance": [{"raw_id": i, "lo_id": i} for i in ids]}
    )


def assignments(groups):
    return {lid: i for i, g in enumerate(groups) for lid in g}


def test_consensus_keeps_majority_pairs_together():
    runs = [
        assignments([["a", "b", "c"], ["d", "e"]]),
        assignments([["a", "b"], ["c", "d", "e"]]),
        assignments([["a", "b", "c"], ["d", "e"]]),
    ]
    # c is with a,b in 2 of 3 runs, with d,e in 1 of 3.
    assert consensus_clusters(list("abcde"), runs) == [["a", "b", "c"], ["d", "e"]]


def test_consensus_is_order_independent_and_splits_weak_pairs():
    runs = [assignments([["a", "b"], ["c"]]), assignments([["a"], ["b", "c"]])]
    first = consensus_clusters(list("abc"), runs)
    assert first == consensus_clusters(list("cab"), list(reversed(runs)))
    assert first == [["a"], ["b"], ["c"]]  # every pair together in only half the runs


def test_parse_ask_maps_labels_and_drops_target_and_unknown():
    labels = {"M1": "G02", "M2": "G01", "M3": "G03"}
    reply = {"required_before": ["M2", "M3", "M9"], "better_before": ["M2", "M1"], "reason": "r"}
    required, better, reason = parse_ask(reply, labels, target="G03")
    assert required == ["G01"] and better == ["G02"] and reason == "r"


def test_ask_messages_shuffle_by_seed_only():
    modules = {"G01": ["a", "b"], "G02": ["c"], "G03": ["d"]}
    texts = {x: f"text {x}" for x in "abcd"}
    m0, l0 = ask_messages(modules, texts, "G01", 0)
    assert (m0, l0) == ask_messages(modules, texts, "G01", 0)
    variants = {tuple(ask_messages(modules, texts, "G01", s)[1].values()) for s in range(6)}
    assert len(variants) > 1
    assert sorted(l0.values()) == ["G01", "G02", "G03"]


def test_majority_edges_need_strict_majority_of_target_asks():
    votes = aggregate(
        [
            Ask(0, "B", {}, required=["A"]),
            Ask(1, "B", {}, required=["A"]),
            Ask(2, "B", {}, required=[]),
            Ask(0, "C", {}, required=["A"]),
            Ask(1, "C", {}, required=[]),
            Ask(2, "C", {}, required=[], error="failed"),
        ]
    )
    # A->B: 2 of 3; A->C: 1 of 2 answered, not a strict majority.
    assert majority_edges(votes) == {("A", "B"): 2 / 3}


def test_break_cycles_removes_weakest_edge():
    edges = {("A", "B"): 1.0, ("B", "A"): 2 / 3, ("B", "C"): 1.0}
    kept, removed = break_cycles(["A", "B", "C"], edges)
    assert removed == [("B", "A")] and set(kept) == {("A", "B"), ("B", "C")}


def test_order_ties_by_preference_then_conceptual_share_then_id():
    votes = ModuleVotes(Counter(), Counter({("C", "B"): 2}), Counter())
    order, ties = order_modules(["A", "B", "C"], {}, votes, {"A": 0.0, "B": 1.0, "C": 0.0})
    # Copeland among {A, B, C}: C +1 (beats B), A 0, B -1. C goes first; then A and B tie on
    # preference and B has the larger conceptual share.
    assert order[0] == "C" and order[1:] == ["B", "A"]
    assert ties[0]["chosen"] == "C"


def test_order_respects_edges():
    votes = ModuleVotes(Counter(), Counter({("B", "A"): 3}), Counter())
    order, _ = order_modules(["A", "B"], {("A", "B"): 1.0}, votes, {})
    assert order == ["A", "B"]  # the edge wins over the preference


def test_order_within_topological_then_conceptual_first():
    track = {"x": "applied", "y": "conceptual", "z": "applied"}
    assert order_within(["x", "y", "z"], {("z", "x")}, track) == ["y", "x", "z"]


def test_sequence_v2_end_to_end_is_valid_and_deterministic():
    ids = ["a", "b", "c", "d"]
    payload = Agent1Input(course="T", los=[InputLO(id=i, text=f"text {i}") for i in ids])
    runs = [
        run_output([["a", "b"], ["c", "d"]], deps={"b": ["a"], "c": ["a"]}),
        run_output([["a", "b"], ["c", "d"]], deps={"b": ["a"]}),
        run_output([["a", "b", "c"], ["d"]], deps={"b": ["a"]}),
    ]

    # Every ask: the other module is required before the target only when target is G02.
    def replies():
        out = []
        for _seed in (0, 1, 2):
            out.append(json.dumps({"required_before": [], "better_before": []}))
            out.append(json.dumps({"required_before": ["M1", "M2"], "better_before": []}))
        return out

    results = []
    for _ in range(2):
        sdk = FakeSDK(replies())
        client = LLMClient(PROVIDER, cache_dir=None, sdk=sdk)
        results.append(sequence_v2(client, payload, runs, [0, 1, 2]))
    r = results[0]
    assert r.clusters == [["a", "b"], ["c", "d"]]
    assert [m.id for m in r.output.modules] == ["G01", "G02"]
    assert r.output.modules[0].lo_ids == ["a", "b"]  # b depends on a in all runs
    checks = run_checks(json.loads(r.output.model_dump_json()), payload)
    assert not [e for res in checks.values() if not isinstance(res, str) for e in res.errors]
    assert results[0].output == results[1].output
