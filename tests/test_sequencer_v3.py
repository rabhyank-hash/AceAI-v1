import json

from aceai.agents.sequencer_v3 import (
    Ask,
    SplitAsk,
    ask_messages,
    ask_once,
    check_output,
    checks_ok,
    consensus_order,
    exact_duplicates,
    parse_answer,
    parse_split,
    sequence_v3,
    split_consensus,
    split_messages,
    split_once,
)
from aceai.config import ProviderConfig
from aceai.ingest.agent1_input import Agent1Input, InputLO
from aceai.llm.client import LLMClient
from fakes import FakeSDK, FunctionSDK

PROVIDER = ProviderConfig("fake", "http://fake/v1", "FAKE_KEY", "m", {})


def los(*texts):
    return [InputLO(id=f"LO-{i}", text=t) for i, t in enumerate(texts)]


def client(sdk):
    return LLMClient(PROVIDER, cache_dir=None, sdk=sdk)


# --- 1. exact duplicates ------------------------------------------------------------------------


def test_exact_duplicates_only_identical_text():
    items = los("Define X.", "Define  X. ", "define X.", "Use X.")
    survivors, merged = exact_duplicates(items)
    assert [s.id for s in survivors] == ["LO-0", "LO-2", "LO-3"]
    assert merged == {"LO-0": ["LO-1"]}  # whitespace differs; case-sensitive: LO-2 kept


# --- 2. order asks ------------------------------------------------------------------------------


def test_ask_messages_relabel_by_seed():
    items = los("a", "b", "c", "d")
    _, l0 = ask_messages(items, 0)
    assert ask_messages(items, 0)[1] == l0
    assert sorted(l0.values()) == [x.id for x in items]
    assert len({tuple(ask_messages(items, s)[1].values()) for s in range(6)}) > 1


def test_parse_answer_reports_missing_repeated_unknown():
    labels = {"L1": "a", "L2": "b", "L3": "c"}
    reply = {"modules": [{"title": "t", "los": ["L1", "L2"]}, {"title": "u", "los": ["L2", "L9"]}]}
    assert parse_answer(reply, labels)[2] == "missing L3; repeated L2; unknown labels L9"
    ok = {"modules": [{"title": "t", "los": ["L3", "L1"]}, {"title": "u", "los": ["L2"]}]}
    order, titles, problem = parse_answer(ok, labels)
    assert problem is None and order == ["c", "a", "b"] and titles == ["t", "u"]


def test_ask_repairs_once_then_accepts():
    items = los("a", "b")
    _, labels = ask_messages(items, 0)
    first = {"modules": [{"title": "t", "los": ["L1"]}]}
    second = {"modules": [{"title": "t", "los": ["L2", "L1"]}]}
    sdk = FakeSDK([json.dumps(first), json.dumps(second)])
    ask = ask_once(client(sdk), items, 0, 100)
    assert ask.valid and ask.repaired and ask.order == [labels["L2"], labels["L1"]]
    assert "missing L2" in sdk.calls[1]["messages"][-1]["content"]


def test_ask_excluded_when_repair_fails():
    items = los("a", "b")
    bad = json.dumps({"modules": [{"title": "t", "los": ["L1"]}]})
    ask = ask_once(client(FakeSDK([bad, bad])), items, 0, 100)
    assert not ask.valid and "missing L2" in ask.error


# --- 3. consensus order -------------------------------------------------------------------------


def test_consensus_order_by_mean_position():
    labels = {"L1": "a", "L2": "b", "L3": "c", "L4": "d"}
    asks = [
        Ask(0, labels, order=["a", "b", "c", "d"]),
        Ask(1, labels, order=["b", "a", "c", "d"]),
        Ask(2, labels, order=["a", "c", "b", "d"]),
        Ask(3, labels, error="bad"),
    ]
    order, mean_pos = consensus_order(asks)
    assert order == ["a", "b", "c", "d"]
    assert mean_pos["a"] == 1 / 9 and mean_pos["d"] == 1.0


# --- 4-5. split asks and modules ----------------------------------------------------------------


def codes_in_order(labels):
    return [c for c, _ in sorted(labels.items(), key=lambda kv: kv[1])]


def test_split_messages_keep_order_and_shuffle_codes():
    order, texts = ["a", "b", "c", "d"], {x: f"text {x}" for x in "abcd"}
    msgs, labels = split_messages(order, texts, 0)
    lines = msgs[1]["content"].splitlines()[1:]
    assert [line.split(": ")[1] for line in lines] == ["text a", "text b", "text c", "text d"]
    assert sorted(labels.values()) == [0, 1, 2, 3]
    assert len({tuple(codes_in_order(split_messages(order, texts, s)[1])) for s in range(6)}) > 1


def test_parse_split_valid_and_invalid():
    labels = {"K3": 0, "K1": 1, "K4": 2, "K2": 3}
    ok = {"modules": [{"los": ["K3", "K1"]}, {"los": ["K4"]}, {"los": ["K2"]}]}
    assert parse_split(ok, labels) == ([1, 2], 3, None)
    swapped = {"modules": [{"los": ["K1", "K3"]}, {"los": ["K4", "K2"]}]}
    assert parse_split(swapped, labels)[2] == "the order was changed"
    short = {"modules": [{"los": ["K3", "K1", "K4"]}]}
    assert parse_split(short, labels)[2] == "missing K2"


def test_split_once_repairs_then_accepts():
    order, texts = ["a", "b", "c"], {x: x for x in "abc"}
    _, labels = split_messages(order, texts, 0)
    codes = codes_in_order(labels)
    bad = json.dumps({"modules": [{"los": [codes[1], codes[0], codes[2]]}]})
    good = json.dumps({"modules": [{"los": codes[:2]}, {"los": codes[2:]}]})
    ask = split_once(client(FakeSDK([bad, good])), order, texts, 0, 100)
    assert ask.valid and ask.repaired and ask.cuts == [1] and ask.n_modules == 2


def test_split_consensus_majority_cuts():
    order = ["a", "b", "c", "d", "e"]
    asks = [
        SplitAsk(0, {}, cuts=[1, 3], n_modules=3),
        SplitAsk(1, {}, cuts=[1], n_modules=2),
        SplitAsk(2, {}, cuts=[1, 2], n_modules=3),
        SplitAsk(3, {}, error="bad"),
    ]
    modules, votes = split_consensus(asks, order)
    assert modules == [["a", "b"], ["c", "d", "e"]]  # cut after b: 3/3; after c or d: 1/3
    assert votes == [0.0, 1.0, 1 / 3, 1 / 3]


# --- pipeline -----------------------------------------------------------------------------------


def topic_answer(messages):
    """A fake model: A's LOs before B's, one module per topic, for both kinds of ask."""
    rows = [line.split(": ", 1) for line in messages[1]["content"].splitlines()[1:]]
    a = [code for code, text in rows if "A." in text]
    b = [code for code, text in rows if "B." in text]
    return json.dumps({"modules": [{"title": "A", "los": a}, {"title": "B", "los": b}]})


def test_sequence_v3_end_to_end_valid_and_deterministic():
    items = los("Define A.", "Use A.", "Define B.", "Use B.", "Define A.")
    payload = Agent1Input(course="T", los=items)
    results = [
        sequence_v3(client(FunctionSDK(topic_answer)), payload, [0, 1, 2, 3, 4]) for _ in range(2)
    ]
    r = results[0]
    assert r.stopped == "completed" and len(r.asks) == 5 and len(r.split_asks) == 5
    assert r.merged == {"LO-0": ["LO-4"]}
    assert [len(m.lo_ids) for m in r.output.modules] == [2, 2]
    assert r.output.provenance_map()["LO-4"] == ["LO-0"]
    assert checks_ok(check_output(json.loads(r.output.model_dump_json()), payload))
    assert results[0].output == results[1].output


def test_sequence_v3_stops_without_enough_valid_asks():
    items = los("Define A.", "Use A.", "Define B.")
    bad = json.dumps({"modules": [{"title": "t", "los": ["L1"]}]})
    r = sequence_v3(
        client(FakeSDK([bad] * 10)), Agent1Input(course="T", los=items), [0, 1, 2, 3, 4]
    )
    assert r.output is None and r.stopped == "0 valid order asks" and r.split_asks == []
