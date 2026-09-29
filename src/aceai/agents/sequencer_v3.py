"""Agent 1 v3: order the LOs, then split the order into modules (plan v3, §4).

1. Exact duplicates (code): LOs with identical text after whitespace normalization are merged.
2. Order asks (LLM, k times): every LO, shuffled and relabelled L1..Ln, returned in teaching
   order. An answer that misses or repeats labels gets one repair message; if still wrong, the
   ask is excluded.
3. Consensus order (code): mean position over valid order asks (Borda), ties by id.
4. Split asks (LLM, k times): the consensus order under codes shuffled per ask; the model splits
   it into modules and decides how many. An answer that changes the order, misses or repeats a
   code gets one repair message; if still wrong, the ask is excluded.
5. Modules (code): a boundary where a strict majority of valid split asks place one.

A run needs at least `MIN_VALID_ASKS` valid asks in each stage. Normalization (verb, Bloom level,
track, target concept) is not part of Agent 1; the output schema still requires those fields, so
they hold placeholders (`PLACEHOLDER`) that no score or check uses.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from statistics import mean
from typing import Any

from aceai.ingest.agent1_input import Agent1Input, InputLO
from aceai.llm.client import LLMClient, LLMError
from aceai.schemas import SequencerOutput
from aceai.tools import check_provenance, validate_output

ORDER_PROMPT_VERSION = "v3-2"
SPLIT_PROMPT_VERSION = "v3-seg-1"
MIN_VALID_ASKS = 3

# The order ask also asks for modules: the recorded v3 experiments used this prompt. Only the
# order of its answer is used.
ORDER_PROMPT = """\
You are an instructional designer. You receive every learning objective (LO) of one course, in \
random order, labelled L1 to Ln. Arrange them into a course: put them in the order they should \
be taught, and split that sequence into modules (topics).

Include every LO. Do not drop, merge or skip any, even if it seems redundant, off-topic, or \
about course logistics. Every label from L1 to Ln must appear exactly once.

Reply with JSON only, modules and LOs in teaching order:
{"modules": [{"title": "...", "los": ["L3", "L1", ...]}, ...]}"""

ORDER_REPAIR = """\
Your answer is not a valid arrangement: {problems}. Every label L1 to L{n} must appear exactly \
once. Reply with the complete JSON again."""

SPLIT_PROMPT = """\
You are an instructional designer. You receive the learning objectives (LOs) of one course, \
already in teaching order, each with a random code. Split this sequence into modules: groups of \
consecutive LOs that belong to one topic. Decide how many modules the course needs.

Keep the given order. Include every LO exactly once. Do not move, drop or merge any LO.

Reply with JSON only, modules in order, each listing its LO codes in the given order:
{"modules": [{"title": "...", "los": ["K7", "K2", ...]}, ...]}"""

SPLIT_REPAIR = """\
Your answer is not a valid split of the given sequence: {problems}. List every code exactly \
once, in the given order, grouped into consecutive modules. Reply with the complete JSON again."""

PLACEHOLDER = {
    "verb": "(not normalized)",
    "bloom_level": "C1",
    "track": "conceptual",
    "target_concept": "(not normalized)",
    "scope": "atomic",
}


# --- 1. exact duplicates ------------------------------------------------------------------------


def exact_duplicates(los: list[InputLO]) -> tuple[list[InputLO], dict[str, list[str]]]:
    """Merge LOs whose text is identical after whitespace normalization (case-sensitive).
    Returns the surviving LOs (smallest id per text, sorted by id) and survivor -> merged ids."""
    by_text: dict[str, list[InputLO]] = {}
    for lo in los:
        by_text.setdefault(" ".join(lo.text.split()), []).append(lo)
    survivors, merged = [], {}
    for group in by_text.values():
        group = sorted(group, key=lambda x: x.id)
        survivors.append(group[0])
        if len(group) > 1:
            merged[group[0].id] = [x.id for x in group[1:]]
    return sorted(survivors, key=lambda x: x.id), merged


# --- 2. order asks ------------------------------------------------------------------------------


@dataclass
class Ask:
    seed: int
    labels: dict[str, str]  # label -> LO id
    order: list[str] = field(default_factory=list)  # LO ids in teaching order
    titles: list[str] = field(default_factory=list)  # model free text, kept for the run record
    repaired: bool = False
    error: str | None = None

    @property
    def valid(self) -> bool:
        return self.error is None and bool(self.order)


def ask_messages(los: list[InputLO], seed: int) -> tuple[list[dict[str, str]], dict[str, str]]:
    order = sorted(los, key=lambda x: x.id)
    random.Random(f"v3-{seed}").shuffle(order)
    labels = {f"L{i}": lo.id for i, lo in enumerate(order, 1)}
    lines = [f"Course with {len(order)} learning objectives:"]
    lines += [f"{label}: {lo.text}" for label, lo in zip(labels, order, strict=True)]
    return [
        {"role": "system", "content": ORDER_PROMPT},
        {"role": "user", "content": "\n".join(lines)},
    ], labels


def parse_answer(reply: Any, labels: dict[str, str]) -> tuple[list[str], list[str], str | None]:
    """(LO ids in teaching order, module titles, problem description or None)."""
    mods = reply.get("modules") if isinstance(reply, dict) else None
    if not isinstance(mods, list) or not mods:
        return [], [], "no modules list"
    seen: list[str] = []
    unknown: list[str] = []
    order, titles = [], []
    for m in mods:
        items = m.get("los") if isinstance(m, dict) else None
        if not isinstance(items, list):
            return [], [], "a module has no los list"
        for item in items:
            label = str(item).strip()
            if label in labels:
                seen.append(label)
                order.append(labels[label])
            else:
                unknown.append(label)
        titles.append(str(m.get("title", "")))
    missing = [lab for lab in labels if lab not in seen]
    repeated = sorted({lab for lab in seen if seen.count(lab) > 1}, key=lambda x: int(x[1:]))
    problems = []
    if missing:
        problems.append("missing " + ", ".join(missing))
    if repeated:
        problems.append("repeated " + ", ".join(repeated))
    if unknown:
        problems.append("unknown labels " + ", ".join(unknown))
    return order, titles, "; ".join(problems) or None


def ask_once(client: LLMClient, los: list[InputLO], seed: int, max_tokens: int) -> Ask:
    messages, labels = ask_messages(los, seed)
    ask = Ask(seed=seed, labels=labels)
    try:
        resp = client.chat(messages, json_mode=True, max_tokens=max_tokens, label=f"v3_{seed}")
        order, titles, problem = parse_answer(resp.parse_json(), labels)
        if problem:
            ask.repaired = True
            messages = messages + [
                {"role": "assistant", "content": resp.content or ""},
                {"role": "user", "content": ORDER_REPAIR.format(problems=problem, n=len(labels))},
            ]
            resp = client.chat(
                messages, json_mode=True, max_tokens=max_tokens, label=f"v3_{seed}_repair"
            )
            order, titles, problem = parse_answer(resp.parse_json(), labels)
        if problem:
            ask.error = problem
        else:
            ask.order, ask.titles = order, titles
    except LLMError as e:
        ask.error = str(e)
    return ask


# --- 3. consensus order -------------------------------------------------------------------------


def consensus_order(asks: list[Ask]) -> tuple[list[str], dict[str, float]]:
    """LO ids sorted by mean relative position over valid asks (ties by id), and those means."""
    valid = [a for a in asks if a.valid]
    if not valid:
        raise ValueError("no valid order asks")
    ids = sorted(valid[0].labels.values())
    positions: dict[str, list[float]] = {i: [] for i in ids}
    for a in valid:
        n = max(len(a.order) - 1, 1)
        for p, lid in enumerate(a.order):
            positions[lid].append(p / n)
    mean_pos = {i: mean(v) for i, v in positions.items()}
    return sorted(ids, key=lambda i: (mean_pos[i], i)), mean_pos


# --- 4. split asks ------------------------------------------------------------------------------


@dataclass
class SplitAsk:
    seed: int
    labels: dict[str, int]  # code -> position in the fixed order
    cuts: list[int] = field(default_factory=list)  # a module ends after these positions
    n_modules: int = 0
    repaired: bool = False
    error: str | None = None

    @property
    def valid(self) -> bool:
        return self.error is None and self.n_modules > 0


def split_messages(
    order: list[str], texts: dict[str, str], seed: int
) -> tuple[list[dict[str, str]], dict[str, int]]:
    """The fixed order, each LO under a random code (codes shuffled per seed)."""
    codes = list(range(1, len(order) + 1))
    random.Random(f"v3-seg-{seed}").shuffle(codes)
    labels = {f"K{c}": pos for pos, c in enumerate(codes)}
    by_pos = {pos: code for code, pos in labels.items()}
    lines = [f"Course with {len(order)} learning objectives, in teaching order:"]
    lines += [f"{by_pos[pos]}: {texts[lid]}" for pos, lid in enumerate(order)]
    return [
        {"role": "system", "content": SPLIT_PROMPT},
        {"role": "user", "content": "\n".join(lines)},
    ], labels


def parse_split(reply: Any, labels: dict[str, int]) -> tuple[list[int], int, str | None]:
    """(cuts, number of modules, problem or None). Valid only if the modules, read in order,
    give back positions 0..n-1."""
    mods = reply.get("modules") if isinstance(reply, dict) else None
    if not isinstance(mods, list) or not mods:
        return [], 0, "no modules list"
    positions: list[int] = []
    unknown: list[str] = []
    cuts: list[int] = []
    n_modules = 0
    for m in mods:
        items = m.get("los") if isinstance(m, dict) else None
        if not isinstance(items, list):
            return [], 0, "a module has no los list"
        known = [labels[str(x).strip()] for x in items if str(x).strip() in labels]
        unknown += [str(x).strip() for x in items if str(x).strip() not in labels]
        if known:
            positions += known
            cuts.append(len(positions) - 1)
            n_modules += 1
    n = len(labels)
    problems = []
    missing = sorted(set(range(n)) - set(positions))
    if missing:
        code = {pos: c for c, pos in labels.items()}
        problems.append("missing " + ", ".join(code[p] for p in missing))
    if len(positions) != len(set(positions)):
        problems.append("repeated codes")
    if unknown:
        problems.append("unknown codes " + ", ".join(unknown))
    if not problems and positions != list(range(n)):
        problems.append("the order was changed")
    return cuts[:-1], n_modules, "; ".join(problems) or None


def split_once(
    client: LLMClient, order: list[str], texts: dict[str, str], seed: int, max_tokens: int
) -> SplitAsk:
    messages, labels = split_messages(order, texts, seed)
    ask = SplitAsk(seed=seed, labels=labels)
    try:
        resp = client.chat(messages, json_mode=True, max_tokens=max_tokens, label=f"seg_{seed}")
        cuts, n_mod, problem = parse_split(resp.parse_json(), labels)
        if problem:
            ask.repaired = True
            messages = messages + [
                {"role": "assistant", "content": resp.content or ""},
                {"role": "user", "content": SPLIT_REPAIR.format(problems=problem)},
            ]
            resp = client.chat(
                messages, json_mode=True, max_tokens=max_tokens, label=f"seg_{seed}_repair"
            )
            cuts, n_mod, problem = parse_split(resp.parse_json(), labels)
        if problem:
            ask.error = problem
        else:
            ask.cuts, ask.n_modules = cuts, n_mod
    except LLMError as e:
        ask.error = str(e)
    return ask


# --- 5. modules ---------------------------------------------------------------------------------


def split_consensus(asks: list[SplitAsk], order: list[str]) -> tuple[list[list[str]], list[float]]:
    """Modules along `order`: a boundary after position i when a strict majority of valid split
    asks place one there. Returns the modules and the vote share per gap."""
    valid = [a for a in asks if a.valid]
    if not valid:
        raise ValueError("no valid split asks")
    votes = [sum(i in a.cuts for a in valid) / len(valid) for i in range(len(order) - 1)]
    modules = [[order[0]]]
    for i, lid in enumerate(order[1:]):
        if votes[i] > 0.5:
            modules.append([lid])
        else:
            modules[-1].append(lid)
    return modules, votes


# --- pipeline -----------------------------------------------------------------------------------


def build_output(
    modules: list[list[str]], texts: dict[str, str], merged: dict[str, list[str]]
) -> SequencerOutput:
    los = [
        {
            "id": lid,
            "raw_text": texts[lid],
            "source_ids": [lid, *merged.get(lid, [])],
            "depends_on": [],
            **PLACEHOLDER,
        }
        for m in modules
        for lid in m
    ]
    provenance = [
        {"raw_id": src, "lo_id": lid}
        for m in modules
        for lid in m
        for src in [lid, *merged.get(lid, [])]
    ]
    mods = [
        {"id": f"S{k:02d}", "title": f"S{k:02d}", "order": k, "lo_ids": m}
        for k, m in enumerate(modules, 1)
    ]
    return SequencerOutput.model_validate({"modules": mods, "los": los, "provenance": provenance})


@dataclass
class V3Result:
    output: SequencerOutput | None
    merged: dict[str, list[str]]
    asks: list[Ask]
    order: list[str] | None
    mean_position: dict[str, float] | None
    split_asks: list[SplitAsk] = field(default_factory=list)
    split_votes: list[float] | None = None
    stopped: str = ""


def sequence_v3(
    client: LLMClient,
    payload: Agent1Input,
    seeds: list[int],
    max_tokens: int = 4000,
    split_max_tokens: int = 3000,
) -> V3Result:
    """The whole v3 pipeline. Order and split asks use the same seeds."""
    survivors, merged = exact_duplicates(list(payload.los))
    texts = {lo.id: lo.text for lo in payload.los}
    asks = [ask_once(client, survivors, s, max_tokens) for s in seeds]
    n_order = sum(a.valid for a in asks)
    if n_order < MIN_VALID_ASKS:
        return V3Result(None, merged, asks, None, None, stopped=f"{n_order} valid order asks")
    order, mean_pos = consensus_order(asks)
    split_asks = [split_once(client, order, texts, s, split_max_tokens) for s in seeds]
    n_split = sum(a.valid for a in split_asks)
    if n_split < MIN_VALID_ASKS:
        return V3Result(
            None, merged, asks, order, mean_pos, split_asks, stopped=f"{n_split} valid split asks"
        )
    modules, votes = split_consensus(split_asks, order)
    return V3Result(
        build_output(modules, texts, merged),
        merged,
        asks,
        order,
        mean_pos,
        split_asks,
        votes,
        "completed",
    )


def asks_to_json(asks: list[Ask]) -> list[dict[str, Any]]:
    return [
        {
            "seed": a.seed,
            "labels": a.labels,
            "order": a.order,
            "titles": a.titles,
            "repaired": a.repaired,
            "error": a.error,
        }
        for a in asks
    ]


def split_asks_to_json(asks: list[SplitAsk]) -> list[dict[str, Any]]:
    return [
        {
            "seed": a.seed,
            "labels": a.labels,
            "cuts": a.cuts,
            "n_modules": a.n_modules,
            "repaired": a.repaired,
            "error": a.error,
        }
        for a in asks
    ]


# --- checks -------------------------------------------------------------------------------------


def check_output(data: dict[str, Any], payload: Agent1Input) -> dict[str, Any]:
    """The structural checks that apply to a v3 output: schema and references
    (`validate_output`) and provenance of every input LO (`check_provenance`). Provenance needs a
    parsed output, so it is skipped (and says so) when validation cannot parse it."""
    results: dict[str, Any] = {"validate_output": validate_output(data)}
    try:
        out = SequencerOutput.model_validate(data)
    except Exception:  # noqa: BLE001 - validate_output already reported the details
        results["skipped"] = "output does not parse; provenance not checked"
        return results
    results["check_provenance"] = check_provenance(payload.los, out)
    return results


def checks_ok(checks: dict[str, Any]) -> bool:
    return not any(r.errors for k, r in checks.items() if k != "skipped")


def checks_to_json(checks: dict[str, Any]) -> dict[str, Any]:
    return {
        k: (v if isinstance(v, str) else json.loads(v.model_dump_json())) for k, v in checks.items()
    }
