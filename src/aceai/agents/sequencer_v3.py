"""Agent 1 v3 prototype: order and split the LOs, repeated on shuffled input (plan v3, §4).

1. Exact duplicates (code): LOs with identical text after whitespace normalization are merged.
2. Asks (LLM, k times): every LO, shuffled and relabelled L1..Ln, returned as an ordered list of
   modules, each an ordered list of LOs. An answer that misses or repeats labels gets one repair
   message; if still wrong, the ask is excluded.
3. Consensus order (code): mean position over valid asks (Borda), ties by id.
4. Consensus modules (code): a boundary between consecutive LOs of the consensus order when a
   strict majority of valid asks put them in different modules.

Normalization (verb, Bloom level, track, target concept) is not part of Agent 1 in v3. The
output schema still requires those fields, so they are filled with placeholders (see
`PLACEHOLDER`); no score or tool uses them.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from itertools import pairwise
from statistics import mean
from typing import Any

from aceai.ingest.agent1_input import Agent1Input, InputLO
from aceai.llm.client import LLMClient, LLMError
from aceai.schemas import SequencerOutput

PROMPT_VERSION = "v3-2"  # v3-1 lacked the "include every LO" rule; 3 of its answers dropped LOs

PROMPT = """\
You are an instructional designer. You receive every learning objective (LO) of one course, in \
random order, labelled L1 to Ln. Arrange them into a course: put them in the order they should \
be taught, and split that sequence into modules (topics).

Include every LO. Do not drop, merge or skip any, even if it seems redundant, off-topic, or \
about course logistics. Every label from L1 to Ln must appear exactly once.

Reply with JSON only, modules and LOs in teaching order:
{"modules": [{"title": "...", "los": ["L3", "L1", ...]}, ...]}"""

REPAIR = """\
Your answer is not a valid arrangement: {problems}. Every label L1 to L{n} must appear exactly \
once. Reply with the complete JSON again."""

MIN_VALID_ASKS = 3

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


# --- 2. asks ------------------------------------------------------------------------------------


@dataclass
class Ask:
    seed: int
    labels: dict[str, str]  # label -> LO id
    modules: list[list[str]] = field(default_factory=list)  # LO ids, teaching order
    titles: list[str] = field(default_factory=list)
    repaired: bool = False
    error: str | None = None

    @property
    def valid(self) -> bool:
        return self.error is None and bool(self.modules)


def ask_messages(los: list[InputLO], seed: int) -> tuple[list[dict[str, str]], dict[str, str]]:
    order = sorted(los, key=lambda x: x.id)
    random.Random(f"v3-{seed}").shuffle(order)
    labels = {f"L{i}": lo.id for i, lo in enumerate(order, 1)}
    lines = [f"Course with {len(order)} learning objectives:"]
    lines += [f"{label}: {lo.text}" for label, lo in zip(labels, order, strict=True)]
    return [
        {"role": "system", "content": PROMPT},
        {"role": "user", "content": "\n".join(lines)},
    ], labels


def parse_answer(
    reply: Any, labels: dict[str, str]
) -> tuple[list[list[str]], list[str], str | None]:
    """(modules as LO ids, titles, problem description or None)."""
    mods = reply.get("modules") if isinstance(reply, dict) else None
    if not isinstance(mods, list) or not mods:
        return [], [], "no modules list"
    seen: list[str] = []
    unknown: list[str] = []
    modules, titles = [], []
    for m in mods:
        items = m.get("los") if isinstance(m, dict) else None
        if not isinstance(items, list):
            return [], [], "a module has no los list"
        ids = []
        for item in items:
            label = str(item).strip()
            if label in labels:
                seen.append(label)
                ids.append(labels[label])
            else:
                unknown.append(label)
        if ids:
            modules.append(ids)
            titles.append(str(m.get("title", "")) if isinstance(m, dict) else "")
    missing = [lab for lab in labels if lab not in seen]
    repeated = sorted({lab for lab in seen if seen.count(lab) > 1}, key=lambda x: int(x[1:]))
    problems = []
    if missing:
        problems.append("missing " + ", ".join(missing))
    if repeated:
        problems.append("repeated " + ", ".join(repeated))
    if unknown:
        problems.append("unknown labels " + ", ".join(unknown))
    return modules, titles, "; ".join(problems) or None


def ask_once(client: LLMClient, los: list[InputLO], seed: int, max_tokens: int) -> Ask:
    messages, labels = ask_messages(los, seed)
    ask = Ask(seed=seed, labels=labels)
    try:
        resp = client.chat(messages, json_mode=True, max_tokens=max_tokens, label=f"v3_{seed}")
        modules, titles, problem = parse_answer(resp.parse_json(), labels)
        if problem:
            ask.repaired = True
            messages = messages + [
                {"role": "assistant", "content": resp.content or ""},
                {"role": "user", "content": REPAIR.format(problems=problem, n=len(labels))},
            ]
            resp = client.chat(
                messages, json_mode=True, max_tokens=max_tokens, label=f"v3_{seed}_repair"
            )
            modules, titles, problem = parse_answer(resp.parse_json(), labels)
        if problem:
            ask.error = problem
        else:
            ask.modules, ask.titles = modules, titles
    except LLMError as e:
        ask.error = str(e)
    return ask


# --- 3-4. consensus -----------------------------------------------------------------------------


@dataclass
class Consensus:
    order: list[str]  # LO ids
    mean_position: dict[str, float]
    split_votes: list[float]  # per consecutive pair of `order`: share of asks splitting them
    modules: list[list[str]]


def consensus(asks: list[Ask]) -> Consensus:
    valid = [a for a in asks if a.valid]
    if not valid:
        raise ValueError("no valid asks")
    ids = sorted(valid[0].labels.values())
    positions: dict[str, list[float]] = {i: [] for i in ids}
    module_of: list[dict[str, int]] = []
    for a in valid:
        flat = [lid for m in a.modules for lid in m]
        n = max(len(flat) - 1, 1)
        for p, lid in enumerate(flat):
            positions[lid].append(p / n)
        module_of.append({lid: k for k, m in enumerate(a.modules) for lid in m})
    mean_pos = {i: mean(v) for i, v in positions.items()}
    order = sorted(ids, key=lambda i: (mean_pos[i], i))
    votes, modules = [], [[order[0]]]
    for x, y in pairwise(order):
        share = sum(mo[x] != mo[y] for mo in module_of) / len(module_of)
        votes.append(share)
        if share > 0.5:
            modules.append([y])
        else:
            modules[-1].append(y)
    return Consensus(order, mean_pos, votes, modules)


# --- output -------------------------------------------------------------------------------------


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
    asks: list[Ask]
    consensus: Consensus | None
    merged: dict[str, list[str]]
    by_k: dict[int, SequencerOutput]  # output recomputed from the first k valid-or-not asks


def sequence_v3(
    client: LLMClient,
    payload: Agent1Input,
    seeds: list[int],
    max_tokens: int = 4000,
    k_values: tuple[int, ...] = (1, 3, 5),
) -> V3Result:
    survivors, merged = exact_duplicates(list(payload.los))
    texts = {lo.id: lo.text for lo in payload.los}
    asks = [ask_once(client, survivors, s, max_tokens) for s in seeds]
    by_k = {}
    for k in k_values:
        sub = [a for a in asks[:k] if a.valid]
        if sub:
            by_k[k] = build_output(consensus(sub).modules, texts, merged)
    if sum(a.valid for a in asks) < MIN_VALID_ASKS:
        return V3Result(None, asks, None, merged, by_k)
    cons = consensus(asks)
    return V3Result(build_output(cons.modules, texts, merged), asks, cons, merged, by_k)


def asks_to_json(asks: list[Ask]) -> list[dict[str, Any]]:
    return [
        {
            "seed": a.seed,
            "labels": a.labels,
            "modules": a.modules,
            "titles": a.titles,
            "repaired": a.repaired,
            "error": a.error,
        }
        for a in asks
    ]


# --- 5. modules on the fixed order (Amendment 2) ------------------------------------------------

SEGMENT_PROMPT_VERSION = "v3-seg-1"

SEGMENT_PROMPT = """\
You are an instructional designer. You receive the learning objectives (LOs) of one course, \
already in teaching order, each with a random code. Split this sequence into modules: groups of \
consecutive LOs that belong to one topic. Decide how many modules the course needs.

Keep the given order. Include every LO exactly once. Do not move, drop or merge any LO.

Reply with JSON only, modules in order, each listing its LO codes in the given order:
{"modules": [{"title": "...", "los": ["K7", "K2", ...]}, ...]}"""

SEGMENT_REPAIR = """\
Your answer is not a valid split of the given sequence: {problems}. List every code exactly \
once, in the given order, grouped into consecutive modules. Reply with the complete JSON again."""


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
        {"role": "system", "content": SEGMENT_PROMPT},
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
                {"role": "user", "content": SEGMENT_REPAIR.format(problems=problem)},
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


def split_consensus(asks: list[SplitAsk], order: list[str]) -> tuple[list[list[str]], list[float]]:
    """Modules along `order`: a boundary after position i when a strict majority of valid asks
    place one there. Returns the modules and the vote share per gap."""
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
