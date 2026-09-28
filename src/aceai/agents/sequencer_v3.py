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

PROMPT_VERSION = "v3-1"

PROMPT = """\
You are an instructional designer. You receive every learning objective (LO) of one course, in \
random order, labelled L1 to Ln. Arrange them into a course: put them in the order they should \
be taught, and split that sequence into modules (topics). Every label must appear exactly once.

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
