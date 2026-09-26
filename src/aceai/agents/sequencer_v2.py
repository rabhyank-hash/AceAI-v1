"""Agent 1 v2 prototype: consistency by construction (plan v2, §4 steps 4–7).

Every LLM judgment is small, asked several times under different presentation orders, and
aggregated by code:

1. Grouping: consensus of k grouping runs (co-association of input LOs, average linkage).
2. Module prerequisites: for each module, under several permutation seeds, the model names the
   modules that must come before it (`required_before`) and the ones it would still teach first
   (`better_before`). An edge is kept when a strict majority of the asks name it.
3. Order: topological sort; cycles broken at the lowest-vote edge; ties broken by aggregated
   preference (Copeland score among available modules), then share of conceptual LOs, then key.
4. LO order inside modules: LO prerequisites stated by a majority of grouping runs, topological,
   ties by conceptual before applied, then id.

Out of scope for the prototype (see docs/v2_preregistration.md): deduplication and containment
(each input LO stays its own LO), a v2-specific grouping prompt (grouping runs are v1 runs).
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

import networkx as nx

from aceai.ingest.agent1_input import Agent1Input
from aceai.llm.client import LLMClient, LLMError
from aceai.schemas import SequencerOutput

PROMPT_VERSION = "v2-order-1"

ORDER_PROMPT = """\
You are an instructional designer ordering the modules of one course. You are given every \
module of the course as a list of its learning objectives, under neutral labels, and one target \
module. Decide which other modules must be taught before the target module.

- "required_before": modules whose content a learner must already have learned before the \
target module's objectives can be taught (a real dependency).
- "better_before": modules with no hard dependency that you would still teach before the \
target module, for example because they give context or motivation.
- A module appears in at most one list. Leave a list empty if nothing applies.

Reply with JSON only: {"required_before": ["M.."], "better_before": ["M.."], \
"reason": "one sentence"}"""


# --- 1. consensus grouping ----------------------------------------------------------------------


def run_assignments(output: SequencerOutput) -> dict[str, int]:
    """Input LO id -> module index in one grouping run (via provenance: merged LOs sit where
    their surviving LO sits). Unplaced LOs are absent."""
    module_of = {}
    for i, m in enumerate(sorted(output.modules, key=lambda m: m.order)):
        for lid in m.lo_ids:
            module_of[lid] = i
    return {e.raw_id: module_of[e.lo_id] for e in output.provenance if e.lo_id in module_of}


def co_association(ids: list[str], runs: list[dict[str, int]]) -> dict[tuple[str, str], float]:
    """Share of runs in which each pair of LOs is in the same module (missing = not together)."""
    out = {}
    for a, b in combinations(ids, 2):
        together = sum(1 for r in runs if a in r and b in r and r[a] == r[b])
        out[(a, b)] = out[(b, a)] = together / len(runs)
    return out


def consensus_clusters(
    ids: list[str], runs: list[dict[str, int]], threshold: float = 0.5
) -> list[list[str]]:
    """Average-linkage agglomerative clustering of the co-association matrix: merge the two
    clusters with the highest mean co-association while it is above `threshold` (strictly).
    Deterministic: ties go to the pair with the smallest keys. Clusters and members are sorted."""
    co = co_association(ids, runs)
    clusters = [[i] for i in sorted(ids)]

    def link(x: list[str], y: list[str]) -> float:
        return sum(co[(a, b)] for a in x for b in y) / (len(x) * len(y))

    while len(clusters) > 1:
        best = None
        for i, j in combinations(range(len(clusters)), 2):
            score = link(clusters[i], clusters[j])
            key = (-score, clusters[i][0], clusters[j][0])
            if best is None or key < best[0]:
                best = (key, i, j)
        (neg, _, _), i, j = best
        if -neg <= threshold:
            break
        merged = sorted(clusters[i] + clusters[j])
        clusters = [c for k, c in enumerate(clusters) if k not in (i, j)] + [merged]
        clusters.sort(key=lambda c: c[0])
    return clusters


# --- 2. module prerequisite asks ----------------------------------------------------------------


@dataclass
class Ask:
    seed: int
    target: str  # module id
    labels: dict[str, str]  # label shown -> module id
    required: list[str] = field(default_factory=list)  # module ids
    better: list[str] = field(default_factory=list)
    reason: str = ""
    error: str | None = None


def ask_messages(
    modules: dict[str, list[str]], texts: dict[str, str], target: str, seed: int
) -> tuple[list[dict[str, str]], dict[str, str]]:
    """Prompt for one target module under one permutation seed: module order, labels and LO
    order are shuffled. Returns messages and the label -> module id map."""
    rng = random.Random(f"order-{seed}")
    order = sorted(modules)
    rng.shuffle(order)
    labels = {f"M{i}": mid for i, mid in enumerate(order, 1)}
    lines = []
    for label, mid in labels.items():
        los = list(modules[mid])
        rng.shuffle(los)
        lines.append(f"{label}:")
        lines += [f"- {texts[lid]}" for lid in los]
        lines.append("")
    target_label = next(label for label, mid in labels.items() if mid == target)
    lines.append(f"Target module: {target_label}")
    messages = [
        {"role": "system", "content": ORDER_PROMPT},
        {"role": "user", "content": "\n".join(lines)},
    ]
    return messages, labels


def parse_ask(reply: Any, labels: dict[str, str], target: str) -> tuple[list, list, str]:
    """Map labels back to module ids; drop unknown labels and the target; a module named in both
    lists counts as required."""

    def ids(key: str) -> list[str]:
        vals = reply.get(key) if isinstance(reply, dict) else None
        out = []
        for v in vals if isinstance(vals, list) else []:
            mid = labels.get(str(v).strip())
            if mid and mid != target and mid not in out:
                out.append(mid)
        return out

    required = ids("required_before")
    better = [m for m in ids("better_before") if m not in required]
    reason = reply.get("reason", "") if isinstance(reply, dict) else ""
    return required, better, reason if isinstance(reason, str) else ""


def ask_all(
    client: LLMClient,
    modules: dict[str, list[str]],
    texts: dict[str, str],
    seeds: list[int],
    max_tokens: int = 1500,
) -> list[Ask]:
    asks = []
    for seed in seeds:
        for target in sorted(modules):
            messages, labels = ask_messages(modules, texts, target, seed)
            ask = Ask(seed=seed, target=target, labels=labels)
            try:
                resp = client.chat(
                    messages, json_mode=True, max_tokens=max_tokens, label=f"order_{seed}_{target}"
                )
                ask.required, ask.better, ask.reason = parse_ask(resp.parse_json(), labels, target)
            except LLMError as e:
                ask.error = str(e)
            asks.append(ask)
    return asks


# --- 3. aggregation and ordering ----------------------------------------------------------------


@dataclass
class ModuleVotes:
    required: Counter  # (before, after) -> asks naming `before` as required for `after`
    preferred: Counter  # (before, after) -> asks naming `before` as required or better
    asks_per_target: Counter  # module -> answered asks


def aggregate(asks: list[Ask]) -> ModuleVotes:
    required, preferred, answered = Counter(), Counter(), Counter()
    for a in asks:
        if a.error:
            continue
        answered[a.target] += 1
        for m in a.required:
            required[(m, a.target)] += 1
            preferred[(m, a.target)] += 1
        for m in a.better:
            preferred[(m, a.target)] += 1
    return ModuleVotes(required, preferred, answered)


def majority_edges(votes: ModuleVotes) -> dict[tuple[str, str], float]:
    """(before, after) -> vote share, for edges named by a strict majority of `after`'s asks."""
    out = {}
    for (before, after), n in votes.required.items():
        total = votes.asks_per_target[after]
        if total and n / total > 0.5:
            out[(before, after)] = n / total
    return out


def break_cycles(
    modules: list[str], edges: dict[tuple[str, str], float]
) -> tuple[dict[tuple[str, str], float], list[tuple[str, str]]]:
    """Remove the lowest-share edge of each cycle until the graph is acyclic (ties: the edge that
    sorts last). Returns kept edges and removed edges, in removal order."""
    kept = dict(edges)
    removed = []
    while True:
        g = nx.DiGraph()
        g.add_nodes_from(sorted(modules))
        g.add_edges_from(sorted(kept))
        try:
            cycle = nx.find_cycle(g)
        except nx.NetworkXNoCycle:
            return kept, removed
        # min() keeps the first minimum, so iterating in reverse order picks the edge that sorts
        # last among equally weak edges.
        cands = sorted(((u, v) for u, v in cycle), reverse=True)
        worst = min(cands, key=lambda e: kept[e])
        del kept[worst]
        removed.append(worst)


def order_modules(
    modules: list[str],
    edges: dict[tuple[str, str], float],
    votes: ModuleVotes,
    conceptual_share: dict[str, float],
) -> tuple[list[str], list[dict[str, Any]]]:
    """Topological order (Kahn). Among available modules pick by: Copeland score of aggregated
    preference among the available ones, then share of conceptual LOs, then module id. Returns
    the order and a log of every tie decision."""
    preds = {m: {b for (b, a) in edges if a == m} for m in modules}
    done: list[str] = []
    ties = []
    while len(done) < len(modules):
        ready = sorted(m for m in modules if m not in done and preds[m] <= set(done))

        def copeland(x: str, pool: list[str] = ready) -> int:
            score = 0
            for y in pool:
                if y != x:
                    diff = votes.preferred[(x, y)] - votes.preferred[(y, x)]
                    score += (diff > 0) - (diff < 0)
            return score

        ranked = sorted(ready, key=lambda m: (-copeland(m), -conceptual_share.get(m, 0.0), m))
        if len(ready) > 1:
            ties.append(
                {
                    "position": len(done),
                    "candidates": ranked,
                    "copeland": {m: copeland(m) for m in ranked},
                    "conceptual_share": {m: conceptual_share.get(m, 0.0) for m in ranked},
                    "chosen": ranked[0],
                }
            )
        done.append(ranked[0])
    return done, ties


# --- 4. LOs within modules and assembly ---------------------------------------------------------


def majority_lo_edges(outputs: list[SequencerOutput]) -> Counter:
    """(lo, prerequisite) -> number of grouping runs stating it, between LOs that survive as
    themselves (merged LOs are skipped in the prototype)."""
    counts: Counter = Counter()
    for out in outputs:
        for lo in out.los:
            for dep in lo.depends_on:
                counts[(lo.id, dep)] += 1
    return counts


def majority_track(outputs: list[SequencerOutput]) -> dict[str, str]:
    votes: dict[str, Counter] = {}
    for out in outputs:
        survivor = {e.raw_id: e.lo_id for e in out.provenance}
        by_id = {lo.id: lo for lo in out.los}
        for raw, lid in survivor.items():
            if lid in by_id:
                votes.setdefault(raw, Counter())[by_id[lid].track] += 1
    # Ties go to "conceptual" (sorted order), which only matters for ordering rules.
    return {raw: min(c.items(), key=lambda kv: (-kv[1], kv[0]))[0] for raw, c in votes.items()}


def order_within(lo_ids: list[str], edges: set[tuple[str, str]], track: dict[str, str]) -> list:
    g = nx.DiGraph()
    g.add_nodes_from(lo_ids)
    g.add_edges_from((dep, lo) for lo, dep in edges if lo in g and dep in g)
    key = {lid: (track.get(lid) != "conceptual", lid) for lid in lo_ids}
    if not nx.is_directed_acyclic_graph(g):
        return sorted(lo_ids, key=key.get)
    return list(nx.lexicographical_topological_sort(g, key=key.get))


@dataclass
class V2Result:
    output: SequencerOutput
    clusters: list[list[str]]
    asks: list[Ask]
    edges: dict[tuple[str, str], float]
    removed_edges: list[tuple[str, str]]
    ties: list[dict[str, Any]]
    lo_edges: list[tuple[str, str]]


def sequence_v2(
    client: LLMClient,
    payload: Agent1Input,
    grouping_runs: list[SequencerOutput],
    order_seeds: list[int],
) -> V2Result:
    ids = sorted(lo.id for lo in payload.los)
    texts = {lo.id: lo.text for lo in payload.los}
    clusters = consensus_clusters(ids, [run_assignments(o) for o in grouping_runs])
    module_ids = {f"G{i:02d}": c for i, c in enumerate(clusters, 1)}
    member_of = {lid: mid for mid, c in module_ids.items() for lid in c}

    asks = ask_all(client, module_ids, texts, order_seeds)
    votes = aggregate(asks)
    edges, removed = break_cycles(list(module_ids), majority_edges(votes))

    track = majority_track(grouping_runs)
    share = {
        mid: sum(track.get(lid) == "conceptual" for lid in c) / len(c)
        for mid, c in module_ids.items()
    }
    order, ties = order_modules(list(module_ids), edges, votes, share)

    need = len(grouping_runs) // 2 + 1
    lo_counts = majority_lo_edges(grouping_runs)
    lo_edges = {
        (lo, dep)
        for (lo, dep), n in lo_counts.items()
        if n >= need and lo in member_of and member_of.get(dep) == member_of[lo]
    }

    first = {}
    for out in grouping_runs:
        for lo in out.los:
            first.setdefault(lo.id, lo)
    los = []
    for lid in ids:
        src = first.get(lid)
        los.append(
            {
                "id": lid,
                "raw_text": texts[lid],
                "source_ids": [lid],
                "verb": src.verb if src else "unknown",
                "bloom_level": src.bloom_level.value if src else "C1",
                "track": track.get(lid, "conceptual"),
                "target_concept": src.target_concept if src else "unknown",
                "scope": "atomic",
                "depends_on": sorted(dep for lo, dep in lo_edges if lo == lid),
            }
        )
    modules = []
    for pos, mid in enumerate(order, 1):
        modules.append(
            {
                "id": mid,
                "title": mid,
                "order": pos,
                "lo_ids": order_within(module_ids[mid], lo_edges, track),
                "depends_on": sorted(b for (b, a) in edges if a == mid),
            }
        )
    output = SequencerOutput.model_validate(
        {
            "modules": modules,
            "los": los,
            "provenance": [{"raw_id": lid, "lo_id": lid} for lid in ids],
        }
    )
    return V2Result(output, clusters, asks, edges, removed, ties, sorted(lo_edges))
