"""Tools 3-6: prerequisite-graph checks and module ordering.

Edge direction everywhere: `a -> b` means "a depends on b" (b must come first).
"""

from __future__ import annotations

import heapq
import itertools

import networkx as nx
from pydantic import BaseModel, ConfigDict

from aceai.schemas import LearningObjective, Module
from aceai.tools.results import IssueLog, ToolResult

MAX_CYCLES = 20


# --- helpers ------------------------------------------------------------------------------------


def _lo_graph(los: list[LearningObjective]) -> nx.DiGraph:
    g = nx.DiGraph()
    known = {lo.id for lo in los}
    g.add_nodes_from(sorted(known))
    for lo in sorted(los, key=lambda x: x.id):
        for dep in sorted(lo.depends_on):
            if dep in known:
                g.add_edge(lo.id, dep)
    return g


def _canonical_cycle(cycle: list[str]) -> list[str]:
    i = cycle.index(min(cycle))
    return cycle[i:] + cycle[:i]


def _membership(modules: list[Module]) -> dict[str, str]:
    """LO id -> module id (module head included). First module wins if an LO is listed twice."""
    where: dict[str, str] = {}
    for m in modules:
        for lid in [*m.lo_ids, *([m.aggregate_lo_id] if m.aggregate_lo_id else [])]:
            where.setdefault(lid, m.id)
    return where


def _sorted_modules(modules: list[Module]) -> list[Module]:
    return sorted(modules, key=lambda m: (m.order, m.id))


# --- Tool 3: cycles -----------------------------------------------------------------------------


class CycleResult(ToolResult):
    cycles: list[list[str]] = []  # each: [a, b, c] means a depends on b, b on c, c on a
    truncated: bool = False


def check_cycles(los: list[LearningObjective]) -> CycleResult:
    """Find cycles in the LO prerequisite graph (unknown dependency ids are ignored here;
    `validate_output` reports them)."""
    log = IssueLog()
    g = _lo_graph(los)
    found = list(itertools.islice(nx.simple_cycles(g), MAX_CYCLES + 1))
    truncated = len(found) > MAX_CYCLES
    cycles = sorted(_canonical_cycle(c) for c in found[:MAX_CYCLES])
    for c in cycles:
        log.error(
            "cycle",
            "Prerequisite cycle: "
            + " depends on ".join([*c, c[0]])
            + ". Remove or reverse one of these edges.",
            c,
        )
    if truncated:
        log.error("too_many_cycles", f"More than {MAX_CYCLES} cycles; only the first are listed.")
    return log.finish(CycleResult(cycles=cycles, truncated=truncated))


# --- Tool 6: module graph -----------------------------------------------------------------------


class ModuleEdge(BaseModel):
    model_config = ConfigDict(frozen=True)

    from_module: str  # depends on ...
    to_module: str  # ... this module
    lo_edges: list[tuple[str, str]]  # (LO, the LO it depends on) pairs that induce the edge
    declared: bool  # listed in from_module.depends_on


class ModuleGraphResult(ToolResult):
    edges: list[ModuleEdge] = []


def _derived_edges(modules: list[Module], los: list[LearningObjective], log: IssueLog | None):
    where = _membership(modules)
    derived: dict[tuple[str, str], list[tuple[str, str]]] = {}
    for lo in sorted(los, key=lambda x: x.id):
        for dep in sorted(lo.depends_on):
            a, b = where.get(lo.id), where.get(dep)
            if a is None or b is None:
                if log is not None:
                    missing = [x for x, w in ((lo.id, a), (dep, b)) if w is None]
                    log.warn(
                        "dependency_outside_modules",
                        f"Edge {lo.id} -> {dep} is ignored at module level: {missing} not in "
                        "any module.",
                        [lo.id, dep],
                    )
                continue
            if a != b:
                derived.setdefault((a, b), []).append((lo.id, dep))
    return derived


def build_module_graph(modules: list[Module], los: list[LearningObjective]) -> ModuleGraphResult:
    """Lift LO-level `depends_on` edges to module-level edges, merged with declared module deps.

    Warns when an LO-derived edge is not declared in `Module.depends_on`, when a declared edge has
    no LO-level evidence, and when an LO edge touches an LO outside every module.
    """
    log = IssueLog()
    derived = _derived_edges(modules, los, log)
    mod_ids = {m.id for m in modules}
    declared = {(m.id, d) for m in modules for d in m.depends_on if d in mod_ids}
    edges = []
    for a, b in sorted(set(derived) | declared):
        is_declared = (a, b) in declared
        edges.append(
            ModuleEdge(
                from_module=a, to_module=b, lo_edges=derived.get((a, b), []), declared=is_declared
            )
        )
        if not is_declared:
            log.warn(
                "undeclared_module_dependency",
                f"Module {a} depends on {b} through its LOs but does not list {b} in depends_on.",
                [a, b],
            )
        elif (a, b) not in derived:
            log.warn(
                "declared_without_lo_evidence",
                f"Module {a} declares a dependency on {b} that no LO edge supports.",
                [a, b],
            )
    return log.finish(ModuleGraphResult(edges=edges))


# --- Tool 4: module order -----------------------------------------------------------------------


class ModuleOrderResult(ToolResult):
    pass


def check_module_order(modules: list[Module], los: list[LearningObjective]) -> ModuleOrderResult:
    """Modules are taken in ascending `order`.

    Errors: a module declares a dependency on a later module; an LO depends on an LO in a later
    module; an LO depends on a later LO in its own module (order = position in `lo_ids`).
    Warnings: Bloom level decreases between consecutive LOs of a module (plan: modules are
    ordered internally by increasing Bloom level).
    """
    log = IssueLog()
    ordered = _sorted_modules(modules)
    pos = {m.id: i for i, m in enumerate(ordered)}
    where = _membership(modules)
    by_id = {lo.id: lo for lo in los}

    for m in ordered:
        for dep in m.depends_on:
            if dep in pos and pos[dep] > pos[m.id]:
                log.error(
                    "module_depends_on_later_module",
                    f"Module {m.id} (position {pos[m.id]}) depends on module {dep}, which comes "
                    f"later (position {pos[dep]}).",
                    [m.id, dep],
                )

    for lo in sorted(los, key=lambda x: x.id):
        a = where.get(lo.id)
        if a is None:
            continue
        for dep in sorted(lo.depends_on):
            b = where.get(dep)
            if b is None:
                continue
            if pos[b] > pos[a]:
                log.error(
                    "lo_depends_on_later_module",
                    f"LO {lo.id} in module {a} depends on LO {dep} in later module {b}.",
                    [lo.id, dep, a, b],
                )
            elif a == b:
                seq = next(m.lo_ids for m in ordered if m.id == a)
                if lo.id in seq and dep in seq and seq.index(dep) > seq.index(lo.id):
                    log.error(
                        "lo_depends_on_later_lo",
                        f"In module {a}, LO {lo.id} depends on {dep}, which is listed after it.",
                        [lo.id, dep, a],
                    )

    for m in ordered:
        seq = [by_id[i] for i in m.lo_ids if i in by_id]
        drops = [
            f"{x.id} ({x.bloom_level}) -> {y.id} ({y.bloom_level})"
            for x, y in itertools.pairwise(seq)
            if y.bloom_level < x.bloom_level
        ]
        if drops:
            log.warn(
                "bloom_decreases_in_module",
                f"Module {m.id}: Bloom level drops at " + "; ".join(drops) + ".",
                [m.id],
            )
    return log.finish(ModuleOrderResult())


# --- Tool 5: topological sort -------------------------------------------------------------------


class Tie(BaseModel):
    model_config = ConfigDict(frozen=True)

    position: int  # index in the sorted order where the choice was made
    candidates: list[str]  # modules that could all go here (tie-break order: current `order`, id)
    chosen: str


class TopoSortResult(ToolResult):
    order: list[str] = []
    ties: list[Tie] = []
    unique: bool = True  # no ties: the graph allows exactly one order
    matches_current_order: bool = False


def topo_sort_modules(
    modules: list[Module], los: list[LearningObjective] | None = None
) -> TopoSortResult:
    """Deterministic topological order of modules (dependencies first).

    Uses declared `Module.depends_on`, plus LO-derived edges when `los` is given. Ties are broken
    by the modules' current `order`, then id, and every tie is reported so the agent can choose
    differently and justify it.
    """
    log = IssueLog()
    ordered = _sorted_modules(modules)
    rank = {m.id: i for i, m in enumerate(ordered)}
    g = nx.DiGraph()
    g.add_nodes_from(rank)
    for m in ordered:
        for dep in m.depends_on:
            if dep in rank:
                g.add_edge(m.id, dep)
            else:
                log.warn(
                    "unknown_module_dependency",
                    f"Module {m.id} depends on unknown module {dep}; ignored.",
                    [m.id],
                )
    if los is not None:
        for a, b in _derived_edges(modules, los, None):
            g.add_edge(a, b)

    # Kahn's algorithm on "b before a" for each a -> b edge.
    remaining = {n: g.out_degree(n) for n in g.nodes}
    ready = [(rank[n], n) for n, d in remaining.items() if d == 0]
    heapq.heapify(ready)
    order: list[str] = []
    ties: list[Tie] = []
    while ready:
        if len(ready) > 1:
            cands = [n for _, n in sorted(ready)]
            ties.append(Tie(position=len(order), candidates=cands, chosen=cands[0]))
        _, n = heapq.heappop(ready)
        order.append(n)
        for dependent in sorted(g.predecessors(n), key=rank.get):
            remaining[dependent] -= 1
            if remaining[dependent] == 0:
                heapq.heappush(ready, (rank[dependent], dependent))

    if len(order) < len(rank):
        stuck = sorted((n for n in rank if n not in order), key=rank.get)
        cycle = [a for a, _ in nx.find_cycle(g.subgraph(stuck))]
        log.error(
            "module_cycle",
            "Modules form a dependency cycle: "
            + " depends on ".join([*cycle, cycle[0]])
            + f". Unsortable modules: {stuck}.",
            cycle,
        )
    return log.finish(
        TopoSortResult(
            order=order,
            ties=ties,
            unique=not ties,
            matches_current_order=order == [m.id for m in ordered],
        )
    )
