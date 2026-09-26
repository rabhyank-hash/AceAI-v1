"""Agent 1 (Sequencer), proof-of-concept version.

One LLM call proposes the whole tree (normalized LOs, merges, containment, prerequisites, modules,
order). The deterministic tools check it, and any errors are sent back for up to `max_repairs`
correction rounds. The plan's multi-step chain (normalize -> containment -> dedup -> prerequisites
-> grouping -> ordering) replaces this once the POC shows where the model struggles.

The model works in Agent 1 input-id space (`LO-xxxxxx`) and never sees raw ids. To keep replies
short it reuses an input id as each surviving LO's id and lists the ids it merged into it; code
then fills in the mechanical fields (`raw_text`, `source_ids`, module `order`, provenance). Every
fill-in is deterministic and the reply is saved verbatim, so nothing is hidden.

POC-ONLY judgment rules (CLAUDE.md open questions 1, 2, 6) live in `SYSTEM_PROMPT`, not in code.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import networkx as nx

from aceai.ingest.agent1_input import Agent1Input
from aceai.llm.client import InvalidJSONReply, LLMClient, LLMError, estimate_tokens
from aceai.schemas import SequencerOutput
from aceai.tools import (
    build_module_graph,
    check_cycles,
    check_module_order,
    check_provenance,
    topo_sort_modules,
    validate_output,
)

PROMPT_VERSION = "poc-3"

SYSTEM_PROMPT = """\
You are an instructional designer. You receive a shuffled list of learning objectives (LOs) \
for one course. Some are broad course-level goals, most are detailed. Turn them into an ordered \
training plan: a deduplicated, dependency-ordered list of modules. Reply with JSON only.

For every input LO decide:
- verb: the main action verb, lowercase.
- bloom_level: Bloom's cognitive level of that verb as used here, C1 Remember, C2 Understand, \
C3 Apply, C4 Analyze, C5 Evaluate, C6 Create. Ambiguous verbs: "understand", "discuss", \
"describe", "explain" -> C2; "use", "utilize", "implement", "write" -> C3; "compare", \
"debug", "inspect" -> C4; "design", "develop", "build" -> C6 only when the learner produces \
something new, else C3.
- track: "conceptual" (knowing/explaining) or "applied" (doing/building).
- target_concept: the concept or skill, 2-6 words.
- scope: "aggregate" for broad LOs that cover several other LOs, else "atomic".

Rules:
1. Merge duplicates: two LOs are merged ONLY if they state the same objective AND have the same \
track AND the same bloom_level. Keep one of their ids as the surviving id, list the others in \
"merged_ids", and give a "canonical_text" that covers both. Never merge otherwise.
2. Containment: an atomic LO that is part of a broad (aggregate) LO sets "parent_id" to that \
aggregate's id. Every aggregate LO must have at least one child; if no input LO fits under it, \
make it atomic instead. Atomic LOs cannot be parents.
3. Prerequisites decide the teaching order, so they are the most important part of your \
answer. For every LO, "depends_on" lists the LOs a learner must have mastered before this one, \
judged from what the LOs say (e.g. using a pandas DataFrame requires knowing what a DataFrame \
is; a project task requires the concepts it applies). Foundational LOs (overviews, \
introductions, basic definitions, setup) have no prerequisites. Most other LOs have at least \
one. List direct prerequisites only, not ones already implied through another prerequisite. \
Prerequisites may be in other modules. No cycles.
4. LOs about course logistics rather than learning (e.g. using the submission platform, \
attempting inline activities) are kept, not dropped: set target_concept to "course logistics" \
and place them where a learner would need them.
5. Modules: group atomic LOs into coherent modules of roughly 3-15 LOs. Every atomic LO names \
its module in "module". An LO that is merged into another (listed in "merged_ids") does not \
appear as an LO of its own. Aggregate LOs have no "module"; an aggregate LO may head a module \
as its "aggregate_lo_id". "aggregate_lo_id" must be null unless that LO's scope is "aggregate".
6. Order: the module order is computed from your prerequisites: a module comes after every \
module that holds a prerequisite of its LOs, and LOs inside a module come after their \
prerequisites. List modules and LOs in the order you prefer; it is used only where the \
prerequisites allow several orders. In "order_rationale", say in one or two sentences how you \
ordered modules that do not depend on each other.
7. Every input id must appear exactly once, either as a surviving LO "id" or in one \
"merged_ids". Every input LO must end up in the plan: none may be left out. Do not invent LOs \
or ids.

JSON shape (omit keys whose value would be null or []):
{"los": [{"id": "LO-...", "module": "M1", "merged_ids": ["LO-..."], "canonical_text": "...", \
"verb": "...", "bloom_level": "C2", "track": "conceptual", "target_concept": "...", \
"scope": "atomic", "parent_id": "LO-...", "depends_on": ["LO-..."]}],
 "modules": [{"id": "M1", "title": "...", "aggregate_lo_id": "LO-..."}],
 "order_rationale": "..."}
"""

REPAIR_PROMPT = """\
Your current plan, one line per LO (id | module | bloom | track | scope | parent | depends_on | \
merged_ids):
{plan}

Modules in order (id | title | aggregate_lo_id | depends_on, derived from LO prerequisites):
{modules}

Deterministic checks found these errors:
{issues}

Fix every error. Reply with a JSON patch containing ONLY what changes:
{{"los": [objects with "id" plus only the fields that change; a field set to null is removed; \
a new id must be a complete LO],
 "remove_ids": [ids of LOs to delete, e.g. LOs now merged into another],
 "modules": [objects with "id" plus only the fields that change; a new id must be a complete \
module and is added at the end],
 "remove_module_ids": [ids of modules to delete],
 "module_order": [all module ids in the new teaching order, only if the order changes]}}
Omit keys you do not need. Do not repeat unchanged LOs, modules or fields."""

FULL_REPAIR_PROMPT = """\
Your reply could not be used:
{issues}

Reply with the complete JSON (same shape as specified, all LOs and modules)."""

_MAX_ISSUES_IN_REPAIR = 40
_BUDGET_MARGIN = 300  # the token estimate is rough
_MIN_OUTPUT_TOKENS = 1000


def user_message(payload: Agent1Input) -> str:
    lines = [f"Course: {payload.course}", f"{len(payload.los)} learning objectives:"]
    lines += [f"{lo.id}: {lo.text}" for lo in payload.los]
    return "\n".join(lines)


# --- assembling the reply into a SequencerOutput ------------------------------------------------


@dataclass
class Assembled:
    data: dict[str, Any]  # SequencerOutput-shaped dict; may still be invalid
    notes: list[str] = field(default_factory=list)  # mechanical fill-ins and oddities, for the log
    order_rationale: str | None = None  # the model's reason for its tie-breaking preferences


def assemble(reply: Any, payload: Agent1Input) -> Assembled:
    """Turn the model's compact JSON into a SequencerOutput-shaped dict.

    Fills: `source_ids` = [id] + merged_ids; `raw_text` = input text of `id`; module `lo_ids` =
    the LOs naming that module in their "module" field; provenance from source_ids. Then orders
    by prerequisites (plan step 6, see `order_by_prerequisites`): module `depends_on` is lifted
    from LO prerequisites, module `order` is the topological order and LOs inside a module are
    topologically ordered, with the model's listing order breaking ties. Anything malformed is
    passed through for `validate_output` to report (an LO naming an unknown module stays in no
    module), not repaired.
    """
    notes: list[str] = []
    text_by_id = {lo.id: lo.text for lo in payload.los}
    if not isinstance(reply, dict):
        return Assembled({"los": [], "modules": [], "provenance": []}, ["reply is not an object"])

    modules_out: list[Any] = []
    members: dict[str, list[str]] = {}
    for i, m in enumerate(reply.get("modules") or []):
        if isinstance(m, dict):
            m = {**m, "order": i + 1}
            for key in ("lo_ids", "depends_on"):
                if m.get(key):
                    notes.append(f"module {m.get('id')!r}: model-supplied {key} ignored")
            m["lo_ids"] = members.setdefault(str(m.get("id")), [])
            m["depends_on"] = []  # derived from LO prerequisites below
        modules_out.append(m)

    los_out, provenance = [], []
    for i, lo in enumerate(reply.get("los") or []):
        if not isinstance(lo, dict):
            notes.append(f"los[{i}] is not an object; passed through")
            los_out.append(lo)
            continue
        lo = dict(lo)
        merged = lo.pop("merged_ids", None) or []
        module = lo.pop("module", None)
        lid = lo.get("id")
        if isinstance(merged, list):
            # An LO "merged into itself", or a merged id listed twice, adds nothing: drop it.
            clean = [m for i, m in enumerate(merged) if m != lid and m not in merged[:i]]
            if clean != merged:
                notes.append(f"LO {lid!r}: dropped self-references / repeats in merged_ids")
            merged = clean
        if module is not None:
            if str(module) in members:
                members[str(module)].append(lid)
            else:
                notes.append(f"LO {lid!r} names unknown module {module!r}; left in no module")
        sources = [lid, *merged] if isinstance(lid, str) else list(merged)
        lo["source_ids"] = sources
        if "raw_text" not in lo:
            lo["raw_text"] = text_by_id.get(lid, "")
            if lid not in text_by_id:
                notes.append(f"LO {lid!r} is not an input id; raw_text left empty")
        lo.setdefault("depends_on", [])
        los_out.append(lo)
        provenance += [{"raw_id": s, "lo_id": lid} for s in sources if isinstance(s, str)]

    data = {"los": los_out, "modules": modules_out, "provenance": provenance}
    data = order_by_prerequisites(data, notes)
    rationale = reply.get("order_rationale")
    return Assembled(data, notes, rationale if isinstance(rationale, str) else None)


def order_by_prerequisites(data: dict[str, Any], notes: list[str]) -> dict[str, Any]:
    """Plan step 6, done by code: derive module dependencies from LO prerequisites and order
    modules, and LOs inside each module, topologically. Where several orders are valid, the
    model's listing order decides. If the output does not parse or the graph has a cycle, the
    model's order is kept and the tools report the problem."""
    try:
        out = SequencerOutput.model_validate(data)
    except Exception:  # noqa: BLE001 - validate_output reports it
        return data
    derived: dict[str, set[str]] = {}
    for e in build_module_graph(out.modules, out.los).edges:
        if e.lo_edges:
            derived.setdefault(e.from_module, set()).add(e.to_module)
    modules = [
        m.model_copy(update={"depends_on": sorted(derived.get(m.id, set()))}) for m in out.modules
    ]
    by_id = {m["id"]: m for m in data["modules"]}
    for m in modules:
        by_id[m.id]["depends_on"] = list(m.depends_on)

    topo = topo_sort_modules(modules, out.los)
    if not topo.ok:
        notes.append("module graph has a cycle; model order kept")
        return data
    listed = [m.id for m in modules]
    if topo.order != listed:
        notes.append(f"module order set by prerequisites: {listed} -> {topo.order}")
    new_modules = []
    for i, mid in enumerate(topo.order, 1):
        m = {**by_id[mid], "order": i}
        m["lo_ids"] = _order_within(m["lo_ids"], out.los, notes, mid)
        new_modules.append(m)
    return {**data, "modules": new_modules}


def _order_within(lo_ids: list[str], los: list[Any], notes: list[str], mid: str) -> list[str]:
    """Topological order of a module's LOs by their prerequisites inside the module, ties broken
    by listing order. Unchanged if the LOs form a cycle."""
    pos = {lid: i for i, lid in enumerate(lo_ids)}
    g = nx.DiGraph()
    g.add_nodes_from(lo_ids)
    for lo in los:
        if lo.id in pos:
            g.add_edges_from((d, lo.id) for d in lo.depends_on if d in pos)
    if not nx.is_directed_acyclic_graph(g):
        return lo_ids
    ordered = list(nx.lexicographical_topological_sort(g, key=pos.get))
    if ordered != lo_ids:
        notes.append(f"module {mid}: LO order set by prerequisites")
    return ordered


# --- checks -------------------------------------------------------------------------------------


def run_checks(data: dict[str, Any], payload: Agent1Input) -> dict[str, Any]:
    """Run every Agent 1 tool. Graph tools need a schema-valid output, so they are skipped (and
    say so) when validation cannot parse it."""
    results: dict[str, Any] = {"validate_output": validate_output(data)}
    try:
        out = SequencerOutput.model_validate(data)
    except Exception:  # noqa: BLE001 - validate_output already reported the details
        results["skipped"] = "output does not parse; graph and provenance checks not run"
        return results
    results["check_provenance"] = check_provenance(payload.los, out)
    results["check_cycles"] = check_cycles(out.los)
    results["check_module_order"] = check_module_order(out.modules, out.los)
    results["build_module_graph"] = build_module_graph(out.modules, out.los)
    results["topo_sort_modules"] = topo_sort_modules(out.modules, out.los)
    return results


def errors_of(results: dict[str, Any]) -> list[tuple[str, Any]]:
    return [(name, e) for name, r in results.items() if name != "skipped" for e in r.errors]


def format_issues(results: dict[str, Any]) -> str:
    errs = errors_of(results)
    lines = []
    for name, e in errs[:_MAX_ISSUES_IN_REPAIR]:
        missing_ids = [i for i in e.ids if i not in e.message]
        ids = f" (ids: {', '.join(missing_ids)})" if missing_ids else ""
        lines.append(f"- [{name}] {e.code}: {e.message}{ids}")
    if len(errs) > _MAX_ISSUES_IN_REPAIR:
        lines.append(f"- ... and {len(errs) - _MAX_ISSUES_IN_REPAIR} more errors")
    return "\n".join(lines)


# --- the loop -----------------------------------------------------------------------------------


@dataclass
class Attempt:
    index: int
    reply_text: str | None
    reply: Any  # parsed JSON, or None if the reply was not JSON
    assembled: Assembled | None
    checks: dict[str, Any]
    error: str | None = None  # LLM/JSON failure, if any

    @property
    def n_errors(self) -> int:
        return len(errors_of(self.checks)) if self.checks else -1

    def summary(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "error": self.error,
            "n_errors": self.n_errors,
            "errors_by_tool": {k: len(v.errors) for k, v in self.checks.items() if k != "skipped"},
            "skipped": self.checks.get("skipped"),
        }


@dataclass
class SequencerRun:
    attempts: list[Attempt]
    output: SequencerOutput | None  # latest output that parses (tool errors may remain)
    output_attempt: int | None  # which attempt `output` came from
    ok: bool  # last attempt passed every tool with no errors
    stopped_because: str


def sequence(
    client: LLMClient,
    payload: Agent1Input,
    *,
    max_repairs: int = 2,
    max_tokens: int = 6000,
    seed: int | None = 0,
) -> SequencerRun:
    base: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message(payload)},
    ]
    messages = base
    current: dict[str, Any] | None = None  # the full plan so far; repairs patch it
    attempts: list[Attempt] = []
    stopped = ""
    for i in range(max_repairs + 1):
        budget = _output_budget(client, messages, max_tokens)
        try:
            resp = client.chat(
                messages, json_mode=True, max_tokens=budget, seed=seed, label=f"attempt_{i}"
            )
        except InvalidJSONReply as e:
            # The provider refused the reply as invalid JSON: treat it like any bad reply.
            reply_text, parsed, err = e.failed_generation, None, str(e)
        except LLMError as e:  # includes RequestTooLarge
            attempts.append(Attempt(i, None, None, None, {}, error=str(e)))
            stopped = f"LLM call failed: {e}"
            break
        else:
            reply_text = resp.content
            try:
                parsed = resp.parse_json()
                err = None
            except LLMError as e:
                parsed, err = None, str(e)
            if resp.finish_reason == "length":
                err = (err + "; " if err else "") + "reply truncated at max_tokens"

        reply = None
        if parsed is not None:
            if current is None:
                reply = parsed
            elif isinstance(parsed, dict):
                reply = apply_patch(current, parsed)
            else:
                err = (err + "; " if err else "") + "patch is not a JSON object"
        assembled = assemble(reply, payload) if reply is not None else None
        checks = run_checks(assembled.data, payload) if assembled else {}
        attempts.append(Attempt(i, reply_text, reply, assembled, checks, error=err))
        if reply is not None:
            current = reply

        if assembled is None:
            issues = f"- your reply was not usable: {err}"
        elif not errors_of(checks):
            stopped = "all checks passed"
            break
        else:
            issues = format_issues(checks)
        if i == max_repairs:
            stopped = f"errors remain after {max_repairs} repair round(s)"
            break
        # Each round sends the current plan once, not the whole history, so the request size
        # stays constant and fits a small per-minute token budget.
        if current is None:
            messages = base + [
                {"role": "assistant", "content": reply_text or ""},
                {"role": "user", "content": FULL_REPAIR_PROMPT.format(issues=issues)},
            ]
        else:
            plan, mods = _plan_summary(current)
            repair = REPAIR_PROMPT.format(plan=plan, modules=mods, issues=issues)
            messages = base + [{"role": "user", "content": repair}]

    last = attempts[-1]
    output, output_attempt = None, None
    for a in reversed(attempts):  # latest attempt whose output parses, even if a repair failed
        if a.assembled is None:
            continue
        try:
            output = SequencerOutput.model_validate(a.assembled.data)
        except Exception:  # noqa: BLE001 - reported by validate_output in a.checks
            continue
        output_attempt = a.index
        break
    ok = last.assembled is not None and last.error is None and not errors_of(last.checks)
    return SequencerRun(attempts, output, output_attempt, ok, stopped)


def apply_patch(current: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    """Apply a repair patch to the current full reply, mechanically. LOs and modules are both
    matched by id: a patch entry is merged field by field into the existing item (a null value
    removes the field) or appended if the id is new; `remove_ids` / `remove_module_ids` drop
    items; `module_order`, if given, reorders modules (ids it omits keep their relative order at
    the end). Returns a new dict; `current` is not changed."""
    los = _patch_items(current.get("los"), patch.get("los"), patch.get("remove_ids"))
    modules = _patch_items(
        current.get("modules"), patch.get("modules"), patch.get("remove_module_ids")
    )
    order = patch.get("module_order")
    if isinstance(order, list):
        rank = {mid: i for i, mid in enumerate(order)}
        modules.sort(key=lambda m: rank.get(m.get("id"), len(rank)))  # stable for the rest
    out: dict[str, Any] = {"los": los, "modules": modules}
    rationale = patch.get("order_rationale", current.get("order_rationale"))
    if rationale is not None:
        out["order_rationale"] = rationale
    return out


def _patch_items(items: Any, updates: Any, remove: Any) -> list[dict[str, Any]]:
    drop = set(remove or [])
    upd = {u.get("id"): u for u in updates or [] if isinstance(u, dict)}
    out = []
    for item in items or []:
        if not isinstance(item, dict) or item.get("id") in drop:
            continue
        change = upd.pop(item.get("id"), None)
        out.append(_merge(item, change) if change else item)
    out += [_merge({}, u) for uid, u in upd.items() if uid not in drop]
    return out


def _merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    out = {**base, **update}
    return {k: v for k, v in out.items() if v is not None}


def _plan_summary(reply: dict[str, Any]) -> tuple[str, str]:
    """The fields the tools check, one line per LO and per module, to keep repair prompts small."""

    def ids(v: Any) -> str:
        return ",".join(map(str, v)) if isinstance(v, list) and v else "-"

    los = []
    for lo in reply.get("los") or []:
        if isinstance(lo, dict):
            fields = ("id", "module", "bloom_level", "track", "scope", "parent_id")
            cells = [str(lo.get(k) or "-") for k in fields]
            cells += [ids(lo.get("depends_on")), ids(lo.get("merged_ids"))]
            los.append(" | ".join(cells))
    mods = []
    for m in reply.get("modules") or []:
        if isinstance(m, dict):
            cells = [str(m.get(k) or "-") for k in ("id", "title", "aggregate_lo_id")]
            mods.append(" | ".join([*cells, ids(m.get("depends_on"))]))
    return "\n".join(los), "\n".join(mods)


def _output_budget(client: LLMClient, messages: list[dict[str, Any]], max_tokens: int) -> int:
    """`max_tokens`, lowered so prompt + reply fit the per-minute token limit (repair prompts grow
    with the previous reply). If too little is left, the client refuses the call and says so."""
    tpm = client.limits.tokens_per_minute
    if tpm is None:
        return max_tokens
    left = tpm - estimate_tokens(messages) - _BUDGET_MARGIN
    return max(_MIN_OUTPUT_TOKENS, min(max_tokens, left))


def checks_to_json(checks: dict[str, Any]) -> dict[str, Any]:
    return {
        k: (v if isinstance(v, str) else json.loads(v.model_dump_json())) for k, v in checks.items()
    }
