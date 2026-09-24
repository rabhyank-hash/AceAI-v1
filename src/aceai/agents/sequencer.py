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

from aceai.ingest.agent1_input import Agent1Input
from aceai.llm.client import LLMClient, LLMError, estimate_tokens
from aceai.schemas import SequencerOutput
from aceai.tools import (
    build_module_graph,
    check_cycles,
    check_module_order,
    check_provenance,
    topo_sort_modules,
    validate_output,
)

PROMPT_VERSION = "poc-1"

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
3. Prerequisites: "depends_on" lists the ids of LOs that must be learned first. Only direct, \
clearly necessary prerequisites. No cycles.
4. LOs about course logistics rather than learning (e.g. using the submission platform, \
attempting inline activities) are kept, not dropped: set target_concept to "course logistics" \
and place them where a learner would need them.
5. Modules: group atomic LOs into coherent modules of roughly 3-15 LOs. Every atomic LO appears in \
exactly one module's "lo_ids". Aggregate LOs are never in "lo_ids"; one may head a module as \
"aggregate_lo_id" (or leave it null). Inside a module, list LOs so prerequisites come first, then \
by increasing bloom_level.
6. Order: list modules in teaching order. A module may only depend on earlier modules; module \
"depends_on" lists the earlier module ids it needs. No LO may depend on an LO in a later module.
7. Every input id must appear exactly once, either as a surviving LO "id" or in one "merged_ids". \
Do not invent LOs or ids.

JSON shape (omit keys whose value would be null or []):
{"los": [{"id": "LO-...", "merged_ids": ["LO-..."], "canonical_text": "...", "verb": "...", \
"bloom_level": "C2", "track": "conceptual", "target_concept": "...", "scope": "atomic", \
"parent_id": "LO-...", "depends_on": ["LO-..."]}],
 "modules": [{"id": "M1", "title": "...", "aggregate_lo_id": "LO-...", "lo_ids": ["LO-..."], \
"depends_on": ["M0"]}]}
"""

REPAIR_PROMPT = """\
Deterministic checks found these errors in your JSON:
{issues}

Fix every error and reply with the complete corrected JSON (same shape, all LOs and modules)."""

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


def assemble(reply: Any, payload: Agent1Input) -> Assembled:
    """Turn the model's compact JSON into a SequencerOutput-shaped dict.

    Fills: `source_ids` = [id] + merged_ids; `raw_text` = input text of `id`; module `order` =
    list position; provenance from source_ids. Anything malformed is passed through for
    `validate_output` to report, not repaired here.
    """
    notes: list[str] = []
    text_by_id = {lo.id: lo.text for lo in payload.los}
    if not isinstance(reply, dict):
        return Assembled({"los": [], "modules": [], "provenance": []}, ["reply is not an object"])

    los_out, provenance = [], []
    for i, lo in enumerate(reply.get("los") or []):
        if not isinstance(lo, dict):
            notes.append(f"los[{i}] is not an object; passed through")
            los_out.append(lo)
            continue
        lo = dict(lo)
        merged = lo.pop("merged_ids", None) or []
        lid = lo.get("id")
        sources = [lid, *merged] if isinstance(lid, str) else list(merged)
        lo["source_ids"] = sources
        if "raw_text" not in lo:
            lo["raw_text"] = text_by_id.get(lid, "")
            if lid not in text_by_id:
                notes.append(f"LO {lid!r} is not an input id; raw_text left empty")
        lo.setdefault("depends_on", [])
        los_out.append(lo)
        provenance += [{"raw_id": s, "lo_id": lid} for s in sources if isinstance(s, str)]

    modules_out = []
    for i, m in enumerate(reply.get("modules") or []):
        if isinstance(m, dict):
            m = {**m, "order": i + 1}
            m.setdefault("lo_ids", [])
            m.setdefault("depends_on", [])
        modules_out.append(m)
    return Assembled({"los": los_out, "modules": modules_out, "provenance": provenance}, notes)


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
    output: SequencerOutput | None  # last attempt's output if it parses (errors may remain)
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
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message(payload)},
    ]
    attempts: list[Attempt] = []
    stopped = ""
    for i in range(max_repairs + 1):
        budget = _output_budget(client, messages, max_tokens)
        try:
            resp = client.chat(
                messages, json_mode=True, max_tokens=budget, seed=seed, label=f"attempt_{i}"
            )
        except LLMError as e:  # includes RequestTooLarge
            attempts.append(Attempt(i, None, None, None, {}, error=str(e)))
            stopped = f"LLM call failed: {e}"
            break
        try:
            reply = resp.parse_json()
            err = None
        except LLMError as e:
            reply, err = None, str(e)
        if resp.finish_reason == "length":
            err = (err + "; " if err else "") + "reply truncated at max_tokens"
        assembled = assemble(reply, payload) if reply is not None else None
        checks = run_checks(assembled.data, payload) if assembled else {}
        attempts.append(Attempt(i, resp.content, reply, assembled, checks, error=err))

        if assembled is None:
            issues = f"- your reply was not valid JSON: {err}"
        elif not errors_of(checks):
            stopped = "all checks passed"
            break
        else:
            issues = format_issues(checks)
        if i == max_repairs:
            stopped = f"errors remain after {max_repairs} repair round(s)"
            break
        messages = messages + [
            {"role": "assistant", "content": resp.content or ""},
            {"role": "user", "content": REPAIR_PROMPT.format(issues=issues)},
        ]

    last = attempts[-1]
    output = None
    if last.assembled is not None:
        try:
            output = SequencerOutput.model_validate(last.assembled.data)
        except Exception:  # noqa: BLE001 - reported by validate_output in last.checks
            output = None
    ok = last.assembled is not None and last.error is None and not errors_of(last.checks)
    return SequencerRun(attempts, output, ok, stopped)


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
