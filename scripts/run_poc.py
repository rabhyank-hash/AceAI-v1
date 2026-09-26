"""Agent 1 proof of concept: run the Sequencer on a sample of one course and compare with the CSV.

    python scripts/run_poc.py --units 0 1 2          # PPP units 0-2 on Groq
    python scripts/run_poc.py --units 0 1 2 --dry-run  # build input + prompt, no API call
    python scripts/run_poc.py --all-units --broad    # whole course (may exceed free-tier TPM)
    python scripts/run_poc.py --course CloudAdmin --modules 3 --min-los 20   # module sample

Everything goes under runs/<timestamp>_<course>/; start with its report.md.
Syllabus broad LOs cover the whole course, so they are left out of partial samples unless
--broad is given (they would have nothing to contain).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

from aceai.agents.sequencer import (
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    checks_to_json,
    sequence,
    user_message,
)
from aceai.config import DATA_RAW, DEFAULT_PROVIDER, RUNS_DIR, get_provider
from aceai.eval.compare import compare
from aceai.eval.draw import graph_markdown, to_html
from aceai.ingest import load_all
from aceai.ingest.agent1_input import make_agent1_input
from aceai.ingest.ground_truth import extract_ground_truth
from aceai.llm.client import LLMClient, estimate_tokens
from aceai.schemas import CsvSource, RawLO


def dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str) + "\n")


def select(los: list[RawLO], units: list[int] | None, broad: bool) -> list[RawLO]:
    keep = []
    for lo in los:
        if isinstance(lo.source, CsvSource):
            if units is None or lo.source.unit_no in units:
                keep.append(lo)
        elif broad:
            keep.append(lo)
    return keep


def select_modules(los: list[RawLO], n_modules: int, min_los: int, broad: bool) -> list[RawLO]:
    """The first `n_modules` CSV modules in file order, extended module by module until the
    sample holds at least `min_los` detailed LOs (or the course runs out)."""
    keep: list[RawLO] = []
    seen: list[tuple] = []
    for lo in los:
        if not isinstance(lo.source, CsvSource):
            continue
        key = (lo.source.unit_no, lo.source.module_type, lo.source.module_name)
        if key not in seen:
            if len(seen) >= n_modules and len(keep) >= min_los:
                break
            seen.append(key)
        keep.append(lo)
    if broad:
        keep += [lo for lo in los if not isinstance(lo.source, CsvSource)]
    return keep


def report(run: Any, cmp: dict[str, Any] | None, id_map: dict[str, str], gt: Any) -> str:
    gt_label = {}
    for u in gt.units:
        for m in u.modules:
            for rid in m.lo_ids:
                gt_label[rid] = f"u{u.unit_no} {m.module_type.value} {m.module_name.strip()}"
    lines = ["# Agent 1 POC run", ""]
    if cmp:
        c = cmp["coverage"]
        verdict = "complete" if c["complete"] else "**INVALID: every LO must be placed**"
        lines.append(f"**Coverage: {c['placed']}/{c['total']} LOs placed in a module** ({verdict})")
        lines.append("")
    lines += [f"Stopped: {run.stopped_because}. Passed all checks: {run.ok}.", ""]
    used = next((a for a in run.attempts if a.index == run.output_attempt), None)
    if used is not None and used.assembled is not None:
        if used.assembled.order_rationale:
            lines += [f"**Model's order rationale:** {used.assembled.order_rationale}", ""]
        ordering = [n for n in used.assembled.notes if "set by prerequisites" in n]
        if ordering:
            lines += ["Order changes made by code from the model's prerequisites:", ""]
            lines += [f"- {n}" for n in ordering] + [""]
    if run.output is not None and run.output_attempt != run.attempts[-1].index:
        lines += [
            f"Scores and modules below are from attempt {run.output_attempt}, the last attempt "
            "that produced a usable plan.",
            "",
        ]
    lines.append("| attempt | errors | by tool |")
    lines.append("|---|---|---|")
    for a in run.attempts:
        s = a.summary()
        by = ", ".join(f"{k} {v}" for k, v in s["errors_by_tool"].items() if v) or "-"
        err = f" ({s['error']})" if s["error"] else ""
        lines.append(f"| {a.index} | {s['n_errors']}{err} | {by} |")
    if cmp:
        lines += [
            "",
            "## Comparison with the CSV structure",
            "",
            f"- modules: {cmp['n_pred_modules']} predicted vs {cmp['n_gt_modules']} CSV modules "
            f"in {cmp['n_gt_units']} CSV units",
            f"- surviving LOs: {cmp['n_surviving_los']} (detailed input LOs: {cmp['n_detailed']})",
            f"- merges: {len(cmp['merges'])} "
            f"({sum(not m['same_gt_module'] for m in cmp['merges'])} across CSV modules)",
            "",
            "| vs CSV | BCubed P | BCubed R | BCubed F1 | pairwise P | pairwise R | ARI "
            "| order agreement (pairs) |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for level in ("module", "unit"):
            g, o = cmp["grouping"][level], cmp["order"][level]
            b = g["bcubed"]
            lines.append(
                f"| {level} | {b['precision']} | {b['recall']} | {b['f1']} | {g['precision']} "
                f"| {g['recall']} | {g['ari']} | {o['agreement']} ({o['pairs']}) |"
            )
        lines += [
            "",
            "Grouping is pairwise: two LOs are together if they share a module. ARI 0 = chance,"
            " 1 = identical. Order agreement: random = 0.5; the CSV order is one valid order,"
            " not the only one. Unplaced LOs count as one-LO modules for grouping; order"
            " agreement skips them"
            f" ({cmp['order']['module']['skipped_unplaced_pairs']} pairs skipped at module level).",
            "",
            "The two numbers that are fair whatever module size the model picks: **module"
            " recall** (does it keep the authors' modules together?) and **unit precision**"
            " (does it avoid mixing topics from different units?). Module precision penalizes"
            " merging a unit's CONCEPT and PROJECT modules; unit recall penalizes splitting a"
            " unit into modules as the authors did.",
        ]
    if run.output is not None:
        lo_by_id = {lo.id: lo for lo in run.output.los}
        src = {}
        for e in run.output.provenance:
            src.setdefault(e.lo_id, []).append(id_map.get(e.raw_id, e.raw_id))
        lines += ["", "## Predicted modules (CSV module of each LO in brackets)", ""]
        for m in sorted(run.output.modules, key=lambda m: m.order):
            dep = f" — depends on {', '.join(m.depends_on)}" if m.depends_on else ""
            lines.append(f"### {m.order}. {m.id}: {m.title}{dep}")
            if m.aggregate_lo_id and m.aggregate_lo_id in lo_by_id:
                lines.append(f"*Head:* {lo_by_id[m.aggregate_lo_id].raw_text}")
            for lid in m.lo_ids:
                lo = lo_by_id.get(lid)
                if lo is None:
                    lines.append(f"- {lid} (unknown LO)")
                    continue
                raws = src.get(lid, [])
                where = "; ".join(sorted({gt_label.get(r, "syllabus") for r in raws}))
                text = lo.canonical_text or lo.raw_text
                merged = f" **merged ×{len(raws)}**" if len(raws) > 1 else ""
                lines.append(f"- `{lo.bloom_level}` {text} [{where}]{merged}")
            lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--course", default="PPP")
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument("--units", type=int, nargs="+", default=[0, 1, 2])
    grp.add_argument("--all-units", action="store_true")
    grp.add_argument(
        "--modules", type=int, help="first N CSV modules (extended to reach --min-los LOs)"
    )
    ap.add_argument("--min-los", type=int, default=20, help="with --modules: minimum sample size")
    ap.add_argument("--broad", action="store_true", help="include syllabus broad LOs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--provider", default=DEFAULT_PROVIDER)
    ap.add_argument("--model", default=None)
    ap.add_argument("--max-repairs", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=6000)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument(
        "--reasoning-effort",
        choices=["low", "medium", "high"],
        help="override the model's reasoning_effort (reasoning models only)",
    )
    ap.add_argument("--dry-run", action="store_true", help="write input and prompt; no API call")
    args = ap.parse_args()

    units = None if args.all_units or args.modules else sorted(set(args.units))
    all_los = load_all(DATA_RAW)[args.course]
    if args.modules:
        sample = select_modules(all_los, args.modules, args.min_los, args.broad)
    else:
        sample = select(all_los, units, args.broad)
    prepared = make_agent1_input(args.course, args.seed, los=sample)
    gt = extract_ground_truth(args.course, sample)
    payload = prepared.payload

    client_kw = {"cache_dir": None} if args.no_cache else {}
    provider = get_provider(args.provider)
    if args.reasoning_effort:
        model = args.model or provider.default_model
        params = {**provider.model_params.get(model, {}), "reasoning_effort": args.reasoning_effort}
        provider = replace(provider, model_params={**provider.model_params, model: params})
    client = LLMClient(provider, args.model, **client_kw)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = RUNS_DIR / f"{stamp}_{args.course}"
    config = {
        "course": args.course,
        "units": units,
        "modules": args.modules,
        "min_los": args.min_los if args.modules else None,
        "broad": args.broad,
        "seed": args.seed,
        "provider": client.provider.name,
        "model": client.model,
        "model_params": client.provider.model_params.get(client.model, {}),
        "max_repairs": args.max_repairs,
        "max_tokens": args.max_tokens,
        "prompt_version": PROMPT_VERSION,
        "n_input_los": len(payload.los),
    }
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message(payload)},
    ]
    config["prompt_tokens_estimate"] = estimate_tokens(messages)
    dump(run_dir / "config.json", config)
    dump(run_dir / "input" / "agent1_input.json", payload.model_dump())
    dump(run_dir / "input" / "id_map.json", prepared.id_map)  # never sent to the model
    dump(run_dir / "input" / "text_edits.json", [e.__dict__ for e in prepared.edits])
    dump(run_dir / "input" / "ground_truth.json", gt.model_dump(mode="json"))
    (run_dir / "input" / "prompt.txt").write_text(
        SYSTEM_PROMPT + "\n---\n" + messages[1]["content"]
    )
    print(f"run dir: {run_dir}")
    print(
        f"{len(payload.los)} input LOs, ~{config['prompt_tokens_estimate']} prompt tokens "
        f"+ max_tokens {args.max_tokens}; limits {client.limits}"
    )
    if args.dry_run:
        print("dry run: no API call")
        return 0

    run = sequence(
        client, payload, max_repairs=args.max_repairs, max_tokens=args.max_tokens, seed=args.seed
    )

    for a in run.attempts:
        d = run_dir / "attempts" / f"{a.index:02d}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "reply.txt").write_text(a.reply_text or "")
        if a.assembled is not None:
            dump(d / "assembled.json", a.assembled.data)
            dump(d / "assemble_notes.json", a.assembled.notes)
        dump(d / "checks.json", checks_to_json(a.checks))
        dump(d / "summary.json", a.summary())
    dump(run_dir / "llm_calls.json", client.call_log)

    cmp = None
    if run.output is not None:
        dump(run_dir / "output.json", run.output.model_dump(mode="json"))
        cmp = compare(run.output, prepared.id_map, gt)
        dump(run_dir / "comparison.json", cmp)
    graph_md = ""
    if run.output is not None:
        graph_md = graph_markdown(run.output, prepared.id_map, gt)
        (run_dir / "module_graph.md").write_text(graph_md)
        (run_dir / "module_graph.html").write_text(
            to_html(f"{args.course}: module graph", graph_md)
        )
    (run_dir / "report.md").write_text(report(run, cmp, prepared.id_map, gt) + "\n" + graph_md)
    dump(
        run_dir / "result.json",
        {
            "ok": run.ok,
            "stopped_because": run.stopped_because,
            "attempts": [a.summary() for a in run.attempts],
        },
    )

    usage = [c["response"]["usage"] for c in client.call_log]
    print(f"stopped: {run.stopped_because}; all checks passed: {run.ok}")
    if cmp:
        c = cmp["coverage"]
        print(f"coverage: {c['placed']}/{c['total']}" + ("" if c["complete"] else " INVALID"))
    print("attempt errors:", [a.n_errors for a in run.attempts])
    print("usage:", usage)
    if cmp:
        for level in ("module", "unit"):
            g, o = cmp["grouping"][level], cmp["order"][level]
            print(
                f"vs CSV {level}: grouping P {g['precision']} R {g['recall']} F1 {g['f1']} "
                f"ARI {g['ari']}; order agreement {o['agreement']}"
            )
        print("merges:", len(cmp["merges"]))
    print(f"report: {run_dir / 'report.md'}")
    return 0 if run.output is not None else 1


if __name__ == "__main__":
    sys.exit(main())
