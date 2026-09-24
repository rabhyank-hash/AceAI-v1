"""Agent 1 proof of concept: run the Sequencer on a sample of one course and compare with the CSV.

    python scripts/run_poc.py --units 0 1 2          # PPP units 0-2 on Groq
    python scripts/run_poc.py --units 0 1 2 --dry-run  # build input + prompt, no API call
    python scripts/run_poc.py --all-units --broad    # whole course (may exceed free-tier TPM)

Everything goes under runs/<timestamp>_<course>/; start with its report.md.
Syllabus broad LOs cover the whole course, so they are left out of partial samples unless
--broad is given (they would have nothing to contain).
"""

from __future__ import annotations

import argparse
import json
import sys
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
from aceai.config import DATA_RAW, DEFAULT_PROVIDER, RUNS_DIR
from aceai.eval.compare import compare
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


def report(run: Any, cmp: dict[str, Any] | None, id_map: dict[str, str], gt: Any) -> str:
    gt_label = {}
    for u in gt.units:
        for m in u.modules:
            for rid in m.lo_ids:
                gt_label[rid] = f"u{u.unit_no} {m.module_type.value} {m.module_name.strip()}"
    lines = [
        "# Agent 1 POC run",
        "",
        f"Stopped: {run.stopped_because}. Passed all checks: {run.ok}.",
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
        g, o = cmp["grouping"], cmp["order"]
        lines += [
            "",
            "## Comparison with the CSV structure",
            "",
            f"- modules: {cmp['n_pred_modules']} predicted vs {cmp['n_gt_modules']} in the CSV",
            f"- surviving LOs: {cmp['n_surviving_los']} (detailed input LOs: {cmp['n_detailed']}, "
            f"placed in a module: {cmp['n_detailed_placed']})",
            f"- grouping (pairwise): P {g['precision']}  R {g['recall']}  F1 {g['f1']}",
            f"- order agreement: {o['agreement']} over {o['pairs']} pairs (random = 0.5)",
            f"- merges: {len(cmp['merges'])} "
            f"({sum(not m['same_gt_module'] for m in cmp['merges'])} across CSV modules)",
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
    ap.add_argument("--broad", action="store_true", help="include syllabus broad LOs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--provider", default=DEFAULT_PROVIDER)
    ap.add_argument("--model", default=None)
    ap.add_argument("--max-repairs", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=6000)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="write input and prompt; no API call")
    args = ap.parse_args()

    units = None if args.all_units else sorted(set(args.units))
    all_los = load_all(DATA_RAW)[args.course]
    sample = select(all_los, units, args.broad)
    prepared = make_agent1_input(args.course, args.seed, los=sample)
    gt = extract_ground_truth(args.course, sample)
    payload = prepared.payload

    client_kw = {"cache_dir": None} if args.no_cache else {}
    client = LLMClient(args.provider, args.model, **client_kw)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = RUNS_DIR / f"{stamp}_{args.course}"
    config = {
        "course": args.course,
        "units": units,
        "broad": args.broad,
        "seed": args.seed,
        "provider": client.provider.name,
        "model": client.model,
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
    (run_dir / "report.md").write_text(report(run, cmp, prepared.id_map, gt))
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
    print("attempt errors:", [a.n_errors for a in run.attempts])
    print("usage:", usage)
    if cmp:
        print("grouping:", cmp["grouping"], "order:", cmp["order"], "merges:", len(cmp["merges"]))
    print(f"report: {run_dir / 'report.md'}")
    return 0 if run.output is not None else 1


if __name__ == "__main__":
    sys.exit(main())
