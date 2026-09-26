"""Tables for the experiment report, from runs/experiments.tsv (label, course, run dir per line).

    python scripts/analyze_experiments.py [--log runs/experiments.tsv]

Per setting (label prefix before "_s<seed>") and course: valid runs, attempts, BCubed F1 vs CSV
modules, order agreement vs CSV, LO prerequisite links, topological steps, tokens. Across seeds
of the same setting: mean and range, and run-to-run consistency (see scripts/consistency.py).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from statistics import mean

from consistency import has_output, pair, read_output

from aceai.config import PROJECT_ROOT


def metrics(run_dir: Path) -> dict | None:
    if not (run_dir / "result.json").exists():
        return None
    result = json.loads((run_dir / "result.json").read_text())
    if (run_dir / "llm_calls.json").exists():
        calls = json.loads((run_dir / "llm_calls.json").read_text())
        usage = [c["response"]["usage"] for c in calls]
    else:
        usage = json.loads((run_dir / "usage.json").read_text())
    tokens = sum(u.get("total_tokens", 0) for u in usage)
    m = {"attempts": len(result["attempts"]), "ok": result["ok"], "tokens": tokens}
    cmp_path = run_dir / "comparison.json"
    if not cmp_path.exists():
        return {**m, "valid": False}
    c = json.loads(cmp_path.read_text())
    out = read_output(run_dir)
    graph_path = run_dir / "module_graph.md"
    graph = graph_path.read_text() if graph_path.exists() else ""  # not in exported records
    steps = next(
        (
            line.split(" in ")[1].split(" steps")[0]
            for line in graph.splitlines()
            if line.startswith("- ") and " steps " in line
        ),
        "-",
    )
    return {
        **m,
        "valid": result["ok"] and c["coverage"]["complete"],
        "bcubed": c["grouping"]["module"]["bcubed"]["f1"],
        "order": c["order"]["module"]["agreement"],
        "links": sum(len(lo["depends_on"]) for lo in out["los"]),
        "steps": steps,
        "modules": c["n_pred_modules"],
    }


def fmt(values: list) -> str:
    vals = [v for v in values if isinstance(v, int | float)]
    if not vals:
        return "–"
    if len(vals) == 1:
        return f"{vals[0]:.2f}" if isinstance(vals[0], float) else str(vals[0])
    return f"{mean(vals):.2f} ({min(vals):.2f}–{max(vals):.2f})"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", type=Path, default=PROJECT_ROOT / "runs" / "experiments.tsv")
    args = ap.parse_args()
    groups: dict[tuple[str, str], list[Path]] = defaultdict(list)
    for line in args.log.read_text().splitlines():
        label, course, d = line.split("\t")
        setting = label.rsplit("_s", 1)[0]
        path = Path(d) if Path(d).is_absolute() else PROJECT_ROOT / d
        groups[(setting, course)].append(path)

    print(
        "| Setting | Course | Runs | Valid | Attempts | BCubed F1 | Order agr. | Prereq links "
        "| Modules | Tokens/run |"
    )
    print("|---|---|---:|---:|---|---|---|---|---|---:|")
    for (setting, course), dirs in sorted(groups.items()):
        ms = [m for m in (metrics(d) for d in dirs) if m]
        valid = [m for m in ms if m["valid"]]
        print(
            f"| {setting} | {course} | {len(ms)} | {len(valid)} | "
            f"{fmt([m['attempts'] for m in ms])} | {fmt([m['bcubed'] for m in valid])} | "
            f"{fmt([m['order'] for m in valid])} | {fmt([m['links'] for m in valid])} | "
            f"{fmt([m['modules'] for m in valid])} | "
            f"{round(mean(m['tokens'] for m in ms)) if ms else '–'} |"
        )

    print("\nRun-to-run consistency (pairs of seeds of the same setting):\n")
    print(
        "| Setting | Course | Pairs | Grouping ARI | Grouping BCubed F1 | Order agreement "
        "| Prereq edge Jaccard |"
    )
    print("|---|---|---:|---|---|---|---|")
    for (setting, course), dirs in sorted(groups.items()):
        ok = [d for d in dirs if has_output(d)]
        rows = [pair(a, b) for a, b in combinations(ok, 2)]
        if not rows:
            continue
        print(
            f"| {setting} | {course} | {len(rows)} | {fmt([r['ari'] for r in rows])} | "
            f"{fmt([r['bcubed_f1'] for r in rows])} | {fmt([r['order_agreement'] for r in rows])} "
            f"| {fmt([r['prereq_jaccard'] for r in rows])} |"
        )


if __name__ == "__main__":
    main()
