"""One table over several Agent 1 runs, for comparing courses.

    python scripts/summarize_runs.py runs/2026...-PPP runs/2026...-DataEng ...
    python scripts/summarize_runs.py --latest      # newest run of each course

Prints markdown and writes it to runs/summary_<timestamp>.md.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from aceai.config import RUNS_DIR

HEADER = (
    "| Course | Model | Effort | Seed | LOs | CSV modules / units | Model modules | Coverage "
    "| Checks | Attempts "
    "| BCubed F1 (module) | Module recall | Unit precision | Module ARI | Order agreement "
    "| LO prereq links | Merges |\n"
    "|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|"
)


def row(run_dir: Path) -> str:
    config = json.loads((run_dir / "config.json").read_text())
    result = json.loads((run_dir / "result.json").read_text())
    course = config["course"]
    model = config["model"].split("/")[-1]
    effort = config.get("model_params", {}).get("reasoning_effort", "-")
    course = f"{course} | {model} | {effort} | {config['seed']}"
    n_attempts = len(result["attempts"])
    checks = "pass" if result["ok"] else "fail"
    cmp_path = run_dir / "comparison.json"
    if not cmp_path.exists():
        return (
            f"| {course} | {config['n_input_los']} | - | - | - | {checks} | {n_attempts} "
            f"| - | - | - | - | - | - | - |"
        )
    c = json.loads(cmp_path.read_text())
    g_mod, g_unit = c["grouping"]["module"], c["grouping"]["unit"]
    cov = c["coverage"]
    output = json.loads((run_dir / "output.json").read_text())
    links = sum(len(lo["depends_on"]) for lo in output["los"])
    bc = g_mod.get("bcubed", {}).get("f1", "-")
    return (
        f"| {course} | {config['n_input_los']} | {c['n_gt_modules']} / {c['n_gt_units']} "
        f"| {c['n_pred_modules']} | {cov['placed']}/{cov['total']} | {checks} | {n_attempts} "
        f"| {bc} | {g_mod['recall']} | {g_unit['precision']} | {g_mod['ari']} "
        f"| {c['order']['module']['agreement']} | {links} | {len(c['merges'])} |"
    )


def latest_runs() -> list[Path]:
    newest: dict[str, Path] = {}
    for d in sorted(RUNS_DIR.glob("*_*")):
        if (d / "result.json").exists():
            newest[json.loads((d / "config.json").read_text())["course"]] = d
    return [newest[c] for c in sorted(newest)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*", type=Path)
    ap.add_argument("--latest", action="store_true")
    args = ap.parse_args()
    dirs = latest_runs() if args.latest else args.runs
    lines = [HEADER] + [row(d) for d in dirs]
    lines += ["", "Runs: " + ", ".join(d.name for d in dirs)]
    text = "\n".join(lines) + "\n"
    print(text)
    out = RUNS_DIR / f"summary_{datetime.now():%Y%m%d-%H%M%S}.md"
    out.write_text(text)
    print(f"written: {out}")


if __name__ == "__main__":
    main()
