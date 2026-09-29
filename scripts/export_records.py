"""Copy run records from runs/ into experiments/ for git, without any LO text.

    python scripts/export_records.py --log runs/v3_test.tsv --out experiments/v3/experiments.tsv

runs/ is git-ignored and holds full run data, including course LO text (which, like data/, is
kept out of git). A run at runs/<version>/<experiment>/<course>_s<seed> is recorded at
experiments/<version>/<experiment>/<course>_s<seed>. Per run listed in the log it writes
config.json, result.json, comparison.json, checks.json, id_map.json (input id -> raw id),
usage.json (token counts per LLM call), consensus.json, the asks without the model's free text,
and structure.json (the output with every text field removed and module titles replaced by ids).
It also merges the runs into the experiment log given by --out, so
scripts/analyze_experiments.py --log <that log> reproduces the tables from git alone.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from aceai.config import PROJECT_ROOT, RUNS_DIR

TEXT_FIELDS = {"raw_text", "canonical_text", "verb", "target_concept"}


def structure(output: dict) -> dict:
    return {  # module titles are model free text: replaced by the module id
        "modules": [{**m, "title": m["id"]} for m in output["modules"]],
        "los": [{k: v for k, v in lo.items() if k not in TEXT_FIELDS} for lo in output["los"]],
        "provenance": output["provenance"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", type=Path, required=True, help="runs to export (label, course, dir)")
    ap.add_argument("--out", type=Path, required=True, help="experiment log to merge into")
    args = ap.parse_args()
    exp_root = PROJECT_ROOT / "experiments"
    rows = []
    for line in args.log.read_text().splitlines():
        label, course, d = line.split("\t")
        src = Path(d) if Path(d).is_absolute() else PROJECT_ROOT / d
        dst = exp_root / src.resolve().relative_to(RUNS_DIR.resolve())
        dst.mkdir(parents=True, exist_ok=True)
        for f in ("config.json", "result.json", "comparison.json"):
            if (src / f).exists():
                shutil.copy(src / f, dst / f)
        shutil.copy(src / "input" / "id_map.json", dst / "id_map.json")
        if (src / "llm_calls.json").exists():
            calls = json.loads((src / "llm_calls.json").read_text())
            usage = [
                {"label": c["label"], "cached": c["response"]["cached"], **c["response"]["usage"]}
                for c in calls
            ]
            (dst / "usage.json").write_text(json.dumps(usage, indent=1) + "\n")
        else:  # v3 sub-runs: calls are counted in the parent run
            shutil.copy(src / "usage.json", dst / "usage.json")
        for f in ("consensus.json", "checks.json", "split.json"):
            if (src / f).exists():
                shutil.copy(src / f, dst / f)
        if (src / "split_asks.json").exists():  # split asks: codes and cuts only
            shutil.copy(src / "split_asks.json", dst / "split_asks.json")
        if (src / "asks.json").exists():  # drop the model's free text
            asks = json.loads((src / "asks.json").read_text())
            for a in asks:  # model free text: v2 reasons, v3 module titles
                a.pop("reason", None)
                a.pop("titles", None)
            (dst / "asks.json").write_text(json.dumps(asks, indent=1) + "\n")
        if (src / "output.json").exists():
            output = json.loads((src / "output.json").read_text())
            (dst / "structure.json").write_text(json.dumps(structure(output), indent=1) + "\n")
        rows.append(f"{label}\t{course}\t{dst.relative_to(PROJECT_ROOT)}")
    out = args.out if args.out.is_absolute() else PROJECT_ROOT / args.out
    existing = out.read_text().splitlines() if out.exists() else []
    keys = {tuple(r.split("\t")[:2]) for r in rows}
    merged = [r for r in existing if tuple(r.split("\t")[:2]) not in keys] + rows
    out.write_text("\n".join(merged) + "\n")
    print(f"{len(rows)} runs -> {out.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
