"""Copy experiment records into experiments/<name>/ for git, without any LO text.

    python scripts/export_records.py v1 [--log runs/experiments.tsv]

runs/ is git-ignored and holds full run data, including course LO text (which, like data/, is
kept out of git). This writes, per run listed in the log: config.json, result.json,
comparison.json, id_map.json (input id -> raw id), usage.json (token counts per LLM call), and
structure.json (the output with every
text field removed: modules with ids, order, members and dependencies; LOs with ids, scope,
parent, prerequisites, Bloom level and track; provenance). It also writes the log with paths
rewritten, so scripts/analyze_experiments.py --log experiments/<name>/experiments.tsv reproduces
the tables from git alone.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from aceai.config import PROJECT_ROOT, RUNS_DIR

TEXT_FIELDS = {"raw_text", "canonical_text", "verb", "target_concept"}


def structure(output: dict) -> dict:
    return {
        "modules": output["modules"],
        "los": [{k: v for k, v in lo.items() if k not in TEXT_FIELDS} for lo in output["los"]],
        "provenance": output["provenance"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--log", type=Path, default=RUNS_DIR / "experiments.tsv")
    args = ap.parse_args()
    out_root = PROJECT_ROOT / "experiments" / args.name
    out_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for line in args.log.read_text().splitlines():
        label, course, d = line.split("\t")
        src = Path(d) if Path(d).is_absolute() else PROJECT_ROOT / d
        dst = out_root / src.name
        dst.mkdir(exist_ok=True)
        for f in ("config.json", "result.json", "comparison.json"):
            if (src / f).exists():
                shutil.copy(src / f, dst / f)
        shutil.copy(src / "input" / "id_map.json", dst / "id_map.json")
        calls = json.loads((src / "llm_calls.json").read_text())
        usage = [
            {"label": c["label"], "cached": c["response"]["cached"], **c["response"]["usage"]}
            for c in calls
        ]
        (dst / "usage.json").write_text(json.dumps(usage, indent=1) + "\n")
        if (src / "output.json").exists():
            output = json.loads((src / "output.json").read_text())
            (dst / "structure.json").write_text(json.dumps(structure(output), indent=1) + "\n")
        rows.append(f"{label}\t{course}\t{dst.relative_to(PROJECT_ROOT)}")
    (out_root / "experiments.tsv").write_text("\n".join(rows) + "\n")
    print(f"{len(rows)} runs -> {out_root}")


if __name__ == "__main__":
    main()
