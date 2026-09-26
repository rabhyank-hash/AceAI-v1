"""Blank labeling sheets for module-order ground truth (docs/evaluation.md, levels 2-3).

    python scripts/make_label_sheets.py [--modules 3 --min-los 20]

Writes annotations/<course>_module_order.csv: one row per pair of CSV modules in the POC sample,
with the module names and a few example LOs. The labeler fills `answer`:
    A  = A must be taught before B
    B  = B must be taught before A
    -  = no dependency; either order is fine
"""

from __future__ import annotations

import argparse
import csv
from itertools import combinations

from run_poc import select_modules

from aceai.config import DATA_RAW, PROJECT_ROOT
from aceai.ingest import load_all
from aceai.ingest.ground_truth import extract_ground_truth


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modules", type=int, default=3)
    ap.add_argument("--min-los", type=int, default=20)
    args = ap.parse_args()
    out_dir = PROJECT_ROOT / "annotations"
    out_dir.mkdir(exist_ok=True)
    for course, los in load_all(DATA_RAW).items():
        gt = extract_ground_truth(course, select_modules(los, args.modules, args.min_los, False))
        mods = [
            (f"u{u.unit_no} {m.module_type.value} {m.module_name.strip()}", m)
            for u in gt.units
            for m in u.modules
        ]
        path = out_dir / f"{course}_module_order.csv"
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "module_a",
                    "module_b",
                    "answer (A / B / -)",
                    "example LOs in A",
                    "example LOs in B",
                    "notes",
                ]
            )
            for (name_a, a), (name_b, b) in combinations(mods, 2):
                w.writerow(
                    [
                        name_a,
                        name_b,
                        "",
                        " | ".join(lo.text for lo in a.los[:3]),
                        " | ".join(lo.text for lo in b.los[:3]),
                        "",
                    ]
                )
        n = len(mods)
        print(f"{course}: {n} modules, {n * (n - 1) // 2} pairs -> {path}")


if __name__ == "__main__":
    main()
