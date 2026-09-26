"""How consistent are Agent 1 runs of the same course with each other (e.g. across seeds)?

    python scripts/consistency.py runs/<a> runs/<b> [runs/<c> ...]

Input ids depend only on the raw LO, not on the seed, so runs can be compared LO by LO. For each
pair of runs: grouping agreement (ARI, BCubed F1), order agreement (share of LO pairs in different
modules in both runs that are ordered the same way), and overlap of LO prerequisite edges
(Jaccard). High values mean the model's structure is stable; low values mean single-run scores
are mostly noise.
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from statistics import mean

from aceai.eval.compare import adjusted_rand_index, bcubed


def has_output(run_dir: Path) -> bool:
    return (run_dir / "output.json").exists() or (run_dir / "structure.json").exists()


def read_output(run_dir: Path) -> dict:
    """The run's output: full (runs/) or text-free structure (experiments/)."""
    for name in ("output.json", "structure.json"):
        if (run_dir / name).exists():
            return json.loads((run_dir / name).read_text())
    raise FileNotFoundError(f"no output in {run_dir}")


def load(run_dir: Path) -> tuple[dict[str, int], set[tuple[str, str]]]:
    """input id -> module position (via provenance), and LO prerequisite edges in input ids."""
    out = read_output(run_dir)
    pos = {}
    for i, m in enumerate(sorted(out["modules"], key=lambda m: m["order"])):
        for lid in m["lo_ids"]:
            pos[lid] = i
    placed = {e["raw_id"]: pos[e["lo_id"]] for e in out["provenance"] if e["lo_id"] in pos}
    edges = {(lo["id"], d) for lo in out["los"] for d in lo["depends_on"]}
    return placed, edges


def pair(a: Path, b: Path) -> dict[str, float | None]:
    pa, ea = load(a)
    pb, eb = load(b)
    ids = sorted(set(pa) & set(pb))
    agree = total = 0
    for x, y in combinations(ids, 2):
        if pa[x] != pa[y] and pb[x] != pb[y]:
            total += 1
            agree += (pa[x] < pa[y]) == (pb[x] < pb[y])
    union = ea | eb
    return {
        "ari": adjusted_rand_index([pa[i] for i in ids], [pb[i] for i in ids]),
        "bcubed_f1": bcubed([pa[i] for i in ids], [pb[i] for i in ids])["f1"],
        "order_agreement": round(agree / total, 3) if total else None,
        "prereq_jaccard": round(len(ea & eb) / len(union), 3) if union else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", type=Path)
    args = ap.parse_args()
    rows = [pair(a, b) for a, b in combinations(args.runs, 2)]
    for (a, b), r in zip(combinations(args.runs, 2), rows, strict=True):
        print(f"{a.name} vs {b.name}: {r}")
    keys = rows[0].keys()
    print("mean:", {k: round(mean(r[k] for r in rows if r[k] is not None), 3) for k in keys})


if __name__ == "__main__":
    main()
