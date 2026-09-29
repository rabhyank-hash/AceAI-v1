"""Re-combine a v3 split run's saved answers with a different boundary tolerance. No LLM calls.

    python scripts/resplit_v3.py --run runs/<split run> --tolerance 1 --label m3v3tol_s0

Reads split_asks.json and split.json of the run, rebuilds the modules with
`split_consensus(..., tolerance)`, and writes a new run directory (usual records; usage.json is
empty because no call is made), appended to runs/experiments.tsv.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from run_poc import dump, select_modules
from run_v3 import write_result

from aceai.agents.sequencer_v3 import SplitAsk, build_output, split_consensus
from aceai.config import DATA_RAW, RUNS_DIR
from aceai.ingest import load_all
from aceai.ingest.agent1_input import make_agent1_input
from aceai.ingest.ground_truth import extract_ground_truth

MIN_VALID = 3


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, required=True, help="a v3 split run directory")
    ap.add_argument("--tolerance", type=int, required=True)
    ap.add_argument("--label", required=True)
    args = ap.parse_args()

    src = json.loads((args.run / "config.json").read_text())
    split = json.loads((args.run / "split.json").read_text())
    asks = [
        SplitAsk(a["seed"], a["labels"], a["cuts"], a["n_modules"], a["repaired"], a["error"])
        for a in json.loads((args.run / "split_asks.json").read_text())
    ]
    course = src["course"]
    sample = select_modules(load_all(DATA_RAW)[course], src["modules"], src["min_los"], False)
    prepared = make_agent1_input(course, 0, los=sample)
    gt = extract_ground_truth(course, sample)
    texts = {lo.id: lo.text for lo in prepared.payload.los}

    output = None
    if sum(a.valid for a in asks) >= MIN_VALID:
        modules, _ = split_consensus(asks, split["order"], tolerance=args.tolerance)
        output = build_output(modules, texts, split["merged"])
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    run_dir = RUNS_DIR / f"{stamp}_{course}"
    config = {**src, "step": "resplit", "tolerance": args.tolerance, "source_run": str(args.run)}
    cmp = write_result(run_dir, config, output, prepared, gt, "all checks passed")
    dump(run_dir / "usage.json", [])
    with (RUNS_DIR / "experiments.tsv").open("a") as f:
        f.write(f"{args.label}\t{course}\t{run_dir}\n")
    if cmp:
        bc = cmp["grouping"]["module"]["bcubed"]["f1"]
        print(f"{run_dir.name}: {cmp['n_pred_modules']} modules; BCubed F1 {bc}")


if __name__ == "__main__":
    main()
