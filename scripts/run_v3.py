"""Run Agent 1 v3 on one course sample (plan v3, §4).

    python scripts/run_v3.py --course DataEng --seeds 0 1 2 3 4 --label v3_s0

Writes runs/<timestamp>_<course>/ (config, result, output, checks, comparison with the CSV,
llm_calls, asks.json, split_asks.json, consensus.json) and appends the run to
runs/experiments.tsv under the given label.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from aceai.agents.sequencer_v3 import (
    ORDER_PROMPT_VERSION,
    SPLIT_PROMPT_VERSION,
    asks_to_json,
    check_output,
    checks_ok,
    checks_to_json,
    sequence_v3,
    split_asks_to_json,
)
from aceai.config import DATA_RAW, RUNS_DIR
from aceai.eval.compare import compare
from aceai.ingest import load_all
from aceai.ingest.agent1_input import make_agent1_input
from aceai.ingest.ground_truth import extract_ground_truth
from aceai.ingest.sample import select_modules
from aceai.llm.client import LLMClient


def dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--course", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    ap.add_argument("--label", required=True, help="e.g. v3_s0")
    ap.add_argument("--modules", type=int, default=3)
    ap.add_argument("--min-los", type=int, default=0)
    args = ap.parse_args()

    sample = select_modules(load_all(DATA_RAW)[args.course], args.modules, args.min_los, False)
    prepared = make_agent1_input(args.course, 0, los=sample)
    gt = extract_ground_truth(args.course, sample)
    # Wait out rate limits (incl. the daily token limit) instead of losing asks to them.
    client = LLMClient("groq", max_retries=30, max_backoff=300)
    result = sequence_v3(client, prepared.payload, args.seeds)

    run_dir = RUNS_DIR / f"{datetime.now():%Y%m%d-%H%M%S}_{args.course}"
    dump(
        run_dir / "config.json",
        {
            "course": args.course,
            "agent": "v3",
            "seed": args.seeds[0],
            "seeds": args.seeds,
            "model": client.model,
            "model_params": client.provider.model_params.get(client.model, {}),
            "order_prompt_version": ORDER_PROMPT_VERSION,
            "split_prompt_version": SPLIT_PROMPT_VERSION,
            "modules": args.modules,
            "min_los": args.min_los,
            "n_input_los": len(prepared.payload.los),
        },
    )
    dump(run_dir / "input" / "id_map.json", prepared.id_map)
    dump(run_dir / "llm_calls.json", client.call_log)
    dump(run_dir / "asks.json", asks_to_json(result.asks))
    dump(run_dir / "split_asks.json", split_asks_to_json(result.split_asks))
    dump(
        run_dir / "consensus.json",
        {
            "merged_exact_duplicates": result.merged,
            "order": result.order,
            "mean_position": result.mean_position,
            "split_votes": result.split_votes,
        },
    )
    ok, cmp = False, None
    if result.output is not None:
        data = json.loads(result.output.model_dump_json())
        checks = check_output(data, prepared.payload)
        ok = checks_ok(checks)
        cmp = compare(result.output, prepared.id_map, gt)
        dump(run_dir / "output.json", data)
        dump(run_dir / "checks.json", checks_to_json(checks))
        dump(run_dir / "comparison.json", cmp)
    stopped = result.stopped if ok or result.output is None else "check errors"
    dump(run_dir / "result.json", {"ok": ok, "stopped_because": stopped, "attempts": []})
    with (RUNS_DIR / "experiments.tsv").open("a") as f:
        f.write(f"{args.label}\t{args.course}\t{run_dir}\n")

    tokens = sum(c["response"]["usage"].get("total_tokens", 0) for c in client.call_log)
    print(f"run dir: {run_dir}")
    print(
        f"order asks {sum(a.valid for a in result.asks)}/{len(result.asks)} valid, "
        f"split asks {sum(a.valid for a in result.split_asks)}/{len(result.split_asks)} valid; "
        f"{tokens} tokens; {stopped}"
    )
    if cmp:
        bcubed = cmp["grouping"]["module"]["bcubed"]["f1"]
        print(
            f"{cmp['n_pred_modules']} modules; BCubed F1 {bcubed}; "
            f"sequence agreement vs CSV {cmp['sequence']['agreement']}"
        )


if __name__ == "__main__":
    main()
