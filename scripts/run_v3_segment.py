"""v3 Amendment 2: split an existing v3 run's consensus order into modules with separate asks.

    python scripts/run_v3_segment.py --run runs/<v3 run> --label m3v3seg_s0

Uses the run's consensus order and seeds (docs/v3_preregistration.md, Amendment 2). Writes a new
run directory with the usual records plus split_asks.json and split.json, and appends it to
runs/experiments.tsv.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from run_poc import dump, select_modules
from run_v3 import write_result

from aceai.agents.sequencer_v3 import (
    SEGMENT_PROMPT_VERSION,
    build_output,
    split_asks_to_json,
    split_consensus,
    split_once,
)
from aceai.config import DATA_RAW, RUNS_DIR
from aceai.eval.draw import graph_markdown, to_html
from aceai.ingest import load_all
from aceai.ingest.agent1_input import make_agent1_input
from aceai.ingest.ground_truth import extract_ground_truth
from aceai.llm.client import LLMClient

MIN_VALID = 3


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, required=True, help="a v3 run directory")
    ap.add_argument("--label", required=True)
    ap.add_argument("--max-tokens", type=int, default=3000)
    args = ap.parse_args()

    src = json.loads((args.run / "config.json").read_text())
    cons = json.loads((args.run / "consensus.json").read_text())
    if src.get("agent") != "v3" or not cons.get("order"):
        raise SystemExit(f"{args.run} is not a v3 run with a consensus order")
    course, seeds = src["course"], src["seeds"]
    sample = select_modules(load_all(DATA_RAW)[course], src["modules"], src["min_los"], False)
    prepared = make_agent1_input(course, 0, los=sample)
    gt = extract_ground_truth(course, sample)
    texts = {lo.id: lo.text for lo in prepared.payload.los}
    order, merged = cons["order"], cons["merged_exact_duplicates"]

    client = LLMClient("groq", max_retries=30, max_backoff=300)
    asks = [split_once(client, order, texts, s, args.max_tokens) for s in seeds]
    n_valid = sum(a.valid for a in asks)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = RUNS_DIR / f"{stamp}_{course}"
    config = {
        **src,
        "step": "split",
        "source_run": str(args.run),
        "order_prompt_version": src["prompt_version"],
        "prompt_version": SEGMENT_PROMPT_VERSION,
        "model": client.model,
        "model_params": client.provider.model_params.get(client.model, {}),
    }
    output = None
    votes = None
    if n_valid >= MIN_VALID:
        modules, votes = split_consensus(asks, order)
        output = build_output(modules, texts, merged)
    stopped = "all checks passed" if output else f"only {n_valid} valid split asks"
    cmp = write_result(run_dir, config, output, prepared, gt, stopped)
    dump(run_dir / "llm_calls.json", client.call_log)
    dump(run_dir / "split_asks.json", split_asks_to_json(asks))
    dump(run_dir / "split.json", {"order": order, "cut_votes": votes, "merged": merged})
    if output is not None:
        graph = graph_markdown(output, prepared.id_map, gt)
        (run_dir / "module_graph.md").write_text(graph)
        (run_dir / "module_graph.html").write_text(to_html(f"{course} v3 split", graph))
    with (RUNS_DIR / "experiments.tsv").open("a") as f:
        f.write(f"{args.label}\t{course}\t{run_dir}\n")

    tokens = sum(c["response"]["usage"].get("total_tokens", 0) for c in client.call_log)
    print(f"run dir: {run_dir}")
    print(
        f"split asks: {n_valid}/{len(asks)} valid ({sum(a.repaired for a in asks)} repaired); "
        f"modules per ask {[a.n_modules for a in asks]}; {tokens} tokens"
    )
    if cmp:
        bcubed = cmp["grouping"]["module"]["bcubed"]["f1"]
        print(f"{cmp['n_pred_modules']} modules; BCubed F1 {bcubed}")


if __name__ == "__main__":
    main()
