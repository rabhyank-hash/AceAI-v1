"""Agent 1 v3 prototype run (docs/v3_preregistration.md).

    python scripts/run_v3.py --course DataEng --seeds 0 1 2 3 4 --label v3_s0

Writes runs/<timestamp>_<course>/ with the same record files as other runs (config, result,
output, comparison, llm_calls, module graph) plus asks.json and consensus.json. The courses
recomputed from the first k asks (k = 1, 3) are written to k1/ and k3/ inside the run directory.
Every run and sub-run is appended to runs/experiments.tsv (labels v3_s<seed>, v3k1_s<seed>,
v3k3_s<seed>).
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from run_poc import dump, select_modules

from aceai.agents.sequencer import checks_to_json, run_checks
from aceai.agents.sequencer_v3 import PROMPT_VERSION, asks_to_json, sequence_v3
from aceai.config import DATA_RAW, RUNS_DIR
from aceai.eval.compare import compare
from aceai.eval.draw import graph_markdown, to_html
from aceai.ingest import load_all
from aceai.ingest.agent1_input import make_agent1_input
from aceai.ingest.ground_truth import extract_ground_truth
from aceai.llm.client import LLMClient
from aceai.schemas import SequencerOutput


def write_result(
    run_dir: Path, config: dict, output: SequencerOutput | None, prepared, gt, stopped: str
) -> dict | None:
    dump(run_dir / "config.json", config)
    dump(run_dir / "input" / "id_map.json", prepared.id_map)
    cmp = None
    ok = False
    if output is not None:
        data = json.loads(output.model_dump_json())
        checks = run_checks(data, prepared.payload)
        ok = not any(r.errors for k, r in checks.items() if k != "skipped")
        cmp = compare(output, prepared.id_map, gt)
        dump(run_dir / "output.json", data)
        dump(run_dir / "checks.json", checks_to_json(checks))
        dump(run_dir / "comparison.json", cmp)
        if not ok:
            stopped = "tool errors"
    dump(run_dir / "result.json", {"ok": ok, "stopped_because": stopped, "attempts": []})
    return cmp


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--course", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    ap.add_argument("--label", required=True, help="e.g. v3_s0")
    ap.add_argument("--modules", type=int, default=3)
    ap.add_argument("--min-los", type=int, default=20)
    args = ap.parse_args()

    sample = select_modules(load_all(DATA_RAW)[args.course], args.modules, args.min_los, False)
    prepared = make_agent1_input(args.course, 0, los=sample)
    gt = extract_ground_truth(args.course, sample)
    client = LLMClient("groq")
    result = sequence_v3(client, prepared.payload, args.seeds)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = RUNS_DIR / f"{stamp}_{args.course}"
    base = {
        "course": args.course,
        "agent": "v3",
        "seed": args.seeds[0],
        "seeds": args.seeds,
        "model": client.model,
        "model_params": client.provider.model_params.get(client.model, {}),
        "prompt_version": PROMPT_VERSION,
        "modules": args.modules,
        "min_los": args.min_los,
        "n_input_los": len(prepared.payload.los),
    }
    n_valid = sum(a.valid for a in result.asks)
    stopped = "all checks passed" if result.output else f"only {n_valid} valid asks"
    config = {**base, "k": len(args.seeds)}
    cmp = write_result(run_dir, config, result.output, prepared, gt, stopped)
    dump(run_dir / "llm_calls.json", client.call_log)
    dump(run_dir / "asks.json", asks_to_json(result.asks))
    dump(
        run_dir / "consensus.json",
        {
            "merged_exact_duplicates": result.merged,
            "order": result.consensus.order if result.consensus else None,
            "mean_position": result.consensus.mean_position if result.consensus else None,
            "split_votes": result.consensus.split_votes if result.consensus else None,
        },
    )
    if result.output is not None:
        graph = graph_markdown(result.output, prepared.id_map, gt)
        (run_dir / "module_graph.md").write_text(graph)
        (run_dir / "module_graph.html").write_text(to_html(f"{args.course} v3", graph))

    lines = [f"{args.label}\t{args.course}\t{run_dir}"]
    setting, seed = args.label.rsplit("_s", 1)
    for k, out in sorted(result.by_k.items()):
        if k == len(args.seeds):
            continue
        sub = run_dir / f"k{k}"
        write_result(sub, {**base, "k": k, "seeds": args.seeds[:k]}, out, prepared, gt, "subset")
        dump(sub / "usage.json", [])  # the calls are counted once, in the parent run
        lines.append(f"{setting}k{k}_s{seed}\t{args.course}\t{sub}")
    with (RUNS_DIR / "experiments.tsv").open("a") as f:
        f.write("\n".join(lines) + "\n")

    uncached = sum(1 for c in client.call_log if not c["response"]["cached"])
    tokens = sum(c["response"]["usage"].get("total_tokens", 0) for c in client.call_log)
    print(f"run dir: {run_dir}")
    print(
        f"asks: {n_valid}/{len(result.asks)} valid "
        f"({sum(a.repaired for a in result.asks)} repaired); {uncached} uncached calls, "
        f"{tokens} tokens; merged duplicates: {sum(len(v) for v in result.merged.values())}"
    )
    if cmp:
        print(
            f"{cmp['n_pred_modules']} modules; coverage {cmp['coverage']['placed']}/"
            f"{cmp['coverage']['total']}; BCubed F1 {cmp['grouping']['module']['bcubed']['f1']}; "
            f"sequence agreement vs CSV {cmp['sequence']['agreement']}"
        )


if __name__ == "__main__":
    main()
