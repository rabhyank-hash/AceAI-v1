"""Agent 1 v2 prototype run (docs/v2_preregistration.md).

    python scripts/run_v2.py --course DataEng --label v2_s0 --order-seeds 0 1 2 \\
        --grouping-runs runs/<v1 seed 0> runs/<v1 seed 1> runs/<v1 seed 2>

Grouping runs must be v1 runs of the same course and sample. Writes runs/<timestamp>_<course>/
with the same record files as a v1 run (config, result, output, comparison, llm_calls, report,
module graph) plus consensus.json, asks.json and module_edges.json, and appends the run to
runs/experiments.tsv under the given label.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from run_poc import dump, select_modules

from aceai.agents.sequencer import checks_to_json, run_checks
from aceai.agents.sequencer_v2 import PROMPT_VERSION, sequence_v2
from aceai.config import DATA_RAW, RUNS_DIR
from aceai.eval.compare import compare
from aceai.eval.draw import graph_markdown, to_html
from aceai.ingest import load_all
from aceai.ingest.agent1_input import make_agent1_input
from aceai.ingest.ground_truth import extract_ground_truth
from aceai.llm.client import LLMClient
from aceai.schemas import SequencerOutput


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--course", required=True)
    ap.add_argument("--grouping-runs", nargs="+", type=Path, required=True)
    ap.add_argument("--order-seeds", nargs="+", type=int, required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--modules", type=int, default=3)
    ap.add_argument("--min-los", type=int, default=20)
    args = ap.parse_args()

    sample = select_modules(load_all(DATA_RAW)[args.course], args.modules, args.min_los, False)
    prepared = make_agent1_input(args.course, 0, los=sample)  # ids do not depend on the seed
    gt = extract_ground_truth(args.course, sample)

    configs = [json.loads((d / "config.json").read_text()) for d in args.grouping_runs]
    for d, c in zip(args.grouping_runs, configs, strict=True):
        same = (
            c["course"] == args.course
            and c.get("modules") == args.modules
            and c.get("min_los") == args.min_los
            and c["n_input_los"] == len(prepared.payload.los)
        )
        if not same:
            raise SystemExit(f"{d} is not a v1 run of this course and sample")
    if len({(c["model"], json.dumps(c.get("model_params"), sort_keys=True)) for c in configs}) != 1:
        raise SystemExit("grouping runs use different models or settings")
    runs = [
        SequencerOutput.model_validate_json((d / "output.json").read_text())
        for d in args.grouping_runs
    ]

    client = LLMClient("groq", configs[0]["model"])
    result = sequence_v2(client, prepared.payload, runs, args.order_seeds)
    data = json.loads(result.output.model_dump_json())
    checks = run_checks(data, prepared.payload)
    ok = not any(r.errors for k, r in checks.items() if k != "skipped")
    cmp = compare(result.output, prepared.id_map, gt)
    failed_asks = sum(1 for a in result.asks if a.error)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = RUNS_DIR / f"{stamp}_{args.course}"
    dump(
        run_dir / "config.json",
        {
            "course": args.course,
            "agent": "v2",
            "label": args.label,
            "seed": args.order_seeds[0],
            "order_seeds": args.order_seeds,
            "grouping_runs": [str(d) for d in args.grouping_runs],
            "grouping_seeds": [c["seed"] for c in configs],
            "model": client.model,
            "model_params": client.provider.model_params.get(client.model, {}),
            "prompt_version": PROMPT_VERSION,
            "grouping_prompt_version": configs[0]["prompt_version"],
            "modules": args.modules,
            "min_los": args.min_los,
            "n_input_los": len(prepared.payload.los),
        },
    )
    dump(run_dir / "input" / "id_map.json", prepared.id_map)
    dump(run_dir / "input" / "ground_truth.json", gt.model_dump(mode="json"))
    dump(run_dir / "consensus.json", {"clusters": result.clusters})
    dump(
        run_dir / "asks.json",
        [
            {
                "seed": a.seed,
                "target": a.target,
                "labels": a.labels,
                "required": a.required,
                "better": a.better,
                "reason": a.reason,
                "error": a.error,
            }
            for a in result.asks
        ],
    )
    dump(
        run_dir / "module_edges.json",
        {
            "kept": [[b, a, share] for (b, a), share in sorted(result.edges.items())],
            "removed_for_cycles": result.removed_edges,
            "ties": result.ties,
            "lo_edges": result.lo_edges,
        },
    )
    dump(run_dir / "output.json", data)
    dump(run_dir / "checks.json", checks_to_json(checks))
    dump(run_dir / "comparison.json", cmp)
    dump(run_dir / "llm_calls.json", client.call_log)
    stopped = "all checks passed" if ok else "tool errors"
    if failed_asks:
        stopped += f"; {failed_asks} of {len(result.asks)} asks failed"
    dump(run_dir / "result.json", {"ok": ok, "stopped_because": stopped, "attempts": []})
    graph_md = graph_markdown(result.output, prepared.id_map, gt)
    (run_dir / "module_graph.md").write_text(graph_md)
    (run_dir / "module_graph.html").write_text(to_html(f"{args.course} v2: module graph", graph_md))
    with (RUNS_DIR / "experiments.tsv").open("a") as f:
        f.write(f"{args.label}\t{args.course}\t{run_dir}\n")

    g = cmp["grouping"]["module"]
    print(f"run dir: {run_dir}")
    print(f"checks: {stopped}; coverage {cmp['coverage']['placed']}/{cmp['coverage']['total']}")
    uncached = sum(1 for c in client.call_log if not c["response"]["cached"])
    print(
        f"{len(result.clusters)} modules, {len(result.edges)} module edges "
        f"({len(result.removed_edges)} removed for cycles), {len(result.ties)} tie decisions, "
        f"{len(result.asks)} asks, {uncached} uncached calls"
    )
    print(f"BCubed F1 {g['bcubed']['f1']}, order agreement {cmp['order']['module']['agreement']}")


if __name__ == "__main__":
    main()
