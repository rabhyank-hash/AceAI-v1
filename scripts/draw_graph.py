"""Draw the module graph of Agent 1 runs.

    python scripts/draw_graph.py runs/<run-dir> [...]
    python scripts/draw_graph.py --latest          # newest run of each course

Writes module_graph.md (Mermaid; renders on GitHub and in VS Code's Markdown preview) and
module_graph.html (open in a browser) into each run directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from summarize_runs import latest_runs

from aceai.eval.draw import graph_markdown, to_html
from aceai.ingest.ground_truth import GroundTruth
from aceai.schemas import SequencerOutput


def draw(run_dir: Path) -> None:
    output_path = run_dir / "output.json"
    if not output_path.exists():
        print(f"{run_dir.name}: no output.json, skipped")
        return
    output = SequencerOutput.model_validate_json(output_path.read_text())
    id_map = json.loads((run_dir / "input" / "id_map.json").read_text())
    gt = GroundTruth.model_validate_json((run_dir / "input" / "ground_truth.json").read_text())
    md = graph_markdown(output, id_map, gt)
    (run_dir / "module_graph.md").write_text(md)
    (run_dir / "module_graph.html").write_text(to_html(f"{gt.course}: module graph", md))
    print(f"{run_dir.name}: {run_dir / 'module_graph.html'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*", type=Path)
    ap.add_argument("--latest", action="store_true")
    args = ap.parse_args()
    for d in latest_runs() if args.latest else args.runs:
        draw(d)


if __name__ == "__main__":
    main()
