#!/usr/bin/env bash
# v3 Amendment 2 (docs/v3_preregistration.md): split the 9 Amendment-1 runs' consensus orders
# into modules with separate asks, then print the tables for all m3 runs.
#   bash scripts/run_v3_segment_experiment.sh
# Resumable: a label and course already in runs/experiments.tsv is skipped.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
LOG=runs/experiments.tsv
grep -P "^m3v3_s(0|5|10)\t" "$LOG" | while IFS=$'\t' read -r label course dir; do
  seg="m3v3seg_s${label##*_s}"
  if grep -qP "^${seg}\t${course}\t" "$LOG"; then echo "skip $seg $course"; continue; fi
  $PY scripts/run_v3_segment.py --run "$dir" --label "$seg"
done
grep -P "^m3" "$LOG" > runs/m3_experiment.tsv
$PY scripts/analyze_experiments.py --log runs/m3_experiment.tsv | tee runs/m3_tables.md
