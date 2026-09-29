#!/usr/bin/env bash
# The v3 test (docs/v3_test_design.md): three independent v3 runs per course on DataEng,
# CloudAdmin and CloudNative (first 3 CSV modules each), then the analysis tables.
#   bash scripts/run_v3_experiment.sh
# The v1 reference runs are recorded in experiments/v3/ (labels m3v1_*); to rerun them, use
# branch agent1-poc: scripts/run_poc.py --course <c> --modules 3 --min-los 0 --seed <0|1|2>.
# Resumable: a label and course already in runs/experiments.tsv is skipped.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
LOG=runs/experiments.tsv
touch "$LOG"
for course in DataEng CloudAdmin CloudNative; do
  for start in 0 5 10; do
    label="v3_s${start}"
    if grep -qP "^${label}\t${course}\t" "$LOG"; then echo "skip $label $course"; continue; fi
    $PY scripts/run_v3.py --course "$course" --modules 3 --min-los 0 --label "$label" \
        --seeds $start $((start+1)) $((start+2)) $((start+3)) $((start+4))
  done
done
grep -P "^(v3_s|m3v1_s)" "$LOG" > runs/v3_test.tsv
$PY scripts/analyze_experiments.py --log runs/v3_test.tsv | tee runs/v3_tables.md
