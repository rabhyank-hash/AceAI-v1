#!/usr/bin/env bash
# The v3 test (docs/v3_test_design.md): three independent v3 runs per course on DataEng,
# CloudAdmin and CloudNative (first 3 CSV modules each), then the analysis tables.
#   bash scripts/run_v3_experiment.sh <experiment id, e.g. E9>
# Runs go to runs/v3/<experiment>/<course>_s<seed>/. The v1 reference is recorded in
# experiments/v3/E6_v1ref/; to rerun it, use branch agent1-poc:
#   scripts/run_poc.py --course <course> --modules 3 --min-los 0 --seed <0|1|2>
# Resumable: a run that already exists is skipped.
set -euo pipefail
cd "$(dirname "$0")/.."
EXP=${1:?usage: run_v3_experiment.sh <experiment id>}
PY=.venv/bin/python
for course in DataEng CloudAdmin CloudNative; do
  for start in 0 5 10; do
    if [[ -d "runs/v3/$EXP/${course}_s${start}" ]]; then echo "skip $EXP ${course}_s${start}"; continue; fi
    $PY scripts/run_v3.py --experiment "$EXP" --course "$course" \
        --seeds $start $((start+1)) $((start+2)) $((start+3)) $((start+4))
  done
done
{ grep -P "^${EXP}_s" runs/experiments.tsv; grep -P "^E6_v1ref_s" runs/experiments.tsv; } > "runs/v3/$EXP.tsv"
$PY scripts/analyze_experiments.py --log "runs/v3/$EXP.tsv" | tee "runs/v3/$EXP.tables.md"
