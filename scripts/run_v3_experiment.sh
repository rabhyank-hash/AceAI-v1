#!/usr/bin/env bash
# The v3 experiment as amended (docs/v3_preregistration.md, Amendment 1), end to end:
# DataEng, CloudAdmin, CloudNative; first 3 CSV modules only; v1 seeds 0-2 as reference;
# three independent v3 runs (seeds 0-4, 5-9, 10-14).
#   bash scripts/run_v3_experiment.sh
# Resumable: a course and label already in runs/experiments.tsv is skipped, and repeated calls
# come from the response cache.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
LOG=runs/experiments.tsv
SAMPLE=(--modules 3 --min-los 0)
logged() { grep -qP "^$1\t$2\t" "$LOG"; }

for course in DataEng CloudAdmin CloudNative; do
  for seed in 0 1 2; do
    label="m3v1_s${seed}"
    if logged "$label" "$course"; then echo "skip $label $course"; continue; fi
    dir=$($PY scripts/run_poc.py --course "$course" "${SAMPLE[@]}" --max-repairs 4 --seed "$seed" \
          | grep '^run dir' | awk '{print $3}')
    printf '%s\t%s\t%s\n' "$label" "$course" "$dir" >> "$LOG"
  done
  for start in 0 5 10; do
    label="m3v3_s${start}"
    if logged "$label" "$course"; then echo "skip $label $course"; continue; fi
    $PY scripts/run_v3.py --course "$course" "${SAMPLE[@]}" --label "$label" \
        --seeds $start $((start+1)) $((start+2)) $((start+3)) $((start+4))
  done
done

grep -P "^m3" "$LOG" > runs/m3_experiment.tsv
$PY scripts/analyze_experiments.py --log runs/m3_experiment.tsv | tee runs/m3_tables.md
