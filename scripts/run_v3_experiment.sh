#!/usr/bin/env bash
# The pre-registered v3 prototype experiment (docs/v3_preregistration.md), end to end.
#   bash scripts/run_v3_experiment.sh
# Two independent runs per course (seeds 0-4 and 5-9, k = 5). Resumable: a course and label
# already in runs/experiments.tsv is skipped, and repeated asks come from the response cache.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
LOG=runs/experiments.tsv
for course in AI_Practitioner CloudAdmin CloudDevOps CloudNative DataEng PPP; do
  for start in 0 5; do
    label="v3_s${start}"
    if grep -qP "^${label}\t${course}\t" "$LOG"; then echo "skip $label $course"; continue; fi
    $PY scripts/run_v3.py --course "$course" --label "$label" \
        --seeds $start $((start+1)) $((start+2)) $((start+3)) $((start+4))
  done
done
$PY scripts/analyze_experiments.py | tee runs/v3_tables.md
