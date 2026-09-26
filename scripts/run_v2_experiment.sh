#!/usr/bin/env bash
# The pre-registered v2 prototype experiment (docs/v2_preregistration.md), end to end.
#   bash scripts/run_v2_experiment.sh
# Needs about 150K gpt-oss-120b tokens (Groq free tier: 200K/day). Appends every run to
# runs/experiments.tsv, then prints the analysis tables to runs/v2_tables.md.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
LOG=runs/experiments.tsv
COURSES=(DataEng PPP)

v1_run() {  # course seed -> run dir of the v1 run with that seed (runs it if missing)
  local course=$1 seed=$2 dir
  dir=$(grep -P "^base_s${seed}\t${course}\t" "$LOG" | tail -1 | cut -f3 || true)
  if [[ -z "$dir" ]]; then
    dir=$($PY scripts/run_poc.py --course "$course" --modules 3 --max-repairs 4 --seed "$seed" \
          | grep '^run dir' | awk '{print $3}')
    [[ -f "$dir/output.json" ]] || { echo "v1 $course seed $seed produced no output: $dir" >&2; exit 1; }
    printf 'base_s%s\t%s\t%s\n' "$seed" "$course" "$dir" >> "$LOG"
  fi
  echo "$dir"
}

for course in "${COURSES[@]}"; do
  A=(); B=()
  for s in 0 1 2; do A+=("$(v1_run "$course" "$s")"); done
  for s in 3 4 5; do B+=("$(v1_run "$course" "$s")"); done
  # Labels v2_s0 / v2_s3: analyze_experiments.py groups them as setting "v2" and scores the pair.
  $PY scripts/run_v2.py --course "$course" --label v2_s0 --order-seeds 0 1 2 --grouping-runs "${A[@]}"
  $PY scripts/run_v2.py --course "$course" --label v2_s3 --order-seeds 3 4 5 --grouping-runs "${B[@]}"
done

$PY scripts/analyze_experiments.py | tee runs/v2_tables.md
