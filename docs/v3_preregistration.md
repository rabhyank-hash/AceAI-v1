# Agent 1 v3 prototype: pre-registration

Written and committed on 28 September 2026, before any v3 code or run. Plan:
[implementation_plan.md](implementation_plan.md) (v3). Deviations will be listed in the report.

## Question

Does v3 (order and split the LOs in one ask, repeat the ask on shuffled input, aggregate in code)
give the same course for the same LOs across runs, and how close is that course to the authors'?

## Hypotheses

- **H1 (consistency), per course, all six courses:**
  - H1a: sequence agreement between two independent v3 runs ≥ 0.90.
  - H1b: module ARI between the two runs ≥ 0.60.
- **H2 (quality), per course:** BCubed F1 against the CSV modules ≥ the v1 mean minus 0.05.
- **H3 (validity):** every run places every LO in exactly one module, merges only identical LOs,
  and passes the structural tools.
- **H4 (aggregation, secondary):** consistency rises with the number of asks k: sequence
  agreement between runs at k = 5 is higher than at k = 1, on average over the six courses.

## Design

- **Courses and samples:** all six, the same samples as v1 (`--modules 3 --min-los 20`,
  21–44 LOs).
- **Model and settings:** `openai/gpt-oss-120b` on Groq, reasoning effort low, temperature 0, as
  in v1 and v2.
- **Runs:** two independent runs per course with k = 5 asks each. Run A uses permutation seeds
  0–4, run B seeds 5–9. No call is shared.
- **Ask:** every LO in a shuffled order, relabelled L1…Ln in that order. The model returns
  `{"modules": [{"title": ..., "los": [labels in teaching order]}, ...]}` in teaching order.
  An answer that misses or repeats a label gets one repair message naming them. If it is still
  wrong, the ask is excluded. A run needs at least 3 valid asks.
- **Exact duplicates:** texts equal after whitespace normalization (case-sensitive). The
  surviving id is the smallest; provenance lists all.
- **Aggregation:** consensus order by mean position over valid asks (ties: id). A module boundary
  between consecutive LOs in the consensus order when a strict majority of valid asks put them in
  different modules.
- **Secondary analysis (H4):** recompute each run from its first k asks, k ∈ {1, 3, 5}, and
  compare the A and B runs at each k. No extra calls.

## Metrics

- **Sequence agreement:** share of LO pairs ordered the same way (0.5 = random). Computed
  between runs, and against the CSV order (file order).
- **Module ARI and BCubed F1:** between runs, and against the CSV modules.
- **Validity:** tool results; coverage; asks excluded.
- **Cost:** calls and tokens per run.
- **References:** v1 values from the existing records (E3; DataEng and PPP with seeds 0–5); v2
  values for DataEng and PPP.

## Analysis

Descriptive, per course. There is one v3 pair per course, so no significance test is used.
Every run is reported. H1–H3 hold only if they hold on all six courses.

## Threats to validity

- One pair of runs per course.
- Mean-position aggregation can place an LO between two topics when asks disagree on topic
  order; the boundary rule then decides its module.
- Temperature 0 and caching: variation comes only from input order, which is what is tested.
- Sequence agreement with the CSV cannot separate errors from valid alternatives; module-order
  labels are still needed.
- The free tier's daily token limit may split the runs across days; the code does not change in
  between.
