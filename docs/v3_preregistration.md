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

## Amendment 1 (28 September 2026, before any run with prompt v3-2)

Changes to the design above. Hypotheses H1–H4 and their thresholds are unchanged.

- **Prompt v3-2** (commit `39102ff`): states that every LO must be included and none dropped,
  merged or skipped. 3 of the first 50 answers with prompt v3-1 dropped an LO. The 10 v3-1 runs
  are a pilot, reported separately.
- **Courses and samples:** DataEng, CloudAdmin and CloudNative, each limited to its first 3 CSV
  modules (`--modules 3 --min-los 0`): 11, 15 and 44 LOs (28 unique; 16 exact duplicates).
  DataEng is the only first-3-module sample that spans two CSV units; CloudAdmin covers the cloud
  domain (CloudDevOps shares two of its three modules); CloudNative tests the exact-duplicate
  rule. PPP (9 LOs) and AI_Practitioner (8) are too small for pairwise scores.
- **v1 reference on the same samples:** v1 (prompt poc-3) with seeds 0, 1, 2 per course. v1 values
  from other samples are not used.
- **Three independent v3 runs per course** (seeds 0–4, 5–9, 10–14; k = 5), giving 3 run pairs per
  course for v3 as for v1. H1 is evaluated on the mean over the 3 pairs; the range is reported.
- **H2** uses the mean BCubed F1 of the new v1 runs.
- **Rate limits:** calls retry up to 30 times, up to 5 minutes apart, so asks are not lost to the
  daily token limit.
- **Labels:** `m3v1_s<seed>`, `m3v3_s<first seed>`, `m3v3k1_…`, `m3v3k3_…`.

## Amendment 2 (28 September 2026, before any segmentation run)

Motivation: in the amended test (Amendment 1, run at `272ed07`), H1a was met but H1b and H2 were
not. Module boundaries were aggregated along the averaged order, where neighbours often were not
neighbours in the individual asks, and the course was over-split. This amendment moves module
splitting to a separate step on the fixed consensus order (plan v3, §4 step 5). Order is
unchanged.

- **Data reused:** the 9 runs of Amendment 1 (`m3v3_s0/s5/s10`, three courses). Their consensus
  orders are kept as they are; the modules returned inside their asks are not used.
- **Split asks:** per run, 5 asks with the run's own seeds (0–4, 5–9, 10–14). Each ask lists the
  consensus order with labels shuffled per ask (the order itself is fixed) and returns the
  modules as lists of labels in the given order. The model decides the number of modules. An
  answer that changes the order, misses or repeats a label gets one repair message; if still
  wrong, the ask is excluded. A run needs at least 3 valid asks.
- **Aggregation:** a boundary after position i is kept when a strict majority of valid asks place
  one there.
- **Model and settings:** unchanged (`gpt-oss-120b`, low effort, temperature 0).
- **Hypotheses:** H1b (module ARI between runs ≥ 0.60) and H2 (BCubed F1 ≥ v1 mean − 0.05) are
  re-evaluated on the split outputs, on all three courses; thresholds unchanged. H1a is unchanged
  by construction. H3 applies to the split asks.
- **Comparison:** the Amendment 1 outputs (boundaries from the ordering asks) and the exploratory
  median-count re-cut reported with them.
- **Labels:** `m3v3seg_s<first seed>`.
