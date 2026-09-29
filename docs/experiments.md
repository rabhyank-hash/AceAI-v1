# Experiment log

Every Agent 1 experiment, with its configuration and outcome. Records (configs, scores, output
structures, token counts; no LO text) are in `experiments/<version>/<experiment>/<course>_s<seed>/`
and listed in `experiments/<version>/experiments.tsv` with label `<experiment>_s<seed>`. Full run
data, including LO text, stays local in `runs/` under the same names. Unless stated otherwise: model `openai/gpt-oss-120b` on Groq, reasoning
effort low, temperature 0.

Sample names: **≥20-LO sample** = a course's first 3 CSV modules, extended module by module to at
least 20 LOs (`--modules 3 --min-los 20`, 21–44 LOs). **3-module sample** = exactly the first 3
CSV modules (`--modules 3 --min-los 0`: DataEng 11, CloudAdmin 15, CloudNative 44 LOs).

## Current state (v3)

v3 = order stage of E6 + split stage of E7 (same seeds per run). Against the
[test design](v3_test_design.md):

| Criterion | DataEng | CloudAdmin | CloudNative | Met |
|---|---|---|---|---|
| C1 sequence agreement between runs ≥ 0.90 | 0.92 | 0.93 | 0.91 | yes |
| C2 module ARI between runs ≥ 0.60 | 0.51 | 0.85 | 0.38 | no |
| C3 BCubed vs CSV ≥ v1 − 0.05 | 0.68 (≥ 0.62) | 0.53 (≥ 0.62) | 0.41 (≥ 0.39) | no |
| C4 validity | 3/3 | 3/3 | 3/3 | yes |

## v1: one call proposes the whole tree (branch `agent1-poc`)

### E1: first prompt
- **Date / code:** 23 Sep; prompt `poc-1`; code committed afterwards in `a5bee99`.
- **Config:** DataEng, all 57 LOs; seed 0; model lists module members separately.
- **Tested:** whether one call can produce a valid tree.
- **Outcome:** 21 of 57 LOs placed in no module.
- **Records:** `experiments/v1/E1/`.

### E2: each LO names its module
- **Date / code:** 26 Sep; prompt `poc-2`; targeted patch repair (up to 4 rounds); code
  committed afterwards in `b6834df`.
- **Config:** six courses, ≥20-LO sample, seed 0.
- **Tested:** validity with the new output format.
- **Outcome:** all valid after repair fixes; order near random; 0–18 prerequisite links per
  course.
- **Records:** `experiments/v1/E2/`.

### E3: order from the model's prerequisites
- **Date / code:** 26 Sep; prompt `poc-3` (module order by topological sort of LO
  prerequisites); code committed in `b6834df`.
- **Config:** six courses, ≥20-LO sample, seeds 0, 1, 2 (DataEng and PPP also 3, 4, 5, run on
  27 Sep for E4).
- **Tested:** quality against the CSV and consistency between seeds.
- **Outcome:** 17/18 valid; BCubed vs CSV 0.44–0.65; between seeds, module ARI 0.24–0.43 and
  only 10–24% of prerequisite edges shared. The result depends on input order.
- **Records:** `experiments/v1/E3/`; DataEng and PPP seeds 3–5 in `experiments/v2/E4_v1ref/`.

## v2: consensus of repeated judgments (branch `agent_v2`)

### E4: consensus grouping, voted module prerequisites
- **Date / code:** 27 Sep; test design `docs/v2_preregistration.md` on `agent_v2` (committed
  before running, `bdb43e6`); code `ebac3e6`; run at `9e96fa0`.
- **Config:** DataEng, PPP; ≥20-LO sample. Two independent v2 runs per course: A = grouping from
  v1 seeds 0–2 + order asks with seeds 0–2; B = v1 seeds 3–5 + order asks 3–5. Order prompt
  `v2-order-1`. Reference: v1 seeds 0–5.
- **Tested:** consistency between runs (targets: order agreement ≥ 0.90, grouping ARI ≥ 0.60).
- **Outcome:** order agreement DataEng 0.54 → 0.77, PPP 0.77 → 0.82; grouping ARI 0.60 / 0.30;
  PPP BCubed fell 0.06. Targets not met. Cost 7–9× v1.
- **Records:** `experiments/v2/E4/` (runs A = `_s0`, B = `_s3`); v1 reference
  `experiments/v2/E4_v1ref/` (seeds 3–5) and `experiments/v1/E3/` (seeds 0–2).

## v3: order first, then split (branch `agent_v3`)

### E5: pilot
- **Date / code:** 27–28 Sep; code `1bac2e2`; prompt `v3-1` (one ask returns order and modules;
  module boundaries from those answers).
- **Config:** ≥20-LO sample; two runs per course (seeds 0–4, 5–9); stopped after 5 courses.
- **Tested:** the v3 pipeline end to end.
- **Outcome:** 3 of 50 answers dropped an LO; the prompt now requires every LO (`v3-2`).
- **Records:** `experiments/v3/E5/` (and first-1 and first-3-ask sub-results in `E5_k1/`,
  `E5_k3/`).

### E6: order stage
- **Date / code:** 28 Sep; run at `272ed07`; prompt `v3-2`; 5 order asks per run; consensus
  order by mean position; modules from the order answers (majority of neighbours split).
- **Config:** 3-module sample of DataEng, CloudAdmin, CloudNative. Three v3 runs per course
  (seeds 0–4, 5–9, 10–14). Reference: v1 (`poc-3`) seeds 0–2 on the same samples. Sub-results
  from the first 1 and 3 asks of each run.
- **Tested:** consistency of the order and modules; effect of the number of asks.
- **Outcome:** sequence agreement between runs 0.92 / 0.93 / 0.91 (v1: 0.59 / 0.76 / 0.80);
  0.87 with 1 ask vs 0.92 with 5. Modules over-split (CloudNative 15 vs 3 in the CSV); module ARI
  0.48–0.52; BCubed 0.59 / 0.57 / 0.25 (v1: 0.67 / 0.67 / 0.44). All 45 asks valid.
- **Records:** `experiments/v3/E6/`, v1 reference `E6_v1ref/`, sub-results `E6_k1/`, `E6_k3/`.

### E7: separate split step
- **Date / code:** 28 Sep; code `e140986`; split prompt `v3-seg-1`; 5 split asks per run on the
  E6 consensus order, codes shuffled per ask, same seeds as the run; a boundary where a strict
  majority of asks place one.
- **Config:** the 9 E6 runs.
- **Tested:** whether splitting on the fixed order fixes module consistency and quality.
- **Outcome:** over-splitting fixed (4–5 modules). Module ARI 0.51 / 0.85 / 0.38; BCubed 0.68 /
  0.53 / 0.41. 13 of 45 first answers changed the given order (12 repaired). The model chooses
  4–10 modules for the same sequence.
- **Records:** `experiments/v3/E7/`.

### E8: near-miss tolerance (no new calls)
- **Date / code:** 28 Sep; code `7f1fbb4`; E7's saved answers re-combined, counting boundaries
  one position apart as the same boundary.
- **Tested:** whether split answers disagree only by one position.
- **Outcome:** no effect (module ARI 0.51 / 0.81 / 0.39; BCubed within 0.01). Single split
  answers on the same sequence agree at ARI 0.59–0.68. Not adopted.
- **Records:** `experiments/v3/E8/`.

### Exploratory (not recorded as runs)
- **Median module count** on E6's saved answers: cuts at the strongest split votes, number of
  modules = median over answers. Fixed E6's over-splitting (CloudNative BCubed 0.25 → 0.50), not
  module consistency (ARI about 0.5). Superseded by E7.

## Not reported

Runs that only tested provider limits (`qwen3.8-27b`; medium reasoning effort) are in
`experiments/v1/limits_qwen/` and `limits_medium/`. They say nothing about the method.

## Corrections

- **CloudNative v1 reference (E6), 28 Sep.** Runs used to be named by start time. The seed-1 and
  seed-2 runs started in the same second, got the same folder, and seed 2 overwrote seed 1; the
  log listed both seeds for that one folder. Seed 2 was therefore counted twice, and one
  "between runs" pair compared a run with itself. Seed 1 was reproduced from the response cache
  (all calls cached, same code on `agent1-poc`). Corrected CloudNative v1 values: BCubed 0.44
  (was 0.52), sequence agreement between runs 0.80 (was 0.90), module ARI 0.43 (was 0.57). The C3
  target for CloudNative is 0.39 (was 0.47). Runs are now named
  `<version>/<experiment>/<course>_s<seed>`, so this cannot recur.
