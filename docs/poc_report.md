# Agent 1 (Sequencer): v1 results, plans v2 and v3

TEEL Lab, ACE-AI, 28 September 2026.

Plan v1 is on branch `agent1-poc`, plan v2 on `agent_v2`, plan v3 on `agent_v3`
(`docs/implementation_plan.md` on each). Pre-registrations: [v2](v2_preregistration.md),
[v3](v3_preregistration.md). Metrics: [evaluation.md](evaluation.md). Run records without LO text:
`experiments/v1/`, `v2/`, `v3_pilot/`, `v3/`.

## Summary

- v1 produces a valid module tree for a sample of every course.
- v1 is not consistent: rerunning with the LOs in a different order gives a different course.
- Plan v2 repeats small LLM judgments under different orders and aggregates them in code.
- The v2 prototype ran on 27 September. It did not meet its pre-registered consistency targets.
  It made DataEng more consistent (order agreement between runs 0.54 → 0.77) but not PPP.
  Grouping is the remaining source of variation.
- Plan v3 limits Agent 1 to ordering the LOs and splitting the sequence into modules. Each ask
  returns the whole course on shuffled input; code aggregates 5 asks.
- v3 makes the order consistent: sequence agreement between runs is 0.91–0.93 on all three test
  courses (v1: 0.59–0.90). The module split is not consistent (ARI 0.48–0.52) and is further
  from the authors' modules than v1.

## 1. Plan v1: implementation and tests

### Agent 1 in plan v1

1. Normalize each LO: verb, Bloom level, track, target concept, scope.
2. Place detailed LOs under course-level LOs.
3. Merge duplicates, keeping provenance.
4. Infer prerequisites between LOs.
5. Group LOs into modules, ordered internally by Bloom level.
6. Order modules by topological sort; the agent picks among valid orders.

The LLM makes judgments. Deterministic code checks structure.

### Implementation

| Component | Function | Code |
|---|---|---|
| Ingestion | Loads six course CSVs and PPP's syllabus LOs. Drops nothing; reports data quirks. | `ingest/loader.py`, `profile.py` |
| Ground truth | Authors' unit → module → LO structure, in file order. | `ingest/ground_truth.py` |
| Agent input | LO texts under opaque ids, shuffled by seed. Order-revealing `(LOn)` markers removed. | `ingest/agent1_input.py` |
| Schemas | LearningObjective, Module, SequencerOutput with provenance; Bloom C1–C6. | `schemas.py` |
| Tools | Validation, provenance, cycles, module order, topological sort, module graph. Errors name the LO ids involved. | `tools/` |
| LLM client | OpenAI-compatible providers by config; JSON mode; rate-limit retry; response cache. | `llm/client.py` |
| Agent 1 v1 | One call returns every LO's normalization, merges, module and prerequisites. Code sets the order by topological sort. Tool errors go back to the model as patch rounds, up to 4. | `agents/sequencer.py` |
| Evaluation | Coverage; BCubed F1 and ARI against CSV modules; order agreement with the CSV; run-to-run consistency; module-graph drawing. | `eval/`, `scripts/consistency.py` |

140 unit tests pass. No test calls an API.

### Experiments

Model: `openai/gpt-oss-120b` on Groq's free tier, low reasoning effort, temperature 0.
Sample per course: the first 3 CSV modules, extended to at least 20 LOs (21–44 LOs).

| # | Date | Setup | Result |
|---|---|---|---|
| E1 | 23 Sep | DataEng, all 57 LOs, first prompt | 21 LOs placed in no module |
| E2 | 26 Sep | Six courses; each LO names its module | All valid; order near random |
| E3 | 26 Sep | Six courses × 3 seeds; order from the model's prerequisites | 17 of 18 runs valid |
| E4 | 26 Sep | `qwen3.8-27b`, seed 0 | 3 of 6 courses ran (rate limit); all 3 valid |
| E5 | 26 Sep | gpt-oss-120b, medium reasoning effort | Empty replies: reasoning used the output limit |

E3, agreement with the authors' structure (mean and range over 3 seeds):

| Course | LOs | Valid | BCubed F1 | Order agreement | Prerequisite links |
|---|---:|---:|---|---|---|
| AI_Practitioner | 23 | 3/3 | 0.54 (0.46–0.60) | 0.55 (0.49–0.62) | 17 |
| CloudAdmin | 21 | 3/3 | 0.65 (0.55–0.80) | 0.56 (0.18–0.84) | 19 |
| CloudDevOps | 23 | 3/3 | 0.47 (0.43–0.51) | 0.59 (0.40–0.83) | 20 |
| CloudNative | 44 | 2/3 | 0.44 (0.36–0.52) | 0.46 (0.43–0.50) | 21 |
| DataEng | 23 | 3/3 | 0.63 (0.61–0.67) | 0.41 (0.34–0.46) | 16 |
| PPP | 28 | 3/3 | 0.59 (0.56–0.64) | 0.44 (0.42–0.47) | 21 |

E3, agreement between runs of the same course (3 seed pairs):

| Course | Grouping ARI | Order agreement | Shared prerequisite edges |
|---|---|---|---|
| AI_Practitioner | 0.25 (0.04–0.64) | 0.61 (0.48–0.79) | 21% |
| CloudAdmin | 0.26 (0.24–0.28) | 0.49 (0.33–0.73) | 15% |
| CloudDevOps | 0.34 (0.30–0.38) | 0.66 (0.55–0.80) | 16% |
| CloudNative | 0.43 (0.34–0.61) | 0.87 (0.79–0.97) | 10% |
| DataEng | 0.24 (0.12–0.40) | 0.63 (0.60–0.67) | 24% |
| PPP | 0.43 (0.37–0.52) | 0.81 (0.70–0.87) | 16% |

Order agreement is the share of LO pairs, in different modules, ordered the same way. Random
order scores 0.5. ARI is 1 for identical groupings and 0 for chance.

## 2. Failures and their impact

1. **Results depend on input order.** Two runs that differ only in LO order share 10–24% of
   prerequisite edges. On CloudAdmin, order agreement with the authors ranges from 0.18 to 0.84.
   A standalone tool must give the same course for the same LOs. A single run's score cannot
   be trusted.
2. **Order agreement mixes errors with valid alternatives.** DataEng's authors teach tools
   before concepts; the model teaches concepts first. Both are valid. The model also places
   overview LOs ("role of a data engineer") 8th of 10, which is an error. The score treats both
   the same. Labeled module order is needed to tell them apart.
3. **One call makes every decision.** Normalization, merges, grouping and about 20 prerequisites
   come from one reply. An early difference changes everything after it.
4. **Bloom level is used as an order.** Plan v1 orders LOs by Bloom level. Bloom level describes
   the kind of thinking, not the learning sequence.
5. **A detailed LO can disappear from the order.** One run made a detailed CloudNative LO the
   parent of three others. Parents are not placed in modules, so it was never taught.
6. **Engineering faults, fixed.** Unplaced LOs, repair patches that deleted modules, error
   messages that named list positions, and a self-referencing merge. None changes the plan.
7. **The free API tier blocks larger tests.** Its limits are 8K tokens/minute and 200K
   tokens/day. Higher reasoning effort, whole courses and repeated calls do not fit.

## 3. Plan changes (v2)

| # | Change | Fixes |
|---|---|---|
| 1 | Each LLM judgment is small, repeated under different presentation orders, and aggregated in code. | 1, 3 |
| 2 | Grouping by consensus: several grouping runs; LOs grouped together in most runs form a module. | 1 |
| 3 | Prerequisites between modules, not LOs. For each module, the model names the modules that must come first. An edge is kept if most asks name it. | 1, 3 |
| 4 | Code orders modules by topological sort. Cycles are broken at the weakest edge. Ties are broken by the model's aggregated preference, then the share of conceptual LOs, then a fixed key. | 1 |
| 5 | Bloom level is not used for ordering. It remains the depth scale for Agent 2. | 4 |
| 6 | Consistency between runs is the first criterion for Agent 1, then labeled order and grouping. | 1, 2 |
| 7 | Test courses: the six SAIL courses, not PPP and Udacity. | — |
| 8 | New open questions: may a detailed LO be a parent; is "conceptual first" the right second tie rule. | 5 |

`git diff agent1-poc agent_v2 -- docs/implementation_plan.md` shows every change to the plan.

## 4. v2 prototype

Run on 27 September 2026 at commit `9e96fa0`. The design was committed before any run
(`bdb43e6`), and the code before running (`ebac3e6`). Records: `experiments/v2/`.

- **Hypotheses.** H1a: order agreement between two v2 runs ≥ 0.90. H1b: grouping ARI between
  them ≥ 0.60. H2: BCubed F1 against the CSV at least the v1 mean minus 0.05. H3: every run
  valid.
- **Courses.** DataEng and PPP: the lowest v1 order agreement, and several CSV units each.
- **Settings.** Same model and settings as v1. Only the method changes.
- **Runs.** Two v2 runs per course with no LLM call in common. Run A uses v1 seeds 0–2 for
  grouping and order seeds 0–2. Run B uses v1 seeds 3–5 and order seeds 3–5. The v1 reference
  is seeds 0–5.
- **Scope.** Grouping reuses v1's calls. Merging and containment are left out; neither sample
  has course-level LOs.
- **Execution.** All 4 v2 runs are valid. Consensus gave 10 modules in both DataEng runs, and 12
  and 15 modules in the PPP runs, many with 1–2 LOs. Each v2 run made 30–45 ordering calls and
  used 25–48K tokens, on top of its 3 grouping runs. A v2 run costs 7–9 times a v1 run.

## 5. Metrics

v1: 6 seeds per course (15 run pairs). v2: one pair of independent runs per course.

| Course | Metric | v1 mean (range) | v2 | Target | Met |
|---|---|---|---|---|---|
| DataEng | Order agreement between runs | 0.54 (0.22–0.77) | 0.77 | ≥ 0.90 | no |
| DataEng | Grouping ARI between runs | 0.37 (0.12–0.62) | 0.60 | ≥ 0.60 | yes |
| DataEng | BCubed F1 vs CSV | 0.639 | 0.593 | ≥ 0.589 | yes |
| DataEng | Valid runs | 6/6 | 2/2 | all | yes |
| PPP | Order agreement between runs | 0.77 (0.66–0.97) | 0.82 | ≥ 0.90 | no |
| PPP | Grouping ARI between runs | 0.35 (0.22–0.58) | 0.30 | ≥ 0.60 | no |
| PPP | BCubed F1 vs CSV | 0.562 | 0.501 | ≥ 0.512 | no |
| PPP | Valid runs | 5/6 | 2/2 | all | yes |

A hypothesis holds only if it holds for both courses (pre-registration).

- **H1a (order consistency): not met.** DataEng rose from 0.54 to 0.77; 1 of 15 v1 pairs reached
  0.77. PPP rose from 0.77 to 0.82; 4 of 15 v1 pairs scored higher.
- **H1b (grouping consistency): not met.** DataEng reached 0.60. PPP fell to 0.30, below v1.
- **H2 (quality): not met.** DataEng stayed within 0.05 of v1. PPP fell 0.06.
- **H3 (validity): met.**

### Exploratory observations (not pre-registered)

- Grouping is the main remaining source of variation. Consensus of 3 unstable groupings is itself
  unstable: the two PPP runs produced 12 and 15 modules. Order agreement between runs is
  computed over LOs, so grouping differences lower it too.
- Many consensus modules hold 1–2 LOs, because few LO pairs are grouped together in at least 2 of
  3 v1 runs.
- Within-module LO prerequisites overlap more between v2 runs (44–62% shared) than between v1 runs
  (16–24%). They are majority votes over grouping runs, so this is expected, and the numbers are
  not directly comparable.

### Implications for the plan

1. Consensus over 3 noisy groupings is not enough. Grouping needs its own stable design, for
   example: agree on the module topics first by consensus, then assign each LO to one topic.
2. The ordering stage should be measured on its own: repeat v2's ordering with a fixed grouping
   and different order seeds. This isolates how much variation the ordering questions add.
3. Order quality still needs labeled module order; consistency alone does not show that an order
   is good.

## 6. Plan v3 and its test

### Changes (plan v3)

| # | Change | Reason |
|---|---|---|
| 1 | Agent 1 only orders the LOs and splits the sequence into modules. | Both have ground truth; v2's separate grouping was the least stable step. |
| 2 | Each ask returns the whole ordered, segmented course on shuffled, relabelled input; 5 asks per run. | Permutation self-consistency (Tang et al., NAACL 2024). |
| 3 | Consensus order by mean position; a module boundary where most asks split two neighbours. | Aggregation in code. |
| 4 | Only LOs with identical text are merged, by code. | No LLM judgment on duplicates. |
| 5 | Normalization (Bloom level etc.) moves to Agent 2. | Agent 1 does not use it. |
| 6 | The prompt states that every LO must be included (prompt v3-2). | In a pilot with prompt v3-1, 3 of 50 answers dropped an LO. |

### Test

Pre-registered (`d1637f3`), amended before any v3-2 run (`dd99ae2`), run at `272ed07` on 28
September. Courses: DataEng, CloudAdmin, CloudNative, each limited to its first 3 CSV modules
(11, 15 and 44 LOs; CloudNative has 16 exact duplicates). Reference: v1 on the same samples,
seeds 0–2. v3: three independent runs per course (5 asks each). Same model and settings as v1.
Records: `experiments/v3/`.

### Results

Mean over 3 run pairs (range). v1 on the same samples in brackets.

| Course | Sequence agreement between runs | Module ARI between runs | BCubed F1 vs CSV | Sequence agreement vs CSV | Valid runs |
|---|---|---|---|---|---|
| DataEng | 0.92 (0.89–0.93) [0.59] | 0.52 [0.87] | 0.59 [0.67] | 0.42 [0.52] | 3/3 [3/3] |
| CloudAdmin | 0.93 (0.92–0.94) [0.76] | 0.48 [0.24] | 0.57 [0.67] | 0.91 [0.82] | 3/3 [3/3] |
| CloudNative | 0.91 (0.89–0.93) [0.90] | 0.48 [0.57] | 0.25 [0.52] | 0.65 [0.61] | 3/3 [1/3] |

| Hypothesis | Target | Result |
|---|---|---|
| H1a sequence consistency | ≥ 0.90 on each course | **Met** (0.91–0.93) |
| H1b module consistency | ARI ≥ 0.60 on each course | Not met (0.48–0.52) |
| H2 quality | BCubed ≥ v1 − 0.05 | Not met on any course |
| H3 validity | all runs valid | **Met** (45 of 45 asks valid; 3 repaired) |
| H4 aggregation | sequence agreement higher at k = 5 than k = 1 | **Met** (0.92 vs 0.87) |

Cost: 3.5–8.5K tokens per v3 run, below v1 on the same samples (4.0–12.5K).

### Findings

1. **The order is consistent.** Every course passes 0.90. Repeating the ask on shuffled input and
   averaging raises agreement from 0.87 (one ask) to 0.92 (five).
2. **The module split is not, and over-splits.** Module counts rise with the number of asks
   (CloudNative: 4 modules at k = 1, 15 at k = 5; the CSV has 3). Averaging positions places LOs
   between topics, and the majority rule then cuts too often. This was named as a threat in the
   pre-registration.
3. **A consistent order exposes a clear disagreement with the authors.** On DataEng, every v3 run
   teaches the Data Landscape unit before the pandas unit; the authors do the reverse. v1 varied
   between both. Whether this is an error needs a label.
4. **Exploratory (not pre-registered), on the saved asks:** setting the number of modules to the
   median across asks and cutting at the strongest split votes fixes the over-splitting
   (CloudNative BCubed 0.25 → 0.50) but not module consistency (ARI about 0.5).

### Next

1. **Decide modules on the fixed consensus order.** The order is stable, so a separate
   segmentation step on it would inherit that stability. Pre-register and test.
2. **Label the order.** 9 module-order pairs for the three test courses (sheets in
   `annotations/`). This decides whether findings like DataEng's concept-first order are errors.

## Requirements

- **API access above the free tier.** The free tier allows one v2 experiment or about 25 v1
  runs per day, and no higher reasoning effort. v1 used about 280 tokens per LO per run. The
  planned work is about 18M tokens: $6 on Groq `gpt-oss-120b`, up to $43 on DeepSeek
  `deepseek-v4-pro`, at 26 September 2026 list prices. Groq's pay-as-you-go plan allows 250K
  tokens/minute.
- **Labels.** 9 module-order pairs for the three v3 test courses (first 3 modules), a few
  minutes; 68 pairs for the larger v1/v2 samples.
  `scripts/make_label_sheets.py` writes the sheets to `annotations/`, which is not in git
  because the sheets contain LO text.
- **Decisions.** Merge rule and Bloom verb rules (plan questions 1–2). Whether a detailed LO may
  be a parent (question 11). The second tie rule (question 12).

## Deviations

- One v1 run (DataEng, seed 3) hit the daily rate limit before any reply on 26 September. It
  is excluded (`runs/excluded.tsv`) and was rerun with the same seed.
- The consensus threshold of 0.5 is applied as strictly greater than 0.5. This was set in code
  before running.
- The run script's check that grouping runs share settings rejected v1 seed-0 runs, whose
  config predates the `model_params` field. The check was changed to compare the request
  parameters actually sent (commit `9e96fa0`); the parameters were identical. No results
  existed at that point.
- The PPP v1 seed-3 run ended with 5 tool errors but placed every LO. It was used as a grouping
  input for v2 run B, as the pre-registration requires all three seeds. It is counted as invalid
  in the v1 quality mean.
- v1 BCubed means are over valid runs only.
- v3: the prompt was changed from v3-1 to v3-2 after 10 runs, and the test was reduced to three
  courses with their first 3 modules. Both were recorded as Amendment 1 to the v3
  pre-registration before any v3-2 run. The v3-1 runs are a pilot (`experiments/v3_pilot/`).
- v3: in the pilot, 5 asks were lost to the daily rate limit. From Amendment 1 on, calls wait
  out rate limits.
- v3: CloudNative's v1 reference has 1 valid run of 3, so its v1 BCubed mean rests on one run.

## Reproduce

```bash
bash scripts/run_v2_experiment.sh                                 # v2 experiment
bash scripts/run_v3_experiment.sh                                 # v3 experiment (amended)
python scripts/run_poc.py --course DataEng --modules 3 --seed 0   # one v1 run
python scripts/analyze_experiments.py --log experiments/v1/experiments.tsv
python scripts/analyze_experiments.py --log experiments/v2/experiments.tsv   # v2 tables
python scripts/analyze_experiments.py --log experiments/v3/experiments.tsv   # v3 tables
```

Prices and limits: https://console.groq.com/docs/models,
https://console.groq.com/docs/rate-limits, https://api-docs.deepseek.com/quick_start/pricing.
