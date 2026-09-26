# Agent 1 (Sequencer): from proof of concept to v2

TEEL Lab, ACE-AI. 26 September 2026.

- Plan v1: [implementation_plan.md on `agent1-poc`](https://github.com/rabhyank-hash/AceAI-v1/blob/agent1-poc/docs/implementation_plan.md).
  Plan v2: [implementation_plan.md](implementation_plan.md) on `agent_v2`; the difference is
  `git diff agent1-poc agent_v2 -- docs/implementation_plan.md`.
- v2 pre-registration: [v2_preregistration.md](v2_preregistration.md), committed before any v2 run.
- Evaluation framework: [evaluation.md](evaluation.md).
- Experiment records (configs, scores, structures; no LO text): `experiments/v1/`.

## Summary

1. The v1 plan is implemented end to end for Agent 1 and runs on a sample of all six courses:
   every LO placed, every structural check passed, order computed from the model's own
   prerequisites.
2. It fails on consistency. The same LOs in a different input order give a different course:
   only 10–24% of prerequisite edges and little of the grouping repeat between runs. A plan that
   changes with input order cannot be trusted as a standalone tool, and cannot be evaluated,
   because one run's score is mostly noise.
3. Plan v2 makes consistency a design property: every LLM judgment is small, repeated under
   different presentation orders, and aggregated by code (consensus grouping, voted module
   prerequisites, fixed tie-breaking rules). Bloom level is removed as an ordering signal.
4. A small pre-registered prototype of v2 (2 courses, 2 independent runs each) is implemented and
   tested offline. **It has not run yet**: the free API tier's daily token budget was used up by
   the v1 experiments. It runs next with one command (`scripts/run_v2_experiment.sh`).

## 1. The initial plan and its implementation

### 1.1 Agent 1 in plan v1

Agent 1 turns a shuffled, mixed-granularity LO list into a deduplicated, dependency-ordered tree
of modules (plan v1 §4): normalize each LO (verb, Bloom level, track, target concept, scope) →
detect containment under course-level LOs → deduplicate with provenance → infer LO prerequisites
→ group into modules, ordered internally by Bloom level → order modules by topological sort,
the agent choosing among valid orders. The LLM makes judgments; deterministic code tools do
every mechanical check.

### 1.2 Implementation

| Component | What it does | Code |
|---|---|---|
| Ingestion | Loads the six course CSVs and PPP's syllabus LOs into typed records (BOM, CRLF, quoted commas handled; nothing dropped; quirks reported in a data profile). | `src/aceai/ingest/loader.py`, `profile.py` |
| Ground truth | The authors' unit → module → LO structure per course, in file order. | `ingest/ground_truth.py` |
| Agent 1 input | LO texts only, under opaque ids (hash of the raw id), shuffled with a seed; leaked `(LOn)` markers stripped and logged. | `ingest/agent1_input.py` |
| Schemas | `LearningObjective`, `Module`, `SequencerOutput` (with provenance), Bloom C1–C6. | `schemas.py` |
| Code tools | `validate_output`, `check_provenance`, `check_cycles`, `check_module_order`, `topo_sort_modules`, `build_module_graph`: structured errors the model can act on. | `tools/` |
| LLM client | Any OpenAI-compatible provider by configuration; JSON mode; retry honoring `retry-after`; response cache; per-minute budget check; call log. | `llm/client.py` |
| Agent 1 (v1) | One LLM call proposes normalization, merges, modules (each LO names its module) and LO prerequisites. Code derives module dependencies and the order by topological sort (plan step 6); the model's listing breaks ties. The six tools check the result; errors go back as targeted patch rounds (up to 4). | `agents/sequencer.py` (prompt `poc-3`) |
| Evaluation | Coverage; BCubed P/R/F1 and ARI against CSV modules and units; order agreement with the CSV; run-to-run consistency; module-graph drawing. | `eval/compare.py`, `eval/draw.py`, `scripts/consistency.py` |

Tests: 140 unit tests (the LLM is replaced by a scripted fake; no test calls an API), ruff clean.

### 1.3 Experiments on v1

All runs: `openai/gpt-oss-120b` on Groq (free tier), reasoning effort low, temperature 0. Sample
per course: the first 3 CSV modules, extended until at least 20 LOs (21–44 LOs).

| # | Date | What | Result |
|---|---|---|---|
| E1 | 23 Sep | DataEng, all 57 LOs, first prompt (model lists module members separately) | 21 of 57 LOs left out of every module; repair round too large for the 8K tokens/min limit |
| E2 | 26 Sep | Six courses, prompt `poc-2` (each LO names its module; model orders modules) | All valid after fixes to repair; order near random; 0–18 prerequisite links per course |
| E3 | 26 Sep | Six courses × seeds 0, 1, 2, prompt `poc-3` (order from the model's prerequisites) | 17 of 18 runs valid; results below |
| E4 | 26 Sep | `qwen3.8-27b`, seed 0 | 3 of 6 courses ran (free-tier output limit); all 3 valid on the first attempt |
| E5 | 26 Sep | gpt-oss-120b, medium reasoning effort | No output: reasoning used the whole output allowance within 8K tokens/min |

E3, quality against the authors' structure (mean and range over 3 seeds):

| Course | LOs | Valid | BCubed F1 (CSV modules) | Order agreement with CSV | Prerequisite links |
|---|---:|---:|---|---|---|
| AI_Practitioner | 23 | 3/3 | 0.54 (0.46–0.60) | 0.55 (0.49–0.62) | 17 (12–23) |
| CloudAdmin | 21 | 3/3 | 0.65 (0.55–0.80) | 0.56 (0.18–0.84) | 19 (19–20) |
| CloudDevOps | 23 | 3/3 | 0.47 (0.43–0.51) | 0.59 (0.40–0.83) | 20 (17–21) |
| CloudNative | 44 | 2/3 | 0.44 (0.36–0.52) | 0.46 (0.43–0.50) | 21 (18–24) |
| DataEng | 23 | 3/3 | 0.63 (0.61–0.67) | 0.41 (0.34–0.46) | 16 (10–20) |
| PPP | 28 | 3/3 | 0.59 (0.56–0.64) | 0.44 (0.42–0.47) | 21 (9–29) |

E3, run-to-run consistency (3 seed pairs per course):

| Course | Grouping ARI between runs | Order agreement between runs | Prerequisite edges shared (Jaccard) |
|---|---|---|---|
| AI_Practitioner | 0.25 (0.04–0.64) | 0.61 (0.48–0.79) | 0.21 |
| CloudAdmin | 0.26 (0.24–0.28) | 0.49 (0.33–0.73) | 0.15 |
| CloudDevOps | 0.34 (0.30–0.38) | 0.66 (0.55–0.80) | 0.16 |
| CloudNative | 0.43 (0.34–0.61) | 0.87 (0.79–0.97) | 0.10 |
| DataEng | 0.24 (0.12–0.40) | 0.63 (0.60–0.67) | 0.24 |
| PPP | 0.43 (0.37–0.52) | 0.81 (0.70–0.87) | 0.16 |

Order agreement: share of LO pairs in different modules that are ordered the same way; 0.5 is
random. ARI: 1 = identical groupings, 0 = chance.

## 2. Where v1 fails, and why it matters

1. **Inconsistent across input orders (the main failure).** Between two runs that differ only in
   input order, 76–90% of prerequisite edges change and grouping agreement is low (ARI
   0.24–0.43). Order agreement with the authors swings from 0.18 to 0.84 on the same course
   (CloudAdmin). *Why it matters:* the tool has to stand alone, so its output must not depend on
   the order LOs happen to arrive in. And no order can be judged good or bad when rerunning gives
   a different one; single-run scores are noise.
2. **Order agreement cannot separate errors from valid alternatives.** On DataEng the authors teach
   tools first (pandas primer) and the model teaches concepts first; both are defensible. The
   model also placed overview LOs ("role of a data engineer", "data science life cycle") 8th of
   10, which is an error. Against the CSV order both look the same. *Why it matters:* improvement
   cannot be measured without labeled module order.
3. **One large judgment per call.** Normalization, merges, grouping and about 20 prerequisites
   are decided together in one JSON reply at the lowest reasoning effort; an early difference
   cascades. This is where v1's proof of concept departed from the plan's chain, and where the
   instability comes from.
4. **Bloom level used as an ordering signal (a plan error).** Plan v1 orders LOs within a module
   by increasing Bloom level. Bloom level classifies the kind of thinking an LO asks for, not when
   it can be learned; a higher-level LO can come before a lower one.
5. **A detailed LO made a parent disappears from the order.** In one run the model made a detailed
   CloudNative LO the parent of three others; parents are not placed in modules, so it was never
   taught. The plan does not say whether this is allowed.
6. **Structural failures found and fixed along the way:** LOs left out of every module (fixed by
   having each LO name its module); repair patches wiping the module list (modules now patched by
   id); tool errors naming list positions instead of LO ids; a self-referencing merge that made
   the model delete an LO. These are engineering fixes, not design changes.
7. **The free API tier limits what can be tested:** 8K tokens/minute and 200K tokens/day for
   gpt-oss-120b; medium reasoning effort, whole courses and repeated calls do not fit.

## 3. Changes to the plan (v2)

| # | Change | Addresses |
|---|---|---|
| 1 | **Consistency by construction** as an Agent 1 principle: every LLM judgment is small, asked several times under different presentation orders, and aggregated by deterministic code. | 1, 3 |
| 2 | **Grouping by consensus:** k grouping runs on different input orders; code clusters how often each pair of LOs lands together (average linkage, threshold 0.5). | 1 |
| 3 | **Prerequisites at module level:** per target module, the model names the modules required before it (and ones better taught before it), under shuffled module order, labels and LO order; an edge is kept on a strict majority. | 1, 3 |
| 4 | **Order by code with fixed tie rules:** topological sort; cycles broken at the weakest edge; ties by aggregated preference, then share of conceptual LOs, then a stable key. | 1 |
| 5 | **Bloom level is not an ordering signal;** it stays Agent 2's depth scale. The Bloom-order warning is removed from `check_module_order`. | 4 |
| 6 | **Evaluation:** run-to-run consistency is Agent 1's primary criterion; quality is then measured against labeled module-order pairs (violations) and the authors' grouping. | 1, 2 |
| 7 | **Test courses:** the six SAIL courses (plan v1 named PPP and a Udacity nanodegree). | — |
| 8 | **Open questions added:** may a detailed LO be a parent; is "more conceptual first" the right second tie rule. | 5 |

## 4. Prototype test of plan v2

**Status: implemented and tested offline; waiting for API budget.** Everything below was fixed
before running, in [v2_preregistration.md](v2_preregistration.md) (commit `bdb43e6`), and the code
was committed before running (`ebac3e6`).

- **Hypotheses.** H1 (primary): two independent v2 runs agree with each other; H1a order
  agreement between runs ≥ 0.90, H1b grouping ARI between runs ≥ 0.60. H2: BCubed F1 against the
  CSV not below the v1 mean minus 0.05. H3: all tools pass, every LO placed.
- **Courses:** DataEng and PPP (lowest v1 order agreement; several CSV units).
- **Same model and settings as v1**, so the design is the only change.
- **Two independent v2 runs per course** that share no LLM call: v2-A uses the v1 grouping runs
  with seeds 0–2 and ordering asks under permutation seeds 0–2; v2-B uses v1 seeds 3–5 and
  permutation seeds 3–5. The v1 reference becomes seeds 0–5 (15 seed pairs).
- **Prototype scope:** consensus grouping reuses v1's grouping calls; deduplication and
  containment are left out (neither sample has syllabus LOs).
- **Offline check (no API calls):** consensus of the seed 0–2 runs gives 10 modules for DataEng
  and 12 for PPP, many with 1–2 LOs, because few LO pairs are grouped together in at least 2 of
  the 3 v1 runs. Estimated cost: about 185K tokens (v1 seeds 3–5 plus 30–36 ordering asks per v2
  run).

## 5. Changes in the metrics

Pending the run. The table will be filled from `runs/v2_tables.md`; v1 values are E3 (seeds
0–2) and will be recomputed over seeds 0–5.

| Course | Metric | v1 (seed pairs) | v2 (A vs B) | Criterion |
|---|---|---|---|---|
| DataEng | Order agreement between runs | 0.63 (0.60–0.67) | pending | ≥ 0.90 |
| DataEng | Grouping ARI between runs | 0.24 (0.12–0.40) | pending | ≥ 0.60 |
| DataEng | BCubed F1 vs CSV | 0.63 | pending | ≥ v1 − 0.05 |
| DataEng | Valid runs | 3/3 | pending | all |
| PPP | Order agreement between runs | 0.81 (0.70–0.87) | pending | ≥ 0.90 |
| PPP | Grouping ARI between runs | 0.43 (0.37–0.52) | pending | ≥ 0.60 |
| PPP | BCubed F1 vs CSV | 0.59 | pending | ≥ v1 − 0.05 |
| PPP | Valid runs | 3/3 | pending | all |

## Requirements

- **API access beyond the free tier.** Groq's free tier (8K tokens/minute, 200K tokens/day for
  `gpt-oss-120b`; 1,000 output tokens/minute for `qwen3.8-27b`) allows about 25 v1 sample runs
  or one v2 prototype experiment per day, and no medium or high reasoning effort. v2 makes many
  small calls by design. Measured v1 cost: about 280 tokens per LO per run (60% prompt, 40%
  completion). At list prices on 26 September 2026 (Groq `gpt-oss-120b` $0.15 input / $0.60
  output per 1M tokens; DeepSeek `deepseek-v4-pro` $1.32 / $3.96 at peak), the planned work
  (about 18M tokens: model and effort comparison, whole courses, pipeline development) costs
  $6–$43 depending on the model. Groq's pay-as-you-go Developer plan raises the limit to 250K
  tokens/minute.
- **Labels.** Module-order pairs for the six samples (68 pairs, about 15 minutes;
  `scripts/make_label_sheets.py` writes the sheets to `annotations/`, which is git-ignored
  because the sheets contain LO text). Needed to measure order quality (§2 item 2).
- **Decisions.** Merge rule and Bloom verb rules (plan open questions 1–2); whether a detailed LO
  may be a parent (open question 11); the second tie rule (open question 12).

## Deviations and exclusions

- One v1 run (DataEng, seed 3) failed before any reply with a tokens-per-day rate limit. It is
  excluded (`runs/excluded.tsv`) and will be rerun with the same seed.
- The consensus threshold "0.5" is implemented as strictly greater than 0.5 (a strict majority,
  as for prerequisite votes). This was fixed in the code committed before running.

## Reproduce

```bash
bash scripts/run_v2_experiment.sh                 # the pre-registered v2 experiment, end to end
python scripts/run_poc.py --course DataEng --modules 3 --max-repairs 4 --seed 0   # one v1 run
python scripts/analyze_experiments.py             # tables from runs/experiments.tsv
python scripts/analyze_experiments.py --log experiments/v1/experiments.tsv       # from git
```

Sources for prices and limits: https://console.groq.com/docs/models,
https://console.groq.com/docs/rate-limits, https://api-docs.deepseek.com/quick_start/pricing.
