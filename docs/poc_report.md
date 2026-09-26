# Agent 1 proof of concept: results and next steps

TEEL Lab, ACE-AI. 26 September 2026. Code: branch `agent1-poc`. Plan:
[implementation_plan.md](implementation_plan.md). Evaluation framework:
[evaluation.md](evaluation.md). Experiment records (no LO text): `experiments/v1/`.

## Summary

- **The pipeline works end to end.** On a sample of every course (21–44 LOs), Agent 1 turns a
  shuffled LO list into a valid module tree: every LO placed, provenance intact, no cycles,
  module order derived from the model's own prerequisites. 17 of 18 runs are valid.
- **The model's judgments are unstable.** Running the same LOs in a different shuffle gives a
  different structure. Only 10–24% of prerequisite edges repeat between two runs, and grouping
  agreement between runs is low (ARI 0.24–0.43). Single-run scores are therefore mostly noise.
- **Agreement with the authors' structure is moderate for grouping and near random for order.**
  BCubed F1 against the CSV modules is 0.44–0.65. Order agreement with the CSV is 0.41–0.59,
  where 0.5 is random. Part of the order gap is legitimate design difference, which we cannot
  separate from real errors without labels.
- **The free API tier is now the bottleneck.** Groq's free tier (8K tokens/minute, 200K/day)
  forces the lowest reasoning setting. Medium reasoning produced no output at all within the
  per-minute limit, a day's allowance covers about 25 sample runs, and the comparison model
  (`qwen3.8-27b`, capped at 1,000 output tokens/minute) could run only half the courses. On the
  three it ran, it passed first time with slightly better grouping.
- **Requirements:** about 15 minutes of labeling (module-order pairs, sheets ready) and API
  access beyond the free tier (see Requirements).

## Setup

- **Sample:** the first 3 CSV modules of each course, extended until the sample has at least
  20 LOs. Syllabus LOs are excluded (they cover whole courses). `run_poc.py --modules 3`.
- **Agent 1:** one LLM call proposes normalization, merges, modules and LO prerequisites. Code
  derives module dependencies and sets the order by topological sort (plan step 6). The six
  deterministic tools check the result, and errors go back for up to 4 targeted repair rounds.
- **Model:** `openai/gpt-oss-120b` on Groq, reasoning effort low, temperature 0. Comparison
  model: `qwen/qwen3.8-27b` on Groq.
- **Seeds:** 3 input shuffles (0, 1, 2) per course.
- **Scores** (see [evaluation.md](evaluation.md)): coverage and tool pass (validity); BCubed F1
  against CSV modules (grouping); order agreement with the CSV (order); run-to-run consistency.

## Results

### Validity

| Course | LOs | Runs | Valid | Attempts (mean, range) |
|---|---:|---:|---:|---|
| AI_Practitioner | 23 | 3 | 3 | 2.7 (2–4) |
| CloudAdmin | 21 | 3 | 3 | 1.7 (1–2) |
| CloudDevOps | 23 | 3 | 3 | 2.0 |
| CloudNative | 44 | 3 | 2 | 3.0 (2–4) |
| DataEng | 23 | 3 | 3 | 1.3 (1–2) |
| PPP | 28 | 3 | 3 | 2.0 (1–3) |

The invalid run (CloudNative, seed 2) passes every tool but leaves one LO outside the module
order: the model made a detailed LO ("Define the characteristics … of cloud-native applications")
the parent of three others, and parents are not placed in modules. See decision 3 below.

### Quality against the authors' structure (mean and range over 3 seeds)

| Course | BCubed F1 (modules) | Order agreement | Prerequisite links | Modules (CSV modules) |
|---|---|---|---|---|
| AI_Practitioner | 0.54 (0.46–0.60) | 0.55 (0.49–0.62) | 17 (12–23) | 11 (7) |
| CloudAdmin | 0.65 (0.55–0.80) | 0.56 (0.18–0.84) | 19 (19–20) | 5 (5) |
| CloudDevOps | 0.47 (0.43–0.51) | 0.59 (0.40–0.83) | 20 (17–21) | 7 (3) |
| CloudNative | 0.44 (0.36–0.52) | 0.46 (0.43–0.50) | 21 (18–24) | 6 (3) |
| DataEng | 0.63 (0.61–0.67) | 0.41 (0.34–0.46) | 16 (10–20) | 8 (7) |
| PPP | 0.59 (0.56–0.64) | 0.44 (0.42–0.47) | 21 (9–29) | 10 (5) |

### Stability: the same model on different shuffles

| Course | Grouping ARI between runs | Order agreement between runs | Prerequisite edges shared (Jaccard) |
|---|---|---|---|
| AI_Practitioner | 0.25 (0.04–0.64) | 0.61 (0.48–0.79) | 0.21 |
| CloudAdmin | 0.26 (0.24–0.28) | 0.49 (0.33–0.73) | 0.15 |
| CloudDevOps | 0.34 (0.30–0.38) | 0.66 (0.55–0.80) | 0.16 |
| CloudNative | 0.43 (0.34–0.61) | 0.87 (0.79–0.97) | 0.10 |
| DataEng | 0.24 (0.12–0.40) | 0.63 (0.60–0.67) | 0.24 |
| PPP | 0.43 (0.37–0.52) | 0.81 (0.70–0.87) | 0.16 |

A model that understood the course structure would give similar answers regardless of input
order. Here, most prerequisite edges change between runs, and grouping agreement between two
runs of the same model is about as low as agreement with the authors.

### Reasoning effort and model

| Setting | Course | Valid | Attempts | BCubed F1 | Order agreement | Tokens/run |
|---|---|---|---:|---:|---:|---:|
| gpt-oss-120b, low effort (3 seeds, mean) | AI_Practitioner | 3/3 | 2.7 | 0.54 | 0.55 | 8.9K |
| | CloudAdmin | 3/3 | 1.7 | 0.65 | 0.56 | 5.0K |
| | CloudDevOps | 3/3 | 2.0 | 0.47 | 0.59 | 5.9K |
| qwen3.8-27b (seed 0) | AI_Practitioner | 1/1 | 1 | 0.61 | 0.62 | 4.4K |
| | CloudAdmin | 1/1 | 1 | 0.74 | 0.93 | 4.0K |
| | CloudDevOps | 1/1 | 1 | 0.54 | 0.23 | 4.4K |
| gpt-oss-120b, medium effort | AI_Practitioner | 0/1 | – | – | – | – |

- **Qwen** produced a valid plan on the first attempt for all three courses it could run, with
  no repairs and half the tokens, and slightly higher grouping scores. With one seed and the
  instability above, this is suggestive, not conclusive. It could not run CloudNative, DataEng
  or PPP: on the free tier `qwen3.8-27b` is limited to **1,000 output tokens per minute**, and
  these samples need about 2,700.
- **Medium reasoning effort** on gpt-oss-120b returned empty replies on every attempt. The model
  spent its whole output allowance (about 6K tokens, all the 8K/minute limit leaves) on hidden
  reasoning before writing any JSON. It stopped there; the day's 200K-token free allowance was
  also used up.

Neither comparison is possible on the free tier. Both need the credits below.

## Diagnosis

1. **Unstable reasoning is the main problem.** Order is now computed from the model's
   prerequisites, so unstable prerequisites give an unstable order. The model makes all
   judgments (Bloom level, merges, grouping, prerequisites) in one JSON reply at the lowest
   reasoning setting.
2. **Order agreement mixes real errors with valid alternatives.** DataEng is an example. The CSV
   teaches tools first (pandas primer in unit 0), and the model teaches concepts first. Both are
   defensible. But the model also placed overview LOs ("role of a data engineer", "data science
   life cycle") 8th of 10, which is a real error. Without labeled prerequisites the score
   cannot tell these apart.
3. **The free tier blocks the obvious experiments.** More reasoning, larger samples, whole
   courses, several seeds per course, and other models all exceed 8K tokens/minute or 200K
   tokens/day.

## What to improve, in order

1. **Label module order** for the six samples (68 pairs, about 15 minutes; sheets in
   `annotations/`). This turns order agreement into a violation count that does not penalize
   valid alternatives, and it tells us whether a change actually helps.
2. **Raise reasoning effort** on the same model (needs credits). This is the cheapest test of
   whether reasoning is the limit.
3. **Split the single call into the plan's chain:** normalize → containment → dedup →
   prerequisites → grouping. Prerequisites get their own call, and with it the full reasoning
   budget.
4. **Aggregate across runs:** keep prerequisite edges that most of 3–5 runs agree on. This
   directly targets the instability (roughly 3–5× the tokens).
5. **Compare models** on the labeled samples (gpt-oss-120b at several efforts, DeepSeek), with
   the same code, samples and seeds.

## Decisions needed

1. Who labels module order (68 pairs), and later unit prerequisites for whole courses?
2. Merge rule and Bloom verb rules (open questions 1 and 2). Merge and Bloom labels depend on
   them.
3. May Agent 1 make a detailed LO the parent of other detailed LOs? If yes, the parent must
   still appear in the order (for example, as the module's head). If no, only syllabus LOs may
   be parents.
4. API access (see Requirements).

## Requirements

**API access beyond the free tier.** The free tier (Groq: 8K tokens/minute and 200K tokens/day
for `gpt-oss-120b`; 1,000 output tokens/minute for `qwen3.8-27b`) allows about 25 sample runs a
day at the lowest reasoning effort. It cannot run medium or high reasoning, whole courses (60–200
LOs), several seeds and models side by side, or any design that makes many small calls. The
client already supports paid providers through configuration.

Measured usage: 18 sample runs with gpt-oss-120b (21–44 LOs, including repairs) used 137K
tokens, about 7.6K per run or about 280 tokens per LO per run; 60% prompt, 40% completion.
Higher reasoning effort increases completion tokens; we assume 2×.

| Planned work | Runs | Tokens (estimate) |
|---|---|---:|
| Effort and model comparison on the six samples: 5 settings × 6 courses × 5 seeds | 150 | 2.4M |
| Whole courses (940 LOs), best 2 settings × 5 seeds | 60 | 5.3M |
| Multi-step pipeline development: ~20 passes over all courses | 120 | 10.5M |
| **Total** | | **~18M** |

Prices listed on 26 September 2026; the blended rate assumes 60% input and 40% output tokens.

| Provider / model | Input / output per 1M tokens | Blended per 1M | 18M tokens |
|---|---|---:|---:|
| Groq `openai/gpt-oss-120b` (Developer plan: 250K tokens/min) | $0.15 / $0.60 | $0.33 | $6 |
| Groq `qwen/qwen3.8-27b` (Developer plan) | $0.80 / $4.00 | $2.08 | $37 |
| DeepSeek `deepseek-flash` (peak) | $0.30 / $1.20 | $0.66 | $12 |
| DeepSeek `deepseek-v4-pro` (peak; off-peak is half) | $1.32 / $3.96 | $2.38 | $43 |

**Labels.** Module-order pairs for the six samples (68 pairs, about 15 minutes; sheets generated
by `scripts/make_label_sheets.py`), then unit prerequisites for whole courses.

## Reproduce

```bash
python scripts/run_poc.py --course DataEng --modules 3 --max-repairs 4 --seed 0
python scripts/run_poc.py --course DataEng --modules 3 --model qwen/qwen3.8-27b
python scripts/analyze_experiments.py --log ../experiments/v1/experiments.tsv   # from scripts/
python scripts/consistency.py runs/<a> runs/<b>
python scripts/make_label_sheets.py          # annotations/<course>_module_order.csv
```

## Sources

- Groq models and prices: https://console.groq.com/docs/models
- Groq rate limits: https://console.groq.com/docs/rate-limits
- DeepSeek prices: https://api-docs.deepseek.com/quick_start/pricing
