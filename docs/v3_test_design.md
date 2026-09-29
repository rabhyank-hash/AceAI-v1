# Agent 1 v3: test design

How v3 (plan: [implementation_plan.md](implementation_plan.md)) is tested. Each experiment's
exact configuration and outcome is in [experiments.md](experiments.md).

## Question

Does v3 give the same course for the same LOs across runs, and how close is that course to the
authors'?

## Criteria

All must hold on every test course.

| Id | Criterion | Target |
|---|---|---|
| C1 | Sequence agreement between independent runs | ≥ 0.90 |
| C2 | Module ARI between independent runs | ≥ 0.60 |
| C3 | BCubed F1 against the CSV modules | ≥ v1 mean − 0.05 on the same sample |
| C4 | Validity: every LO in exactly one module, only identical LOs merged, checks pass | all runs |

## Design

- **Courses and samples:** DataEng, CloudAdmin, CloudNative, each limited to its first 3 CSV
  modules (11, 15 and 44 LOs; CloudNative has 16 exact duplicates). DataEng spans two CSV units;
  CloudNative tests the exact-duplicate rule.
- **Model and settings:** `openai/gpt-oss-120b` on Groq, reasoning effort low, temperature 0.
- **Runs:** three independent v3 runs per course, seeds 0–4, 5–9 and 10–14 (5 order asks and 5
  split asks per run, same seeds). No call is shared between runs, giving 3 run pairs per
  course.
- **Reference:** v1 on the same samples, seeds 0, 1, 2 (branch `agent1-poc`).
- **Rate limits:** calls wait out rate limits, so no ask is lost to them.

## Metrics

- **Sequence agreement:** share of LO pairs ordered the same way (0.5 = random); between runs
  and against the CSV order.
- **Module ARI and BCubed F1:** between runs and against the CSV modules.
- **Validity:** check results, coverage, asks excluded.
- **Cost:** tokens per run.

Definitions: [evaluation.md](evaluation.md).

## Analysis

Descriptive, per course: mean and range over the 3 run pairs. No significance test (3 pairs per
course). Every run is reported.

## Limitations

- Three run pairs per course.
- Sequence agreement with the CSV cannot separate errors from valid alternatives; labeled module
  order is needed for that.
- Temperature 0 and caching: variation comes only from input order and codes, which is what is
  tested.

## Run

```bash
bash scripts/run_v3_experiment.sh
```
