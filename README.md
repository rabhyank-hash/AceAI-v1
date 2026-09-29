# ACE-AI: Training Plan Generation

TEEL Lab, Carnegie Mellon University.

ACE-AI turns a set of Learning Objectives (LOs) into a training plan with two LLM agents:

- **Agent 1, the Sequencer**, puts a course's LOs in teaching order and splits that order into
  modules. It never sees time, cost or learner constraints.
- **Agent 2, the Constraint Optimizer**, fits the result to time, cost and learner level. Not
  built yet.

This branch, `agent_v3`, holds **version 3** of Agent 1. Earlier versions are on `agent_v2` and
`agent1-poc`.

## Documents

| File | Content |
|---|---|
| [docs/implementation_plan.md](docs/implementation_plan.md) | The plan, as of v3 |
| [docs/v3_test_design.md](docs/v3_test_design.md) | How v3 is tested: criteria, design, metrics |
| [docs/experiments.md](docs/experiments.md) | Every experiment so far: configuration, outcome, records |
| [docs/evaluation.md](docs/evaluation.md) | Metrics and the labels still needed |
| [docs/poc_report.md](docs/poc_report.md) | Report across v1, v2 and v3 |

## How Agent 1 v3 works

1. Merge LOs with identical text (code).
2. Ask the model for the course in teaching order, 5 times, each time with the LOs shuffled and
   relabelled (permutation self-consistency).
3. Order the LOs by their mean position across the answers (code).
4. Ask the model, 5 times, to split that fixed order into modules; it decides how many.
5. Keep a module boundary where most answers place one (code).

Every answer is checked (all LOs present; order kept when splitting) and repaired once if needed.
The output is checked for schema and provenance.

## Repository layout

```
src/aceai/
  schemas.py              data models: LearningObjective, Module, SequencerOutput, RawLO
  config.py               paths, LLM providers, model settings
  ingest/
    loader.py             course CSVs -> RawLO records
    profile.py            data profile (quirks, duplicates, logistics LOs)
    ground_truth.py       the authors' unit -> module -> LO structure
    agent1_input.py       LO texts under opaque ids, shuffled by seed
    sample.py             test samples (first N CSV modules)
  tools/                  checks: validate_output, check_provenance (results.py: result types)
  llm/client.py           provider-agnostic LLM client: retries, cache, call log
  agents/sequencer_v3.py  Agent 1 v3
  eval/compare.py         scores against the authors' structure
scripts/
  profile_data.py         writes data/processed/profile.md
  build_ground_truth.py   writes data/processed/ground_truth/<course>.json
  llm_smoke.py            one tiny request: checks the API key and limits
  run_v3.py               one v3 run on one course sample
  run_v3_experiment.sh    the v3 test end to end
  analyze_experiments.py  tables over an experiment log
  consistency.py          agreement between runs
  export_records.py       copies run records to experiments/<name>/ without LO text
  make_label_sheets.py    module-order labeling sheets in annotations/
tests/                    unit tests; the LLM is replaced by a fake (tests/fakes.py)
docs/                     see Documents
experiments/<version>/<experiment>/<course>_s<seed>/
                          experiment records: configs, scores, structures (no LO text)
```

Local only (git-ignored): `data/` (course CSVs and derived files), `runs/` (full run output,
including LO text), `annotations/` (labeling sheets), `.cache/` (LLM response cache), `.env`
(API key).

## Setup

Python 3.11+. In WSL, keep the repository on the Linux filesystem.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env    # then set GROQ_API_KEY (https://console.groq.com/keys)
python scripts/llm_smoke.py
```

Data: put the six course CSVs (`*_learning_objectives_20260916.csv`, Drive folder "Documents for
Ruchi (Training Plan Generation)/sail_course_LOs") and `PPP_syllabus_broad_LOs.csv` in
`data/raw/`, then:

```bash
python scripts/profile_data.py
python scripts/build_ground_truth.py
```

## Running

```bash
python scripts/run_v3.py --experiment E9 --course DataEng --seeds 0 1 2 3 4   # one run
bash scripts/run_v3_experiment.sh E9                                          # the v3 test
python scripts/analyze_experiments.py --log experiments/v3/experiments.tsv  # tables from git
```

Each run writes `runs/v3/<experiment>/<course>_s<seed>/`: config, the model's answers, consensus
order, output, checks, comparison with the authors' structure, and every LLM call. Experiment ids
(E1, E2, ...) are listed in [docs/experiments.md](docs/experiments.md);
`scripts/export_records.py` copies runs, without LO text, to the same path under `experiments/`.

## Tests

```bash
pytest
ruff check .
```
