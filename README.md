# ACE-AI — Training Plan Generation

Two LLM agents that turn a set of Learning Objectives into a training plan. See `CLAUDE.md` for the
design summary and current phase.

## Setup (WSL)

Keep the repo on the Linux filesystem (not under `/mnt/c`).

```bash
sudo apt install python3 python3-venv python3-pip   # Python 3.11+
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env    # then set GROQ_API_KEY
```

## Data (not in git)

`data/` is git-ignored. Put these in `data/raw/`:

- the six course CSVs (`*_learning_objectives_20260916.csv`) from the Drive folder
  "Documents for Ruchi (Training Plan Generation)/sail_course_LOs"
- `PPP_syllabus_broad_LOs.csv` (from the ACE-AI kit)

Then generate the derived files in `data/processed/`:

```bash
python scripts/profile_data.py       # data/processed/profile.md
python scripts/build_ground_truth.py # data/processed/ground_truth/<course>.json
```

Tests that need the real data are skipped when it is missing.

## Tests and lint

```bash
pytest
ruff check .
```

## Layout

```
src/aceai/   schemas.py, config.py, ingest/, tools/, llm/, agents/
data/raw/    input CSVs (git-ignored)
data/processed/   generated (git-ignored)
runs/        per-run JSON outputs (git-ignored)
tests/
scripts/
```
