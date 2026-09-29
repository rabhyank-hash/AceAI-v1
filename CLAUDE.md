# ACE-AI — Training Plan Generation (TEEL Lab / CMU)

Project brief for Claude Code. Read this first in every session. It summarizes the current state; the
authoritative documents are in `docs/` on the current branch (below). The Google Drive document "ACE-AI
Training Plan Generation — Approach & Implementation Plan" is **plan v1** and is superseded by
`docs/implementation_plan.md`. Schemas still follow the Drive document "ACE-AI Schema Concepts v1". If this
file and `docs/` disagree, stop and ask.

## What we are building

Two LLM agents that turn a set of Learning Objectives (LOs) into a training plan.

- **Agent 1 — Sequencer.** Input: one course's raw LO list, which may mix course-level goals with detailed
  LOs. Output: the LOs in teaching order, split into modules (contiguous segments of that order), with
  provenance back to every raw LO. Only LOs with identical text are merged, and no LO may be dropped.
  Course-level LOs are to become parents of the detailed LOs they contain (planned; not implemented in v3).
  Never sees constraints. Sees one course at a time; duplicates across courses are out of scope.
- **Agent 2 — Constraint Optimizer.** Input: Agent 1's output + constraints (time, cost, learner level).
  Output: time allocation, schedule, delivery & grouping, and logged compression decisions. Never reorders
  Agent 1's order; compresses only contiguous spans. Owns LO normalization (verb, Bloom level, track, target
  concept), which it needs for its depth scale. **Not started.**

Learning activities (lectures, labs, …) are **out of scope** for this system.

**Design rules:** the LLM makes judgment calls; deterministic **code** does mechanical checks and
aggregation. Code never calls an LLM. The tool must stand alone: the same LOs must give the same course
regardless of input order, so every LLM judgment is repeated on shuffled input and aggregated by code.
Bloom level is a depth scale for Agent 2, not a teaching-order signal.

## Versions and branches

Each version lives on its own branch; the current branch holds only that version's code and documents.

| Branch | Version | Agent 1 design | Outcome |
|---|---|---|---|
| `main` | — | scaffold, schemas, ingestion, ground truth, Agent 1 input | base |
| `agent1-poc` | v1 | one LLM call proposes the whole tree (normalization, merges, modules, LO prerequisites); module order by topological sort of the prerequisites; six code tools | valid output, but the result depends on input order |
| `agent_v2` | v2 | consensus grouping over repeated runs, module prerequisites voted over repeated asks, rule-based tie-breaking | more consistent on one course only; grouping unstable |
| `agent_v3` | **v3 (current)** | order first, then split; repeated asks on shuffled input, aggregated in code | order consistent; module split not yet |

A new version (branch, plan, test design) is created only for a substantial change in architecture, data
flow or metrics. Otherwise, update the current version's documents in place (no amendment history; git
history is the record) and log the experiment in `docs/experiments.md`.

## Agent 1 v3 (current)

Code: `src/aceai/agents/sequencer_v3.py`; run with `scripts/run_v3.py`.

1. **Exact duplicates (code):** LOs whose text is identical after whitespace normalization (case-sensitive)
   are merged; provenance lists every original.
2. **Order asks (LLM, k = 5):** every LO, shuffled and relabelled L1…Ln per ask (permutation
   self-consistency, Tang et al., NAACL 2024); the model returns the course in teaching order. Prompt
   `v3-2`, which states that every LO must be included. (The prompt also asks for modules; only its order is
   used.)
3. **Consensus order (code):** mean position across valid asks (ties by id).
4. **Split asks (LLM, k = 5):** the fixed consensus order, LOs under codes shuffled per ask; the model
   splits it into modules and decides how many. Prompt `v3-seg-1`.
5. **Modules (code):** a boundary where a strict majority of valid split asks place one.

Every answer is checked (all LOs present exactly once; the split keeps the given order) and gets one repair
message if not; still-invalid asks are excluded, and a run needs at least 3 valid asks per stage. The output
is checked with `validate_output` and `check_provenance`. The output schema's normalization fields hold
placeholders (`PLACEHOLDER` in the code) that nothing uses.

**Current test results** (`docs/experiments.md`, `docs/v3_test_design.md`; DataEng, CloudAdmin, CloudNative,
first 3 CSV modules each; 3 independent runs per course):

- Sequence agreement between runs 0.92 / 0.93 / 0.91 (target ≥ 0.90): met.
- Module ARI between runs 0.51 / 0.85 / 0.38 (target ≥ 0.60): not met. The model picks 4–10 modules for
  the same sequence.
- BCubed F1 against the CSV modules 0.68 / 0.53 / 0.41 (target: v1 mean − 0.05 = 0.62 / 0.62 / 0.39):
  not met on CloudAdmin.
- All runs valid.

## Data

`data/raw/` (git-ignored; copy from the Drive folder "Documents for Ruchi (Training Plan Generation)/
sail_course_LOs"):

- `PPP_learning_objectives_20260916.csv` — Practical Programming with Python (202 LOs, units 0–10)
- `AI_Practitioner_learning_objectives_20260916.csv` (125 LOs)
- `CloudAdmin_learning_objectives_20260916.csv` (197 LOs)
- `CloudDevOps_learning_objectives_20260916.csv` (171 LOs)
- `CloudNative_learning_objectives_20260916.csv` (166 LOs)
- `DataEng_learning_objectives_20260916.csv` (57 LOs)
- `PPP_syllabus_broad_LOs.csv` — 22 course-level LOs from the PPP syllabus (5 course goals, 9 conceptual,
  8 project-level). Only PPP has course-level LOs.

LO CSV columns (UTF-8 with BOM, CRLF line endings, quoted fields may contain commas):
`course name, Unit no, Unit Name, module type, Module name, LO no, Learning Objective`

- `module type` ∈ {CONCEPT, PRIMER, PROJECT}.
- The LO id within a course is (Unit no, module type, Module name, LO no) — `LO no` restarts per module.
- Known quirks, reported in `data/processed/profile.md` and not fixed silently: casing/whitespace variants in
  unit and module names, logistics-only LOs (e.g. "Attempt Sail() inline activities."), exact and
  near-duplicate LOs, and modules shared across courses.

**Agent 1 input** = a course's LOs with all structure removed, under opaque ids (hash of the raw id),
shuffled by seed; trailing `(LOn)` markers are stripped because they reveal module order. **Ground truth** =
the CSV's unit → module → LO structure (`src/aceai/ingest/ground_truth.py`).

## Schemas (from "Schema Concepts v1")

Pydantic v2 models in `src/aceai/schemas.py`. Names and enums must match exactly.

- **BloomLevel**: `C1` Remember … `C6` Create (ordered; compare by number).
- **LearningObjective**: `id`, `raw_text`, `canonical_text | None`, `source_ids: list[str]`, `verb`,
  `bloom_level`, `track` ("conceptual" | "applied"), `target_concept`, `scope` ("atomic" | "aggregate"),
  `parent_id | None`, `depends_on: list[str]`, `priority: int | None` (1–5, 1 = highest), `depth_floor | None`.
- **RawLO** (ingestion record): `raw_id`, `course`, `text`, `original_text`, and a `source` (CSV position, or
  syllabus level) kept separate so all ground truth can be stripped before an LO reaches Agent 1.
- **Module**: `id`, `title`, `order`, `aggregate_lo_id | None`, `lo_ids`, `depends_on`.
- **SequencerOutput** (Agent 1 output): ordered `modules`, all `LearningObjective`s, and a provenance list
  of (raw_id → surviving LO id).
- Constraint / TrainingPlan models: stubs (Agent 2 not started).

## Checks (`src/aceai/tools/`)

Deterministic, never mutate inputs, return a structured result (`ok`, `errors`, `warnings`; each issue has a
code, a message and the ids involved):

- `validate_lo(lo)` / `validate_output(output)` — schema and enums; ids and references exist; aggregate LOs
  have children; no LO is its own ancestor; every atomic LO in exactly one module; module order ascending.
- `check_provenance(raw_los, output)` — every raw LO maps to exactly one surviving LO; reports missing,
  duplicated and unknown ids.

## Evaluation (`docs/evaluation.md`)

- **Consistency between runs first:** sequence agreement (share of LO pairs in the same order; 0.5 =
  random) and module ARI.
- **Against the authors' structure:** BCubed F1 and ARI against CSV modules; sequence agreement with the CSV
  order. The CSV is one valid answer, not the only one.
- **Needs labels:** violations of labeled module-order pairs (`scripts/make_label_sheets.py` writes the
  sheets to `annotations/`); PPP containment.
- Code: `src/aceai/eval/compare.py`, `scripts/consistency.py`, `scripts/analyze_experiments.py`.

## LLM access

- Provider: **Groq**, model `openai/gpt-oss-120b` (open weights), reasoning effort low, temperature 0.
  `llama-3.3-70b-versatile` is no longer served to this key. OpenRouter, Mistral and DeepSeek are configured
  in `src/aceai/config.py` and need only a key.
- Groq free tier for `gpt-oss-120b`: 8K tokens/minute, 200K tokens/day, 1K requests/day. Record current
  limits in `config.py`.
- Client `src/aceai/llm/client.py`: thin wrapper over the `openai` SDK; JSON mode and tool calling; retries
  429 honoring `retry-after` / `x-ratelimit-reset-*` (and 5xx, connection errors); refuses requests that
  exceed the per-minute budget; raises `InvalidJSONReply` when the provider rejects a JSON-mode reply;
  on-disk response cache in `.cache/llm/` keyed by (provider, model, messages, params), so reruns are free;
  a call log per run. v3 runs wait out rate limits instead of losing asks to them.
- Secrets in `.env` (`GROQ_API_KEY=...`), never committed; template in `.env.example`.

## Documents (`docs/`)

- `implementation_plan.md` — the plan as of the current version.
- `v3_test_design.md` — criteria, design and metrics of the v3 test.
- `experiments.md` — every experiment (E1, E2, …): configuration, outcome, records, corrections.
- `evaluation.md` — metrics and labels needed.
- `poc_report.md` — report across v1, v2 and v3.

## Conventions

- Python ≥ 3.11, `src/` layout, package `aceai`, `pyproject.toml`, `pytest`, `ruff` (line length 100).
- Runs are named by version and experiment, never by timestamp:
  `runs/<version>/<experiment>/<course>_s<first seed>/` (local, git-ignored; full data including LO text).
  `runs/experiments.tsv` lists every run as `<experiment>_s<seed>`, course, path.
- Records for git: `scripts/export_records.py` copies runs to the same path under `experiments/` without LO
  text or model free text (module titles replaced by ids). Tables regenerate from
  `experiments/<version>/experiments.tsv` with `scripts/analyze_experiments.py`.
- No course LO text in git: `data/`, `runs/`, `annotations/` are git-ignored.
- Every experiment is logged in `docs/experiments.md` with its configuration (branch, commit, prompt
  versions, courses, sample, seeds, rules) before or with its results; code is committed before a run.
- Deterministic by default: seeded shuffles, sorted outputs, stable ids, temperature 0. No silent drops:
  anything merged or excluded is logged.
- Tests never call an LLM (`tests/fakes.py` replaces the SDK).
- Don't add infrastructure (databases, cloud services, orchestration). Local files only.

## Open questions (don't decide these in code — ask)

1. Bloom C1–C6 as Agent 2's depth scale; how ambiguous verbs ("discuss", "use", "understand") are resolved.
2. Contiguous span definition for Agent 2 (module boundaries, parallel tracks).
3. Cost/time model for Agent 2 (contact hours / cost points / time only).
4. Whether course-level LOs will be added for the five non-PPP courses.
5. Whether a detailed LO may be made the parent of other detailed LOs (containment); if yes, it must still
   appear in the teaching order.
6. How many modules a course should have: the model's choice varies (4–10 for the same sequence).

Decided (by the user): Agent 1 orders and splits into modules; only identical LOs are merged; no LO may be
excluded (logistics LOs are kept); Bloom level is not an ordering signal; the tool must stand alone.

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).
