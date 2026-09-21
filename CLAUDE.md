# ACE-AI — Training Plan Generation (TEEL Lab / CMU)

Project brief for Claude Code. Read this first in every session. The full design lives in the Drive doc
"ACE-AI Training Plan Generation — Approach & Implementation Plan" and "ACE-AI Schema Concepts v1"; this file
is the working summary. If this file and those docs disagree, stop and ask.

## What we are building

Two LLM agents that turn a set of Learning Objectives (LOs) into a training plan.

- **Agent 1 — Sequencer.** Input: a raw, mixed-granularity LO list (course-level goals mixed with detailed LOs).
  Output: a deduplicated, dependency-ordered tree — ordered modules, each headed by a course-level LO where one
  applies, containing the detailed LOs it covers, with provenance back to every raw LO. Never sees constraints.
- **Agent 2 — Constraint Optimizer.** Input: Agent 1's tree + constraints (time, cost, learner level).
  Output: time allocation, schedule, delivery & grouping per module, and logged compression decisions.
  Never reorders Agent 1's tree; compresses only contiguous spans. **Not in scope yet.**

Learning activities (lectures, labs, …) are **out of scope** for this system.

**Design rule for both agents:** the LLM makes judgment calls; deterministic **code tools** do mechanical checks
(schema validation, graph checks, constraint arithmetic). Tools must never call an LLM.

## Current phase: scaffold + Agent 1 code tools

In scope now:
1. Project scaffold, config, provider-agnostic LLM client (no agent prompts yet).
2. LO ingestion from CSV into typed objects.
3. Pydantic schemas (below).
4. Ground-truth extraction (the human unit/module structure from the CSVs).
5. Agent 1's deterministic tools, with tests.

Out of scope now: Agent 1 prompts/agent loop, Agent 2, evaluation metrics, any cloud infrastructure.

## Data

`data/raw/` (copy from the Drive folder "Documents for Ruchi (Training Plan Generation)/sail_course_LOs"):

- `PPP_learning_objectives_20260916.csv` — Practical Programming with Python (202 LOs, units 0–10)
- `AI_Practitioner_learning_objectives_20260916.csv`
- `CloudAdmin_learning_objectives_20260916.csv`
- `CloudDevOps_learning_objectives_20260916.csv`
- `CloudNative_learning_objectives_20260916.csv`
- `DataEng_learning_objectives_20260916.csv`
- `PPP_syllabus_broad_LOs.csv` — 22 course-level LOs from the PPP syllabus (5 course goals, 9 conceptual,
  8 project-level). Provided with this kit.

LO CSV columns (UTF-8 with BOM, CRLF line endings, quoted fields may contain commas):
`course name, Unit no, Unit Name, module type, Module name, LO no, Learning Objective`

- `module type` ∈ {CONCEPT, PRIMER, PROJECT}.
- The LO id within a course is (Unit no, module type, Module name, LO no) — `LO no` restarts per module.
- Known quirks to handle, not "fix" silently: casing/whitespace inconsistencies in unit and module names
  (e.g. "Introduction TO DAta Engineering", trailing spaces), LOs that are course-logistics rather than learning
  (e.g. "Attempt Sail() inline activities."), near-identical LOs repeated across CONCEPT and PRIMER modules,
  and whole modules shared across courses (Cloud Admin and Cloud Native both contain the Docker/Kubernetes
  modules). Flag these in a data profile; do not delete them — they are useful test cases for deduplication.

**Agent 1 input** = a course's CSV LOs (flattened and shuffled, structure removed), plus the syllabus broad LOs
where available (only PPP has them for now). **Ground truth** = the CSV's unit → module → LO structure.
Only PPP has course-level broad LOs; for the other five courses, input is detailed LOs only (open question
whether broad LOs will be added later).

## Schemas (Phase 0 — from "Schema Concepts v1")

Implement as Pydantic v2 models in `src/aceai/schemas.py`. Names and enums must match exactly.

- **BloomLevel**: `C1` Remember … `C6` Create (ordered; compare by number).
- **LearningObjective**: `id`, `raw_text`, `canonical_text | None`, `source_ids: list[str]`, `verb`,
  `bloom_level`, `track` ("conceptual" | "applied"), `target_concept`, `scope` ("atomic" | "aggregate"),
  `parent_id | None`, `depends_on: list[str]`, `priority: int | None` (1–5, 1 = highest), `depth_floor | None`.
- **RawLO** (ingestion record, not in the design doc): `raw_id`, `course`, `text`, and source metadata
  (`unit_no`, `unit_name`, `module_type`, `module_name`, `lo_no`, or `level` for syllabus broad LOs).
  Ground-truth fields must be kept separate so they can be stripped before an LO reaches Agent 1.
- **Module**: `id`, `title`, `order`, `aggregate_lo_id | None`, `lo_ids`, `depends_on`.
- **SequencerOutput** (Agent 1 output): ordered `modules`, all `LearningObjective`s, and a provenance map
  raw_id → surviving LO id.
- Constraint / TrainingPlan models: stub only (Agent 2 is not in scope yet).

## Agent 1 code tools (deterministic, fully tested)

Each returns a structured result (`ok: bool`, `errors: list[...]`) that an LLM agent can read and act on.

1. `validate_lo(lo)` / `validate_output(output)` — schema + enum validation; parent/child ids exist;
   aggregate LOs have children; no LO is its own ancestor.
2. `check_provenance(raw_los, output)` — every raw LO maps to exactly one surviving LO; report missing,
   duplicated, and unknown ids.
3. `check_cycles(los)` — cycle detection on the LO prerequisite graph; return the cycle(s) as id lists.
4. `check_module_order(modules, los)` — no module depends on a later module; no LO inside a module
   depends on a later LO in the same module.
5. `topo_sort_modules(modules)` — deterministic topological order of modules; also report where the graph
   allows more than one valid order (ties), so the agent can choose and justify.
6. `build_module_graph(modules, los)` — aggregate LO-level `depends_on` edges up to module-level edges.

Use `networkx` for graph work. Tools never mutate their inputs.

## LLM access

- Initial provider: **Groq**, model `llama-3.3-70b-versatile` (OpenAI-compatible API,
  base URL `https://api.groq.com/openai/v1`). Other providers (Mistral, OpenRouter, DeepSeek) should be
  addable by config only.
- Client in `src/aceai/llm/client.py`: thin wrapper over the `openai` Python SDK with `base_url` + key
  from config; JSON-mode / tool-calling support; retry with backoff on HTTP 429 honoring `retry-after`;
  on-disk response cache keyed by (provider, model, messages, params) so reruns are free.
- Free-tier rate limits (requests and **tokens per minute/day**) are low and change; read the current values
  from the Groq console and put them in config. A whole course's LO list must fit the per-minute token budget
  or be batched.
- Secrets in `.env` (`GROQ_API_KEY=...`), never committed. Provide `.env.example`.

## Conventions

- Python ≥ 3.11, `src/` layout, package `aceai`, `pyproject.toml`, `pytest`, `ruff`.
- Layout:
  ```
  src/aceai/{schemas.py, config.py, ingest/, tools/, llm/, agents/ (empty for now)}
  data/raw/  data/processed/  runs/  tests/
  ```
- Everything a run produces goes under `runs/<timestamp>/` as JSON (inputs, outputs, tool results, LLM
  calls). No silent drops: anything filtered or merged is logged.
- Deterministic by default: seeded shuffles, sorted outputs, stable ids.
- Don't add infrastructure (databases, cloud services, orchestration). Local files only for now.

## Open questions (don't decide these in code — ask)

1. Bloom C1–C6 as the depth scale; how ambiguous verbs ("discuss", "use", "understand") are resolved.
2. Dedup rule: collapse only LOs with the same track *and* Bloom level?
3. Contiguous span definition for Agent 2 (module boundaries, parallel tracks).
4. Cost/time model for Agent 2 (contact hours / cost points / time only).
5. Whether course-level broad LOs will be added for the five non-PPP courses.
6. How to treat logistics-only LOs (e.g. "Attempt Sail() inline activities") — drop, flag, or keep.
