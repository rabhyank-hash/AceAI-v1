# ACE-AI: Training Plan Generation

TEEL Lab, Carnegie Mellon University.

ACE-AI turns a set of Learning Objectives (LOs) into a training plan with two LLM agents:

- **Agent 1, the Sequencer**, takes a raw LO list that mixes course-level and detailed LOs. It
  produces a deduplicated, dependency-ordered tree of modules, with provenance back to every raw LO.
  It never sees time, cost or learner-level constraints.
- **Agent 2, the Constraint Optimizer**, fits that tree to constraints (time, cost, learner level):
  time allocation, schedule, delivery and grouping, and logged compression decisions. It may
  compress contiguous spans of the tree but never reorder it. **Not built yet.**

Selecting learning activities (lectures, labs, etc.) is out of scope.

Design documents on Google Drive: *ACE-AI Training Plan Generation — Approach & Implementation
Plan*, *ACE-AI Schema Concepts v1*, *ACE-AI Literature Survey*. `CLAUDE.md` is the working summary.

## Contents

- [Status](#status)
- [Design principles](#design-principles)
- [Repository layout](#repository-layout)
- [Setup](#setup)
- [Data](#data)
- [Schemas](#schemas)
- [Agent 1 code tools](#agent-1-code-tools)
- [LLM client](#llm-client)
- [Agent 1 proof of concept](#agent-1-proof-of-concept)
- [Evaluation](#evaluation)
- [Running Agent 1](#running-agent-1)
- [Results so far](#results-so-far)
- [Differences from the plan](#differences-from-the-plan)
- [Open questions](#open-questions)
- [Tests](#tests)
- [References](#references)

## Status

| Component | Status |
|---|---|
| Schemas (Phase 0) | Done. Agent 2 models (`Constraints`, `TrainingPlan`) are stubs. |
| LO ingestion, data profile, ground truth | Done for all six courses |
| Agent 1 code tools (6) | Done |
| LLM client | Done |
| Agent 1 prompts and loop | Proof of concept: one LLM call plus tool-driven repair, not yet the plan's six-step chain |
| Evaluation against the CSV structure | Grouping and order agreement done; dependency, merge and containment scores need labeled data |
| Baselines | Not started |
| Agent 2 | Not started |

## Design principles

1. **The LLM judges; code checks.** The model makes judgment calls (Bloom level, duplicates,
   containment, prerequisites, grouping). Deterministic tools make the mechanical checks (schema,
   provenance, graph structure). Tools never call an LLM. Syahputra et al. [1] show that
   role-specialized decomposition plus deterministic verification roughly doubles structural
   compliance over single-agent generation, across four open-weight backbones.
2. **Tool results are written for the agent.** Every tool returns `ok`, `errors` and `warnings`.
   Each issue has a stable code, a one-sentence message and the ids involved, so the agent can act
   on it.
3. **No silent drops.** Every raw LO maps to exactly one surviving LO. Every text edit, merge and
   fill-in is logged, and every run is saved under `runs/`.
4. **Deterministic by default.** Shuffles are seeded, ids are stable, outputs are sorted, and LLM
   responses are cached, so reruns reproduce.
5. **Depth is Bloom's revised taxonomy**, C1 Remember to C6 Create [7], inferred from each LO's
   action verb. Syahputra et al. [1] use the same C1–C6 coding.
6. **Scores come from code and human-built structure, not LLM judges.** Instructional Agents [2]
   found that LLM reviewers gave compressed mid-range scores and could not separate good course
   materials from weak ones.

## Repository layout

```
src/aceai/
  schemas.py            Pydantic models: BloomLevel, LearningObjective, Module, SequencerOutput, RawLO
  config.py             paths, LLM providers, per-model rate limits
  ingest/
    loader.py           CSV -> RawLO
    profile.py          data profile (quirks, duplicates, logistics LOs)
    ground_truth.py     CSV unit/module structure; ground_truth_to_output()
    agent1_input.py     strips structure, assigns opaque ids, seeded shuffle
  tools/                Agent 1 deterministic tools
    validate.py         validate_lo, validate_output
    provenance.py       check_provenance
    graph.py            check_cycles, check_module_order, topo_sort_modules, build_module_graph
    results.py          shared ToolResult / Issue types
  llm/client.py         provider-agnostic chat client
  agents/sequencer.py   Agent 1 proof of concept: prompt, assembly, check-and-repair loop
  eval/compare.py       comparison with the CSV structure
scripts/
  profile_data.py       writes data/processed/profile.md
  build_ground_truth.py writes data/processed/ground_truth/<course>.json
  llm_smoke.py          one tiny request: checks key, model and rate limits
  run_poc.py            runs Agent 1 on a course or units and writes runs/<timestamp>_<course>/
tests/                  pytest suite; LLM tests use a fake SDK (tests/fakes.py)
data/raw/               input CSVs (git-ignored)
data/processed/         generated files (git-ignored)
runs/                   run outputs (git-ignored)
```

## Setup

Python 3.11+. Keep the repository on the Linux filesystem in WSL (not under `/mnt/c`).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env    # then set GROQ_API_KEY
```

Get a Groq key at https://console.groq.com/keys. Check that it works:

```bash
python scripts/llm_smoke.py
```

## Data

`data/` is git-ignored. Put these in `data/raw/`:

- the six course CSVs (`*_learning_objectives_20260916.csv`) from the Drive folder
  "Documents for Ruchi (Training Plan Generation)/sail_course_LOs"
- `PPP_syllabus_broad_LOs.csv` (from the ACE-AI kit)

Then generate the derived files:

```bash
python scripts/profile_data.py        # data/processed/profile.md
python scripts/build_ground_truth.py  # data/processed/ground_truth/<course>.json
```

| Course | Detailed LOs | Syllabus LOs | Units | Modules |
|---|---:|---:|---:|---:|
| PPP (Practical Programming with Python) | 202 | 22 | 10 | 33 |
| CloudAdmin | 197 | 0 | 10 | 35 |
| CloudDevOps | 171 | 0 | 8 | 29 |
| CloudNative | 166 | 0 | 8 | 30 |
| AI_Practitioner | 125 | 0 | 6 | 34 |
| DataEng | 57 | 0 | 5 | 14 |

CSV columns: `course name, Unit no, Unit Name, module type, Module name, LO no, Learning Objective`.
`module type` is CONCEPT, PRIMER or PROJECT. Only PPP has syllabus (course-level) LOs: 5 course
goals, 9 conceptual, 8 project-level.

### Ingestion

`ingest/loader.py` handles a byte-order mark, CRLF line endings and quoted fields containing
commas. It normalizes whitespace into `text` and keeps the original cell in `original_text`. A
row that cannot be parsed raises an error naming the file and line; nothing is dropped. Raw ids
encode the source position, e.g. `ppp-u03-concept-data-structures-lo02`.

### Data profile

`data/processed/profile.md` lists quirks without fixing them: casing and whitespace variants in
unit and module names, exact and near-duplicate LOs, and logistics-only LOs such as "Attempt
Sail() inline activities.". These stay in the data as test cases for deduplication. Each course is
processed on its own, so duplicates across courses are ignored.

### Ground truth

`extract_ground_truth` reads each course's unit → module → LO structure in file order. Names are
kept as written. Syllabus LOs are listed separately by level. Each LO is stored as
`{"id", "text"}`. The raw id is the key everything joins on, because texts are not unique within
a course and change during processing. The text is stored only so the file can be read by eye.

### Agent 1 input

`make_agent1_input(course, seed)` returns the only object the model sees:

- **Opaque ids.** Raw ids reveal unit, module and position, so each LO gets
  `LO-<first 6 hex of sha256(raw_id)>`. The map back to raw ids is never sent.
- **Seeded shuffle.** LOs are sorted by id, then shuffled with the seed.
- **Leak removal.** A trailing `(LOn)` marker equal to the CSV `LO no` appears in 49 PPP texts. It
  is stripped, and each edit is logged.

## Schemas

`src/aceai/schemas.py`, from *Schema Concepts v1*. All models reject unknown fields, so a
malformed LLM reply fails validation instead of losing fields.

- **BloomLevel**: `C1`–`C6`, ordered by number.
- **LearningObjective**: `id`, `raw_text`, `canonical_text`, `source_ids`, `verb`, `bloom_level`,
  `track` (`conceptual` | `applied`), `target_concept`, `scope` (`atomic` | `aggregate`),
  `parent_id`, `depends_on`, `priority` (1–5, 1 highest), `depth_floor`.
- **Module**: `id`, `title`, `order`, `aggregate_lo_id` (heading course-level LO, if any),
  `lo_ids`, `depends_on`.
- **SequencerOutput**: `modules`, `los`, `provenance`. Provenance is a list of
  `{raw_id, lo_id}` entries rather than a dict, so a duplicated mapping is visible to
  `check_provenance` instead of being collapsed by the JSON parser.
- **RawLO**: ingestion record. All ground-truth position fields sit in one `source` object so
  they can be stripped before an LO reaches Agent 1.

Models check only what a single object can check (types, ranges, self-reference). Checks across
objects are the tools' job.

## Agent 1 code tools

`src/aceai/tools/`. Graph work uses `networkx`. No tool mutates its input.

| Tool | Errors | Warnings |
|---|---|---|
| `validate_output` | schema errors; duplicate ids; unknown parent, dependency, member or head; parent not aggregate; aggregate LO without children; LO its own ancestor; atomic LO in no module or several; module `order` not ascending | module head whose children sit in other modules |
| `check_provenance` | raw LO unmapped, mapped more than once, or unknown; disagreement with `source_ids` | — |
| `check_cycles` | each prerequisite cycle as an id path | — |
| `check_module_order` | module depends on a later module; LO depends on an LO in a later module, or on a later LO in its own module | Bloom level decreases within a module |
| `topo_sort_modules` | module-level cycles | returns the order and every point where several orders are valid |
| `build_module_graph` | — | LO edges lifted to module edges; undeclared module edges; declared edges without LO evidence; edges to LOs outside all modules |

`validate_lo` checks a single LO. As a smoke test, the tools are run on PPP's human structure
converted to a `SequencerOutput` (`ground_truth_to_output`).

## LLM client

`src/aceai/llm/client.py`, a thin wrapper over the `openai` SDK. Providers are defined in
`config.py`. Adding one takes a config entry and a key in `.env`.

- **Providers**: Groq (default), OpenRouter, Mistral, DeepSeek.
- **Default model**: `openai/gpt-oss-120b` on Groq. `qwen/qwen3.8-27b` is also configured.
- **Rate limits**: Groq free tier, read from response headers on 2026-09-23: 8,000 tokens per
  minute and 1,000 requests per day. Before calling, the client refuses a request whose prompt
  plus `max_tokens` exceeds the per-minute limit (`RequestTooLarge`).
- **Retry**: on 429, waits for `retry-after` or Groq's `x-ratelimit-reset-*`, else exponential
  backoff. 5xx and connection errors are also retried.
- **Cache**: on disk in `.cache/llm/`, keyed by provider, model, messages and parameters.
- **JSON mode and tool calling.** If the provider rejects a JSON-mode reply (Groq
  `json_validate_failed`), the client raises `InvalidJSONReply` with the rejected text.
- **Per-model parameters**: `reasoning_effort: low` for gpt-oss, whose hidden reasoning tokens
  count against the per-minute budget.
- **Call log**: every call (request, reply, usage, rate-limit headers) is kept for the run record.

## Agent 1 proof of concept

`src/aceai/agents/sequencer.py`. A first version to find where the model fails before building
the plan's six-step chain.

1. **One call.** The model receives the shuffled LO list and returns JSON: per-LO normalization
   (verb, Bloom level, track, target concept, scope), merges, parent links, prerequisites, and
   modules in teaching order.
2. **Compact reply.** The model reuses an input id as each surviving LO's id and lists merged ids.
   `assemble()` fills in the mechanical fields: `source_ids`, `raw_text`, module `order` and
   provenance.
3. **Check.** All six tools run on the assembled output.
4. **Repair.** Errors (with their ids) go back to the model with its previous reply, up to
   `max_repairs` rounds (default 2). A provider JSON rejection is treated as a bad reply. Reply
   `max_tokens` is lowered to whatever the per-minute budget leaves.

Temporary rules for open questions 1, 2 and 6 are in the prompt, not the code, and are marked for
the proof of concept only:

- Bloom level from the verb. Ambiguous verbs: *understand, discuss, describe, explain* → C2;
  *use, utilize, implement, write* → C3; *compare, debug, inspect* → C4; *design, develop, build*
  → C6 only if the learner creates something new, else C3.
- Merge two LOs only if they state the same objective and share both track and Bloom level.
- Keep logistics LOs, with `target_concept` set to "course logistics".

## Evaluation

`src/aceai/eval/compare.py` compares an output with the CSV structure, per raw LO:

- **Grouping**: pairwise precision, recall and F1 (a pair is "together" if both LOs are in the
  same module), and the adjusted Rand index (ARI) [8]: 1 means identical groupings, about 0 means
  chance.
- **Order agreement**: over pairs of LOs that are in different modules in both, the fraction
  ordered the same way as the CSV. A random order scores 0.5.
- **Merges**: which raw LOs were merged, and whether they came from the same CSV module.
- **Syllabus LOs**: the scope and number of children for each.

Each score is computed against the CSV **module** and the CSV **unit**. A unit's CONCEPT, PRIMER
and PROJECT modules share a topic, so neither merging them nor splitting a unit is necessarily
wrong. The pair that is fair at any module size is **module recall** (are the authors' modules
kept together?) and **unit precision** (are topics from different units kept apart?). The CSV
order is one valid order, not the only one.

### Not yet measurable

The plan also asks for dependency violations, merge accuracy and containment accuracy. The CSVs
record none of these, so they need labels:

| Label | Scope | Enables |
|---|---|---|
| Unit prerequisites | per course, ≤ 45 unit pairs | dependency violations |
| Duplicate pairs | ~60 candidate pairs from the data profile | merge precision and recall |
| Syllabus LO → CSV modules | PPP, 22 rows | containment precision and recall |

### Planned baselines

From the plan (§6) and the literature survey:

- embeddings + clustering for grouping, ordered by the same topological sort;
- a graph-based curriculum-sequencing method over the prerequisite graph [5];
- a single-agent, single-pass baseline, as used by Syahputra et al. [1] and Instructional
  Agents [2].

## Running Agent 1

```bash
python scripts/run_poc.py --course DataEng --all-units
python scripts/run_poc.py --course PPP --units 0 1 2 --broad
python scripts/run_poc.py --course PPP --units 0 1 2 --dry-run   # writes input and prompt, no API call
```

Options: `--course`, `--units` / `--all-units`, `--broad` (include syllabus LOs; off by default
for partial samples because they cover the whole course), `--seed`, `--provider`, `--model`,
`--max-repairs`, `--max-tokens`, `--no-cache`, `--dry-run`.

Output in `runs/<timestamp>_<course>/`:

| Path | Contents |
|---|---|
| `report.md` | attempts, scores, and each predicted module with the CSV module of every LO |
| `config.json` | course, units, seed, provider, model, prompt version, prompt token estimate |
| `input/` | model input, id map (not sent), text edits, ground truth, full prompt |
| `attempts/NN/` | raw reply, assembled output, fill-in notes, all tool results, summary |
| `llm_calls.json` | every LLM call with usage and rate-limit headers |
| `output.json`, `comparison.json` | final output and scores, when the output parses |

### Token budget

At 8,000 tokens per minute, only DataEng (57 LOs, about 2.1K prompt + 3.8K reply tokens) fits in
one call. The larger courses need unit subsets, the multi-step chain, or a provider with higher
limits. Repair rounds resend the previous reply and currently do not fit.

## Results so far

**DataEng, all units, `gpt-oss-120b`, seed 0 (2026-09-23).** 2,091 prompt tokens, 3,802 reply
tokens, about 8 s.

- No cycles, no module depending on a later module. The one merge joined a MongoDB PRIMER LO
  with a NoSQL PROJECT LO from the same unit.
- 32 errors: 21 LOs placed in no module; 8 modules headed by an atomic LO (DataEng has no
  course-level LOs); 2 LOs in two modules; 1 LO both merged and kept.
- The repair round was refused as too large for the per-minute budget.
- On the 36 placed LOs: module recall 0.52, unit precision 0.65, module ARI 0.37, order agreement
  0.57 (module level). The model made 8 modules; the CSV has 14 in 5 units.

The errors are mostly about output format. Planned fixes:

1. Each LO names its module, and code builds `lo_ids`, so an LO cannot be unplaced or placed twice.
2. The prompt requires `aggregate_lo_id` to be empty unless the LO is course-level, and forbids
   keeping an LO that was merged away.
3. Repair asks only for the changed LOs and modules, following the targeted retry of
   Syahputra et al. [1].
4. The report falls back to the last attempt that produced an output.

## Differences from the plan

- **Model.** `CLAUDE.md` names `llama-3.3-70b-versatile` on Groq. Groq no longer serves it to this
  account; the default is `openai/gpt-oss-120b`.
- **Test courses.** The plan names PPP and a Udacity nanodegree. The test set is the six SAIL
  courses above.
- **PPP size.** The plan describes PPP as 129 LOs across 8 modules and 8 projects; the CSV has 202
  detailed and 22 syllabus LOs across 10 units. To confirm which version is the reference.
- **One call, not a chain.** The plan has six steps with a tool check after each. The proof of
  concept makes one call, code runs the repair loop, and the model orders modules itself. In the
  plan, `topo_sort_modules` sets the order and the agent only chooses among valid orders
  (step 6).
- **No prerequisite rationale.** The plan asks for a one-line rationale per prerequisite edge. The
  schema has no field for it yet.

## Open questions

From `CLAUDE.md` and the plan (§10):

1. Bloom C1–C6 as the depth scale; resolution of ambiguous verbs.
2. Dedup rule: merge only LOs with the same track and Bloom level?
3. Contiguous-span definition for Agent 2.
4. Cost and time model for Agent 2 (contact hours, cost points, or time only).
5. Whether course-level LOs will be added for the five non-PPP courses. Without them, containment
   (step 2) can only be tested on PPP.
6. Logistics-only LOs: drop, flag or keep.

Raised during implementation:

7. Who labels unit prerequisites, duplicate pairs and PPP containment? The labels depend on the
   answers to 1 and 2.
8. May `LearningObjective` get a field for prerequisite rationales?

## Tests

```bash
pytest
ruff check .
```

116 tests: schemas (22), ingestion (22), ground truth and input leak checks (10), tools (30), LLM
client (19), Agent 1 loop and metrics (11), scaffold (2). LLM tests replay scripted replies through
a fake SDK and never call an API. Tests that need the course CSVs are skipped when `data/raw/` is
empty.

## References

1. Syahputra, M. F., Sitompul, O. S., Fahmi, Lydia, M. S., Nainggolan, P. I., Mahardika, R., &
   Sulaiman, R. (2026). Development of multi-agent generative pipelines framework for learning plan
   generation with deterministic constraint verification. *Eastern-European Journal of Enterprise
   Technologies*, 2/2(140), 17–31. https://doi.org/10.15587/1729-4061.2026.356830
2. Yao et al. (2026). Instructional Agents. *EACL 2026*. arXiv:2508.19611.
3. Zhang et al. (2025). EduPlanner. *IEEE Transactions on Learning Technologies*. arXiv:2504.05370.
4. ISD-Agent-Bench: A comprehensive benchmark for evaluating LLM-based instructional design agents.
   arXiv:2602.10620.
5. A hybrid Transformer–Graph framework for curriculum sequencing and prerequisite optimization in
   CS education. *Algorithms* (MDPI), 2026. https://doi.org/10.3390/a19040308
6. Rubrics as Rewards. arXiv:2507.17746.
7. Anderson, L. W., & Krathwohl, D. R. (Eds.) (2001). *A Taxonomy for Learning, Teaching, and
   Assessing: A Revision of Bloom's Taxonomy of Educational Objectives*. Longman.
8. Hubert, L., & Arabie, P. (1985). Comparing partitions. *Journal of Classification*, 2(1),
   193–218.

[3], [4] and [6] are background. EduPlanner [3] and ISD-Agent-Bench [4] are related
instructional-design agent work that operates at the lesson and content level. Rubrics as
Rewards [6] is the planned pattern for Agent 2's soft-quality rubric. [7] and [8] are standard
references not in the literature survey.
