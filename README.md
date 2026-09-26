# ACE-AI: Training Plan Generation

> Branch `agent_v2`: plan v2 and its prototype. Start with [docs/poc_report.md](docs/poc_report.md).
> Plan v1 and the v1 implementation are on `agent1-poc`.

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
| Agent 1 prompts and loop | Proof of concept: one LLM call plus targeted repair; module order set by topological sort of the model's prerequisites (plan step 6). Not yet the plan's six-step chain. Runs on a module sample of all six courses. |
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
  agents/sequencer.py   Agent 1 v1: prompt, assembly, check-and-repair loop
  agents/sequencer_v2.py  Agent 1 v2 prototype: consensus grouping, voted module order
  eval/compare.py       comparison with the CSV structure
  eval/draw.py          module graph as a Mermaid diagram
scripts/
  profile_data.py       writes data/processed/profile.md
  build_ground_truth.py writes data/processed/ground_truth/<course>.json
  llm_smoke.py          one tiny request: checks key, model and rate limits
  run_poc.py            runs Agent 1 v1 on a course sample and writes runs/<timestamp>_<course>/
  run_v2.py             runs the v2 prototype from k v1 grouping runs
  run_v2_experiment.sh  the pre-registered v2 experiment, end to end
  summarize_runs.py     one table over several runs
  draw_graph.py         redraws the module graph of existing runs
  consistency.py        run-to-run consistency of two or more runs
  analyze_experiments.py tables over runs/experiments.tsv (settings, seeds, consistency)
  export_records.py     copies experiment records to experiments/<name>/ without LO text
  make_label_sheets.py  blank module-order labeling sheets in annotations/ (git-ignored)
docs/implementation_plan.md  the implementation plan (tracked version of the Drive document)
docs/evaluation.md      evaluation framework
docs/poc_report.md      report: v1 implementation and tests, failures, plan v2, prototype, metrics
docs/v2_preregistration.md  hypotheses and criteria for the v2 prototype, fixed before running
experiments/<name>/     experiment records: configs, scores, structures (no LO text)
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
   (verb, Bloom level, track, target concept, scope), merges, parent links, prerequisites, the
   module each LO belongs to, and modules in teaching order.
2. **Compact reply.** The model reuses an input id as each surviving LO's id and lists merged ids.
   Each atomic LO names its module; `assemble()` builds each module's `lo_ids` from those names,
   in the order the LOs are listed, so an LO cannot be listed in two modules or forgotten in the
   module lists. It also fills in `source_ids`, `raw_text`, module `order` and provenance.
3. **Order by prerequisites (plan step 6).** For every LO the model states which LOs a learner
   must master first, judged from the LO text. Code lifts these to module dependencies
   (`build_module_graph`), sets the module order by topological sort (`topo_sort_modules`) and
   orders LOs inside each module the same way. The model's own listing order is used only to
   break ties where several orders are valid, and it explains its tie-breaking in
   `order_rationale`, shown in the report.
4. **Check.** All six tools run on the assembled output.
5. **Targeted repair.** If any tool reports errors, the model gets a compact view of its current
   plan (one line per LO: id, module, Bloom level, track, scope, parent, prerequisites, merges),
   the module list, and the errors with the ids involved. It replies with a patch: only the LOs
   and modules that change, matched by id, plus ids to remove and an optional new module order.
   `apply_patch()` applies it mechanically. Each round sends the current plan, not the
   conversation history, so request size stays constant. This follows the targeted retry of
   Syahputra et al. [1]. Up to `max_repairs` rounds (default 2). A provider JSON rejection is
   treated as a bad reply.

Every input LO must end up in a module; a run with an unplaced LO is invalid.

Temporary rules for open questions 1, 2 and 6 are in the prompt, not the code, and are marked for
the proof of concept only:

- Bloom level from the verb. Ambiguous verbs: *understand, discuss, describe, explain* → C2;
  *use, utilize, implement, write* → C3; *compare, debug, inspect* → C4; *design, develop, build*
  → C6 only if the learner creates something new, else C3.
- Merge two LOs only if they state the same objective and share both track and Bloom level.
- Keep logistics LOs, with `target_concept` set to "course logistics".

## Evaluation

The full framework (levels, metrics, reference data, what is implemented, labels needed) is in
[docs/evaluation.md](docs/evaluation.md). `src/aceai/eval/compare.py` implements the parts that
need no labels, per raw LO:

- **Coverage**: LOs placed in a module / LOs in. Below 100% the run is invalid.
- **Grouping**: BCubed precision, recall and F1 (primary; per-LO averages, so large modules do
  not dominate) [9]; the adjusted Rand index (ARI) [8], where 1 means identical groupings and
  about 0 means chance; and pairwise precision and recall.
- **Order agreement**: over pairs of LOs that are in different modules in both, the fraction
  ordered the same way as the CSV (Kendall's τ rescaled [10]). A random order scores 0.5.
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

### Module graph drawing

`src/aceai/eval/draw.py` draws each run's module graph as a Mermaid flowchart, appended to
`report.md` and written to `module_graph.md` and `module_graph.html` in the run directory.
`scripts/draw_graph.py --latest` redraws existing runs.

- Columns are topological steps: a module sits one step after the latest module it depends on.
  Modules in the same step have no dependencies between them; these are the ties
  `topo_sort_modules` reports.
- Arrows point from a prerequisite to the module that needs it. Solid arrows come from LO-level
  prerequisites (labelled with the number of LO links); dashed arrows are declared by the model
  with no LO prerequisite behind them.
- Colour is the CSV unit most of the module's LOs come from, labelled with how many.

### Planned baselines

From the plan (§6) and the literature survey:

- embeddings + clustering for grouping, ordered by the same topological sort;
- a graph-based curriculum-sequencing method over the prerequisite graph [5];
- a single-agent, single-pass baseline, as used by Syahputra et al. [1] and Instructional
  Agents [2].

## Running Agent 1

```bash
python scripts/run_poc.py --course CloudAdmin --modules 3 --max-repairs 4   # module sample
python scripts/run_poc.py --course DataEng --all-units
python scripts/run_poc.py --course PPP --units 0 1 2 --broad
python scripts/run_poc.py --course PPP --units 0 1 2 --dry-run   # writes input and prompt, no API call
python scripts/summarize_runs.py --latest                         # one table over the newest runs
```

Sample options (pick one): `--modules N` (the first N CSV modules in file order, extended module
by module until the sample has `--min-los` LOs, default 20), `--units`, `--all-units`.
Other options: `--broad` (include syllabus LOs; off by default for partial samples because they
cover the whole course), `--seed`, `--provider`, `--model`, `--max-repairs`, `--max-tokens`,
`--no-cache`, `--dry-run`.

Output in `runs/<timestamp>_<course>/`:

| Path | Contents |
|---|---|
| `report.md` | coverage, attempts, scores, each predicted module with the CSV module of every LO, and the module graph |
| `module_graph.md`, `module_graph.html` | the module graph as Mermaid, and as a standalone page |
| `config.json` | course, sample, seed, provider, model, prompt version, prompt token estimate |
| `input/` | model input, id map (not sent), text edits, ground truth, full prompt |
| `attempts/NN/` | raw reply (a patch, on repair rounds), assembled full output, fill-in notes, tool results, summary |
| `llm_calls.json` | every LLM call with usage and rate-limit headers |
| `result.json` | pass/fail, stop reason, errors per attempt |
| `output.json`, `comparison.json` | output and scores from the last attempt that produced a usable plan |

### Token budget

Groq's free tier allows 8,000 tokens per minute. A first call for 20–60 LOs fits (about 1.5–2.5K
prompt + 2–4K reply tokens); a repair round is about 3.5K prompt tokens. Whole courses above
about 60 LOs need the multi-step chain or a provider with higher limits.

## Results so far

### Proof of concept: module sample of every course (2026-09-26)

`gpt-oss-120b` on Groq, seed 0, `--modules 3 --min-los 20 --max-repairs 4`, prompt `poc-3`
(order set from the model's prerequisites). Generated with `scripts/summarize_runs.py --latest`.

| Course | LOs | CSV modules / units | Model modules | Coverage | Checks | Attempts | BCubed F1 (module) | Module recall | Unit precision | Module ARI | Order agreement | LO prereq links | Merges |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| AI_Practitioner | 23 | 7 / 2 | 6 | 23/23 | pass | 4 | 0.6 | 0.533 | 0.577 | 0.282 | 0.54 | 23 | 4 |
| CloudAdmin | 21 | 5 / 1 | 5 | 21/21 | pass | 2 | 0.549 | 0.345 | 1.0 | 0.239 | 0.659 | 19 | 2 |
| CloudDevOps | 23 | 3 / 1 | 6 | 23/23 | pass | 2 | 0.507 | 0.261 | 1.0 | 0.187 | 0.527 | 21 | 1 |
| CloudNative | 44 | 3 / 1 | 5 | 44/44 | pass | 4 | 0.52 | 0.333 | 1.0 | 0.058 | 0.497 | 18 | 15 |
| DataEng | 23 | 7 / 3 | 10 | 23/23 | pass | 1 | 0.608 | 0.25 | 0.667 | 0.229 | 0.337 | 20 | 0 |
| PPP | 28 | 5 / 2 | 13 | 28/28 | pass | 1 | 0.578 | 0.23 | 0.892 | 0.244 | 0.468 | 9 | 0 |

**The pipeline works end to end.** All six courses produce a valid plan with every LO placed.
DataEng and PPP pass on the first call; the others need 1–3 repair rounds.

**Order now comes from prerequisites.** The model states 9–23 LO prerequisites per course, and
every module dependency is backed by them. Module graphs have 2–4 topological steps.

**Grouping improved; order agreement with the CSV did not.**

- BCubed F1 against CSV modules is 0.51–0.61 for every course, up from 0.20–0.78 with prompt
  `poc-2`, where module sizes were erratic (one 23-LO module, or modules of about 3 LOs).
- Order agreement with the CSV is 0.34–0.66 (random is 0.5). The CSV order is one valid order,
  and without labeled prerequisites we cannot tell whether the model's order is wrong or just
  different. Unit-level prerequisite labels (see [docs/evaluation.md](docs/evaluation.md)) are
  the next step.
- PPP still has few prerequisites (9 links, 11 of 13 modules in the first step).
- CloudAdmin, CloudDevOps and CloudNative samples fall inside one CSV unit, so unit precision is
  1.0 by construction there; it only means something for AI_Practitioner, DataEng and PPP.
- CloudNative's 15 merges match its exact-duplicate LOs (15 groups in the data profile).

**Prompt `poc-2` → `poc-3`** (same sample and settings):

| Course | BCubed F1 poc-2 → poc-3 | Order agreement poc-2 → poc-3 | LO prereq links poc-2 → poc-3 |
|---|---|---|---|
| AI_Practitioner | 0.468 → 0.600 | 0.543 → 0.540 | 18 → 23 |
| CloudAdmin | 0.642 → 0.549 | 0.865 → 0.659 | 1 → 19 |
| CloudDevOps | 0.778 → 0.507 | – → 0.527 | 0 → 21 |
| CloudNative | 0.204 → 0.520 | 0.366 → 0.497 | 11 → 18 |
| DataEng | 0.390 → 0.608 | 0.584 → 0.337 | 4 → 20 |
| PPP | 0.335 → 0.578 | 0.434 → 0.468 | 0 → 9 |

CloudDevOps's 0.778 under `poc-2` came from putting all 23 LOs in one module, which scores high
only because one CSV module holds 18 of them.

**Fixes made during these runs:**

- Repair patches originally replaced the whole module list. The model sent only the changed
  module and wiped the rest (CloudNative). Modules are now patched by id like LOs.
- Schema errors named list positions (`los.25.verb`), which the model cannot map to an LO; they
  now name the LO id. A prerequisite pointing at an LO that was merged away now names the LO it
  was merged into.
- An LO listing itself in its own `merged_ids` produced a schema error about `source_ids`, a
  field the model never sees; its repair then deleted the LO. `assemble()` now drops such
  self-references (and repeated ids) and logs a note.

### Earlier: DataEng, all 57 LOs, first prompt (2026-09-23)

The first prompt had the model list module members separately. 21 of 57 LOs were left out of
every module, mostly C2 "describe / explain / explore" LOs from CONCEPT and PRIMER modules. With
each LO naming its module (prompt `poc-2`), the same 57 LOs were all placed and passed every check
on the first call.

## Differences from the plan

- **Model.** `CLAUDE.md` names `llama-3.3-70b-versatile` on Groq. Groq no longer serves it to this
  account; the default is `openai/gpt-oss-120b`.
- **Test courses.** The plan names PPP and a Udacity nanodegree. The test set is the six SAIL
  courses above.
- **PPP size.** The plan describes PPP as 129 LOs across 8 modules and 8 projects; the CSV has 202
  detailed and 22 syllabus LOs across 10 units. To confirm which version is the reference.
- **One call, not a chain.** The plan has six steps with a tool check after each. The proof of
  concept does steps 1–5 in one call and code runs the repair loop. Step 6 follows the plan:
  `topo_sort_modules` sets the order from the model's prerequisites and the model only chooses
  among valid orders, with a short rationale.
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

130 tests: schemas (22), ingestion (22), ground truth and input leak checks (11), tools (32), LLM
client (19), Agent 1 loop, metrics and drawing (22), scaffold (2). LLM tests replay scripted replies through
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
9. Amigó, E., Gonzalo, J., Artiles, J., & Verdejo, F. (2009). A comparison of extrinsic
   clustering evaluation metrics based on formal constraints. *Information Retrieval*, 12(4),
   461–486.
10. Lapata, M. (2006). Automatic evaluation of information ordering: Kendall's tau.
    *Computational Linguistics*, 32(4), 471–484.

[3], [4] and [6] are background. EduPlanner [3] and ISD-Agent-Bench [4] are related
instructional-design agent work that operates at the lesson and content level. Rubrics as
Rewards [6] is the planned pattern for Agent 2's soft-quality rubric. [7]–[10] are standard
references not in the literature survey. The evaluation framework's own references are in
[docs/evaluation.md](docs/evaluation.md).
