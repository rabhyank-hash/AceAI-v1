# Kickoff prompt for Claude Code

Paste the block below into Claude Code, started in the project folder in WSL (e.g. `~/ace-ai`), after
`CLAUDE.md` and the CSVs are in place. It works in steps and stops after each one for review.

---

Read CLAUDE.md fully before doing anything. We are in the "scaffold + Agent 1 code tools" phase.
Work through the steps below in order. After each step: run the tests, show me a short summary of what
changed and the test results, and wait for my go-ahead before starting the next step. If anything in
CLAUDE.md is ambiguous or conflicts with what you find in the data, stop and ask instead of guessing.

**Step 1 — Scaffold.**
Create the `src/` layout, `pyproject.toml` (deps: pydantic>=2, networkx, openai, python-dotenv, pandas;
dev: pytest, ruff), `.gitignore` (include `.env`, `runs/`, `data/processed/`, `.cache/`), `.env.example`,
and a README with setup commands for WSL (venv, install, run tests). Initialize git.
Done when: `pip install -e ".[dev]"` works and `pytest` runs (0 tests is fine).

**Step 2 — Schemas.**
Implement `src/aceai/schemas.py` exactly as specified in CLAUDE.md, including BloomLevel ordering.
Add tests for valid/invalid objects and Bloom comparisons.

**Step 3 — Ingestion + data profile.**
Implement `aceai.ingest` to load every CSV in `data/raw/` into `RawLO` records (handle BOM, CRLF,
quoted commas; normalize whitespace but keep the original text). Generate stable `raw_id`s
(e.g. `ppp-u03-concept-data-structures-lo02`). Then write a script that produces
`data/processed/profile.md` with, per course: LO count by unit and module type, module names with
casing/whitespace variants, exact and near-duplicate LO texts (within a course and across courses),
and candidate logistics-only LOs. Do not remove anything.
Done when: all 6 course CSVs + the PPP syllabus file load, tests cover the CSV quirks, and the profile
exists. Show me the profile summary.

**Step 4 — Ground truth + Agent 1 input builder.**
(a) Extract each course's ground-truth structure (units → modules → raw LO ids) to
`data/processed/ground_truth/<course>.json`.
(b) Build `make_agent1_input(course, seed)` that returns the course's LO texts with raw ids only —
all unit/module/type fields stripped — shuffled with the seed, plus PPP's syllabus broad LOs when
the course is PPP. Test that no ground-truth field leaks into the input.

**Step 5 — Agent 1 tools.**
Implement the six tools in CLAUDE.md under `aceai.tools`, each returning a structured result an LLM
can act on. Tests must include: a cycle (and its reported path), a module depending on a later module,
a missing and a duplicated provenance mapping, an aggregate LO without children, and a module graph
with ties in `topo_sort_modules`. Also run the tools on the PPP ground truth converted into a
`SequencerOutput`-shaped object (no prerequisite edges yet) as a smoke test.

**Step 6 — LLM client.**
Implement `aceai.llm.client` per CLAUDE.md (Groq via OpenAI SDK, config-driven providers, JSON mode,
429 retry with retry-after, on-disk cache). Add a `scripts/llm_smoke.py` that sends one tiny request and
prints the model's reply and the rate-limit headers. Unit-test the retry and cache with a mocked client;
do not call the real API in tests.
Done when: I can run the smoke script with my key in `.env`.

Stop after Step 6. Do not start on Agent 1 prompts.
