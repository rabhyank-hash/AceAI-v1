# Agent 1 v2 prototype: pre-registration

Written and committed on 26 September 2026, before any v2 run. Plan: [implementation_plan.md](implementation_plan.md)
(v2). Any deviation from this document is listed in the report under "Deviations".

## Question

Does v2's design (consensus grouping, module prerequisites asked repeatedly under different
presentation orders and aggregated by majority, fixed tie-breaking rules) make Agent 1's course
structure consistent across input orders, without making it worse against the authors'
structure?

## Hypotheses

- **H1 (primary, consistency).** Two independent v2 runs on the same LOs agree more with each
  other than v1 runs do.
  - H1a: order agreement between the two v2 runs is **≥ 0.90**.
  - H1b: grouping ARI between the two v2 runs is **≥ 0.60**.
- **H2 (secondary, quality).** v2 is not worse than v1 against the authors' structure: mean BCubed
  F1 against CSV modules of the two v2 runs is at least the v1 mean (over 6 seeds) minus 0.05.
  Order agreement with the CSV is reported but is not a criterion (the CSV order is one valid
  order; module-order labels are not yet available).
- **H3 (validity).** Every v2 output passes all six tools with every LO placed.

## Design

- **Courses:** DataEng and PPP, same samples as v1 (`--modules 3 --min-los 20`: 23 and 28 LOs).
  Chosen before running because they had the lowest v1 order agreement with the CSV (0.41, 0.44)
  and span several CSV units.
- **Model and settings:** `openai/gpt-oss-120b` on Groq, reasoning effort low, temperature 0,
  the same as v1, so that the design is the only change.
- **v1 reference:** v1 runs with seeds 0–5 per course (seeds 0–2 exist; seeds 3–5 are new).
  v1 consistency = mean and range over the 15 seed pairs.
- **v2 runs:** two independent runs per course that share no LLM call.
  - v2-A: grouping from the v1 runs with seeds 0, 1, 2; ordering asks under permutation seeds
    0, 1, 2.
  - v2-B: grouping from the v1 runs with seeds 3, 4, 5; ordering asks under permutation seeds
    3, 4, 5.
- **v2 stages as prototyped:**
  1. Grouping: consensus of the three v1 runs' module assignments (co-association of input LOs,
     average-linkage clustering at threshold 0.5).
  2. Module prerequisites: one ask per target module per permutation seed (3 per module). The
     model sees every module as a shuffled list of its LO texts under shuffled neutral labels and
     returns `required_before` and `better_before`. An edge X → Y is kept when a strict majority
     (≥ 2 of 3) of Y's asks list X in `required_before`.
  3. Order: topological sort; cycles broken by removing the lowest-vote edge; ties broken by
     aggregated preference (`required_before` or `better_before` votes, Copeland score among
     the available modules), then share of conceptual LOs (majority track over grouping runs),
     then smallest LO id.
  4. LO order inside a module: LO prerequisites stated in ≥ 2 of the 3 grouping runs, topological,
     ties by conceptual before applied, then id.
- **Out of scope for the prototype:** deduplication and containment (each input LO is kept as its
  own LO; neither sample has syllabus LOs), a grouping prompt written for v2 (v1's call is reused
  for grouping), and module titles (not scored).

## Metrics

Computed with the existing, unchanged code (`scripts/consistency.py`, `eval/compare.py`):

- Between-run: grouping ARI; order agreement over LO pairs that are in different modules in both
  runs; LO prerequisite-edge Jaccard (reported; v2 keeps only within-module LO edges, so it is
  not comparable and not a criterion).
- Against the CSV: BCubed F1 (modules), order agreement, coverage, tool pass.
- Cost: LLM calls and tokens per v2 run.

## Analysis

Descriptive only. With two courses and one v2 pair per course, no significance test is
meaningful. Every run is reported, including failures. A hypothesis holds only if it holds for
both courses.

## Threats to validity

- n is small: 2 courses, 1 v2 pair each.
- Consensus uses v1's grouping calls, so v2 grouping inherits v1's prompt.
- Order agreement with the CSV cannot separate errors from valid alternatives without labels.
- Temperature 0 and caching: repeated identical prompts give identical answers, so variation
  comes only from presentation order, which is the point of the test.
