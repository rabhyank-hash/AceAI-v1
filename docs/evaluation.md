# Agent 1 evaluation

How Agent 1 v3's output (the LOs in teaching order, split into modules) is evaluated: the metrics,
what is implemented, and the labels still needed. Metric choices follow the literature cited at
the end. The test that applies them is in [v3_test_design.md](v3_test_design.md).

Agent 1 merges identical LOs (code), orders the LOs, splits the order into modules, and places
detailed LOs under course-level LOs (containment). The identical-LO merge is deterministic and
needs no evaluation. Bloom level and the other normalization fields belong to Agent 2 and are
evaluated there.

## Levels

| Level | Question | Metric | Reference data | Status |
|---|---|---|---|---|
| 0. Validity | Is the output a usable course? | checks passed; coverage (LOs placed / LOs in); asks excluded | none | implemented |
| 1. Consistency | Do independent runs give the same course? | sequence agreement and module ARI between runs | none | implemented |
| 2. Modules | Do the modules match the authors'? | BCubed P / R / F1 [1]; ARI [2] | CSV modules | implemented |
| 3. Order | Is the teaching order right? | sequence agreement with the CSV (Kendall's τ rescaled) [3]; violations of labeled module-order pairs | CSV order; labeled pairs | agreement implemented; violations need labels |
| 4. Containment | Are detailed LOs under the right course-level LO? | parent-edge P / R / F1 [4] | labeled syllabus LO → module map (PPP) | needs labels |
| 5. Expert review | Would a designer accept the course? | checklist rubric scored by humans; inter-rater agreement | 2–3 raters | not started |

Baselines: a random order with the same module sizes; the CSV itself; a single ask with no
repetition (k = 1); embeddings + clustering (plan §6); a graph-based curriculum-sequencing method
[5]; a single-pass model, as in [6, 7].

## Level details

### 0. Validity

`validate_output` (schema, references, every LO in exactly one module) and `check_provenance`
(every input LO maps to exactly one output LO) in `src/aceai/tools/`, plus the answer checks in
`agents/sequencer_v3.py`: each order answer contains every LO exactly once, and each split answer
keeps the given order. A run is **invalid** if any LO is left out.

### 1. Consistency

The primary criterion. A course structure that changes with the order its LOs arrive in cannot be
trusted or evaluated. Between independent runs:

- **Sequence agreement:** share of LO pairs ordered the same way (0.5 = random).
- **Module ARI** [2]: 1 for identical modules, about 0 for chance. BCubed F1 between runs is
  reported alongside.

### 2. Modules

- **BCubed** (primary). For each LO, precision is the share of its module that shares its CSV
  module, and recall the share of its CSV module that shares its module; both are averaged over
  LOs. Of the extrinsic clustering metrics Amigó et al. [1] compared, only BCubed satisfies all
  four of their formal constraints; pairwise counts let one large module dominate.
- **ARI** [2], chance-corrected.

The CSV modules are one valid split, not the only one; a finer or coarser split lowers the score
without being wrong.

### 3. Order

- **Sequence agreement with the CSV:** Kendall's τ rescaled to [0, 1] over all LO pairs; Lapata
  [3] validated τ against human judgments of ordering quality.
- **Limitation:** the CSV order is one valid order. Teaching two independent topics in the other
  order lowers agreement without being wrong.
- **Violations of labeled module-order pairs** (needs labels): for each pair of CSV modules a
  person marks "A before B", "B before A" or "either". The score is the share of required pairs
  the output orders the wrong way; "either" pairs are not scored, so every valid order scores 0.

### 4. Containment (PPP only)

For each of PPP's 22 syllabus LOs, label the CSV modules it covers. Score the output's parent links
as edge P / R / F1 [4].

### 5. Expert review

Checklist items scored independently by humans: modules are coherent; order is teachable; nothing
is missing. Instructional Agents [7] found LLM reviewers gave compressed mid-range scores and could
not separate good materials from weak ones, so an LLM judge is used only after it has been checked
against human scores.

## Implemented now

`src/aceai/eval/compare.py` (scores against the CSV), `scripts/consistency.py` (between runs),
`scripts/analyze_experiments.py` (tables over an experiment log).

## Labels needed

1. Module-order pairs for the test samples (Level 3): 9 pairs for the three test courses.
   `scripts/make_label_sheets.py` writes the sheets to `annotations/`.
2. PPP syllabus LO → module map (Level 4).

## References

1. Amigó, E., Gonzalo, J., Artiles, J., & Verdejo, F. (2009). A comparison of extrinsic
   clustering evaluation metrics based on formal constraints. *Information Retrieval*, 12(4),
   461–486.
2. Hubert, L., & Arabie, P. (1985). Comparing partitions. *Journal of Classification*, 2(1),
   193–218.
3. Lapata, M. (2006). Automatic evaluation of information ordering: Kendall's tau.
   *Computational Linguistics*, 32(4), 471–484. https://aclanthology.org/J06-4002
4. Taxonomy-generation evaluation: edge F1 and ancestor F1, e.g. HiExpan (arXiv:1910.08194).
5. A hybrid Transformer–Graph framework for curriculum sequencing and prerequisite optimization
   in CS education. *Algorithms* (MDPI), 2026. https://doi.org/10.3390/a19040308
6. Syahputra, M. F., et al. (2026). Development of multi-agent generative pipelines framework
   for learning plan generation with deterministic constraint verification. *EEJET*,
   2/2(140), 17–31. https://doi.org/10.15587/1729-4061.2026.356830
7. Yao et al. (2026). Instructional Agents. *EACL 2026*. arXiv:2508.19611
