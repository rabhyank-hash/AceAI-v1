# Agent 1 evaluation framework

How Agent 1's output (a deduplicated, dependency-ordered module tree) is evaluated, what is
implemented, and what each remaining level needs. Metric choices follow the literature cited at
the end.

Agent 1 does five things, and each is evaluated separately so failures can be traced: it
**normalizes** each LO (Bloom level, track, scope), **deduplicates**, nests detailed LOs under
course-level LOs (**containment**), infers **prerequisites**, and **groups and orders** LOs into
modules. On top of these sits **validity**: the output must be a well-formed tree that covers
every LO.

## Levels

| Level | Question | Metric | Reference data | Status |
|---|---|---|---|---|
| 0. Validity | Is the output a usable tree? | pass rate of the six tools; coverage (LOs placed / LOs in); repair rounds needed | none | implemented |
| 1. Grouping | Do modules match the authors' grouping? | BCubed P / R / F1 [1]; ARI [2]; pairwise P / R | CSV modules and units | implemented |
| 2. Order | Is the teaching order plausible? | pairwise order agreement (Kendall's τ rescaled) [3]; prerequisite violation rate | CSV order; labeled prerequisites | agreement implemented; violations need labels |
| 3. Prerequisites | Are the inferred dependencies right? | edge P / R / F1 [4, 5]; transitive (ancestor) P / R / F1 [6] | labeled prerequisites | needs labels |
| 4. Containment | Are detailed LOs under the right course-level LO? | parent-edge P / R / F1; ancestor F1 [6] | labeled syllabus LO → module map (PPP) | needs labels |
| 5. Deduplication | Are the right LOs merged? | pairwise merge P / R / F1 | labeled duplicate pairs | needs labels |
| 6. Normalization | Are Bloom levels right? | quadratic-weighted Cohen's κ [7] vs human labels; human–human κ as ceiling | Bloom labels on a sample | needs labels |
| 7. Expert review | Would a designer accept the plan? | checklist rubric scored by humans; inter-rater agreement | 2–3 raters | not started |

Across all levels:

- **Stability.** Run 3–5 seeds (different input shuffles). Report mean and spread, and the ARI
  between runs as self-consistency. A single run cannot tell model judgment from noise.
- **Baselines.** Random grouping with the same module sizes; the CSV order itself; embeddings +
  clustering + topological sort (plan §6); a graph-based sequencing method [8]; a single-pass
  model with no tools or repair, as in [9, 10].
- **Ablations.** Remove one component at a time (repair loop, containment, dedup, topological
  ordering), following Syahputra et al. [9].

## Level details

### 0. Validity

The six tools (`src/aceai/tools/`) define validity: schema, provenance, acyclicity, module order.
A run is **invalid** if any LO is left out of every module; its scores are still reported but not
used. Report per course: checks passed (yes/no), coverage, and the number of repair rounds.

### 1. Grouping

Two LOs are "together" if they share a module. The authors' CSV gives two reference groupings:
modules and units. A unit's CONCEPT, PRIMER and PROJECT modules share one topic, so both are
reported.

- **BCubed** (primary). For each LO, precision is the share of its predicted module that shares
  its reference group, and recall is the share of its reference group that shares its predicted
  module; both are averaged over LOs. Amigó et al. [1] compared extrinsic clustering metrics
  against four formal constraints (homogeneity, completeness, rag bag, small-cluster
  preservation), and only BCubed satisfies all four. Pairwise counts, by contrast, are
  quadratic in cluster size, so one large module dominates the score.
- **ARI** [2]: chance-corrected agreement. 0 means random, 1 means identical.
- **Pairwise P / R**, kept for comparison with earlier runs.

Read two numbers together: recall against CSV **modules** (are the authors' modules kept
together?) and precision against CSV **units** (are topics from different units kept apart?).
Both are fair whatever module size the model picks. Unit precision says nothing when the sample
lies within one unit.

### 2. Order

- **Order agreement** (implemented): over LO pairs that are in different modules in both the
  output and the CSV, the share ordered the same way. This is Kendall's τ rescaled to [0, 1] and
  restricted to cross-module pairs. Lapata [3] validated τ against human judgments of ordering
  quality. A random order scores 0.5.
- **Limitation.** The CSV order is one valid order, not the only one. Swapping two independent
  units lowers agreement without being wrong.
- **Prerequisite violation rate** (needs labels): the share of labeled "A before B" pairs that the
  output orders B before A. This checks the order against a partial order instead of a single
  sequence, so every valid order scores zero violations.

### 3. Prerequisites

Standard in prerequisite-relation work (MOOC concepts [4], LectureBank [5]): precision, recall
and F1 over directed edges against expert labels. Because a model may give A → B → C where the
labels give A → C, also report ancestor (transitive-closure) P / R / F1, as used for taxonomy
evaluation [6]. Cheapest labeling: **unit-level prerequisites** (at most about 45 unit pairs per
course), scored after lifting LO edges to units.

### 4. Containment (PPP only)

Label, for each of PPP's 22 syllabus LOs, the CSV modules it covers. Score the model's parent
links as edge P / R / F1 (a detailed LO under the right syllabus LO) and ancestor F1 [6].
Normalized tree edit distance is an option for the whole tree [6], but it is harder to explain
and not needed yet.

### 5. Deduplication

Label candidate pairs (the data profile lists exact and near-duplicates, about 60 pairs) as
same / different under the agreed merge rule. Score merges as pairwise P / R / F1. The labels
depend on open question 2 (the merge rule).

### 6. Normalization (Bloom level)

Bloom levels are ordinal, so use quadratic-weighted Cohen's κ [7]: C2 vs C3 counts as less wrong
than C2 vs C5. Two humans label the same sample of about 50 LOs per course; their κ is the
ceiling the model can be expected to reach. The labels depend on open question 1 (how ambiguous
verbs are resolved).

### 7. Expert review

Checklist items scored independently by humans, not one holistic score. For example: modules are
coherent; order is teachable; no prerequisite is missing; merges are correct. Instructional
Agents [10] found that LLM reviewers gave compressed mid-range scores and could not separate
good course materials from weak ones. An LLM judge is used only after it has been checked
against human scores. Work on LLM-built course knowledge graphs [11] and on prerequisite
prediction [12] likewise relies on expert ratings or expert-defined links as the reference.

## Implemented now

`src/aceai/eval/compare.py`, called by `scripts/run_poc.py`; `scripts/summarize_runs.py` gives
one table across runs.

- Level 0: tool results, coverage, repair rounds.
- Level 1: BCubed, ARI, pairwise, at CSV module and unit level.
- Level 2: order agreement.
- Merges and syllabus LO children are listed but not scored.

## Labels needed, in order of value

1. Unit-level prerequisites for each sampled course (Levels 2 and 3). About 10 minutes per course.
2. Duplicate-pair labels (Level 5), once the merge rule is agreed.
3. Bloom labels on a sample (Level 6), once the verb rules are agreed.
4. PPP syllabus LO → module map (Level 4).

## References

1. Amigó, E., Gonzalo, J., Artiles, J., & Verdejo, F. (2009). A comparison of extrinsic
   clustering evaluation metrics based on formal constraints. *Information Retrieval*, 12(4),
   461–486.
2. Hubert, L., & Arabie, P. (1985). Comparing partitions. *Journal of Classification*, 2(1),
   193–218.
3. Lapata, M. (2006). Automatic evaluation of information ordering: Kendall's tau.
   *Computational Linguistics*, 32(4), 471–484. https://aclanthology.org/J06-4002
4. Pan, L., Li, C., Li, J., & Tang, J. (2017). Prerequisite relation learning for concepts in
   MOOCs. *ACL 2017*. https://aclanthology.org/P17-1133
5. Li, I., Fabbri, A. R., Tung, R. R., & Radev, D. R. (2019). What should I learn first:
   Introducing LectureBank for NLP education and prerequisite chain learning. *AAAI 2019*,
   6674–6681. arXiv:1811.12181
6. Taxonomy-generation evaluation: edge F1, ancestor F1 and normalized tree edit distance, e.g.
   HiExpan (arXiv:1910.08194) and hierarchical catalogue generation (arXiv:2304.03512).
7. Cohen, J. (1968). Weighted kappa: Nominal scale agreement with provision for scaled
   disagreement or partial credit. *Psychological Bulletin*, 70(4), 213–220.
8. A hybrid Transformer–Graph framework for curriculum sequencing and prerequisite optimization
   in CS education. *Algorithms* (MDPI), 2026. https://doi.org/10.3390/a19040308
9. Syahputra, M. F., et al. (2026). Development of multi-agent generative pipelines framework
   for learning plan generation with deterministic constraint verification. *EEJET*,
   2/2(140), 17–31. https://doi.org/10.15587/1729-4061.2026.356830
10. Yao et al. (2026). Instructional Agents. *EACL 2026*. arXiv:2508.19611
11. LLM-powered construction of course knowledge-competency graphs. *ICETAI 2025*.
    https://dl.acm.org/doi/10.1145/3766557.3766569
12. Le, N. L., & Abel, M.-H. (2025). How well do LLMs predict prerequisite skills? Zero-shot
    comparison to expert-defined concepts. arXiv:2507.18479
