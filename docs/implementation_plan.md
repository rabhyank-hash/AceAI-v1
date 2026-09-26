# ACE-AI Training Plan Generation — Approach & Implementation Plan

*Prepared by Ruchi, TEEL Lab / CMU. September 2026.*

> Version 1: the plan as written in the Google Drive document of the same name, copied into the
> repository on 26 September 2026 so that changes to it can be tracked in git. The content is
> unchanged. Implementation status against this plan is in [poc_report.md](poc_report.md).

## 1. The problem

TEEL Lab wants agents that take a set of Learning Objectives (LOs) — for a corporate training
program at Accenture or for an academic course — and turn them into a concrete training plan.

The LOs we receive are a mixture of broad, course-level objectives and detailed, atomic
objectives. The goal is a system that (1) turns that mixed list into a feasible,
dependency-respecting ordering, and (2) turns that ordering into a detailed plan that fits
real-world constraints (time, cost, learner level). Where the constraints make full coverage
impossible, the system makes deliberate, legible trade-offs (teach an objective at reduced depth,
merge a few adjacent objectives, shorten a module) rather than ignoring the constraints or
silently dropping objectives.

**Scope.** This system works at the level of LOs, modules, time, schedule, delivery, and grouping.
Selecting or generating individual learning activities (lectures, labs, discussions, etc.) is
**not** part of this system. Activity schema and strategy/method/technique taxonomy remain
relevant as a possible downstream stage, but nothing in this plan depends on them.

**Research framing.** For now, constraints are fixed for each run: the question is how well a
plan can be reconciled against a given constraint set. Constraints changing mid-program (drift,
and when to repair vs. regenerate a plan) is a potential future direction, not part of the
current build.

## 2. Architecture: two agents, one direction

Both agents are **LLM agents**. The LLM makes the judgment calls, and each agent calls
deterministic **code tools** for the mechanical checks that should never depend on model
judgment (schema validation, graph checks, constraint arithmetic). The model and agent framework
are not chosen yet (§8).

**Agent 1, the Sequencer**, takes the raw, mixed-granularity LO list and produces a coherent,
deduplicated, dependency-ordered tree: modules in a fixed order, each headed by a course-level
objective where one applies, containing the detailed objectives it covers. Agent 1 never sees
time, cost, or learner-level constraints. Its only job is deciding what needs to be taught and in
what order, which is a question about the logical structure of the content, not about the budget.

**Agent 2, the Constraint Optimizer**, takes that fixed tree plus the constraints and produces
the detailed plan:

- **Time allocation** — hours (or weeks) per module and per LO, fitted to the total duration.
- **Schedule / timeline** — which modules and LOs land in which week or session.
- **Delivery & grouping** — for each module: modality (online sync, online async, in person,
  hybrid), grouping (individual, small group, cohort), and session cadence.
- **Depth / scope decisions** — which LOs are taught at reduced depth or merged with adjacent LOs
  to fit the constraints, each one logged.

The handoff is deliberately one-directional: Agent 2 can annotate, allocate, schedule, and
compress within Agent 1's tree, but it can never reorder it. Prerequisite structure is a fact
about the content (objective B needs objective A first regardless of budget), while depth, time,
and delivery are resource-allocation decisions that depend on the constraints. Keeping them
separate also means each agent can be evaluated independently: Agent 1 purely on whether its
sequence is coherent and dependency-respecting; Agent 2 purely on whether its trade-offs were
good, given a skeleton that is already known to be structurally sound.

One rule keeps this honest: Agent 2 may only compress *contiguous* spans of the sequence. Merging
two objectives that Agent 1 placed far apart would be a structural edit disguised as constraint
optimization. (**Open:** whether a contiguous span may cross a module boundary, and how
contiguity is defined when a module holds parallel conceptual and applied tracks — see §10.)

## 3. Depth scale

Compression ("teach this LO at reduced depth") only makes sense if depth is ordered. I am
proposing that every LO carries a **Bloom's Revised Taxonomy level (C1 Remember → C6 Create)**,
inferred from its action verb during normalization and reviewable by a human. A downgrade moves
an LO down this scale (e.g. C3 Apply → C2 Understand), never below its depth floor.

This deliberately does *not* reuse the Activity schema's learning-goal field (knowledge
acquisition, automaticity, application, higher-order thinking, reflection, transfer): those
values describe types of learning outcome and have no natural ordering, and activities are out of
scope for this system anyway.

**Open:** confirm Bloom C1–C6 as the scale, and decide how ambiguous verbs (e.g. "discuss", "use")
are resolved.

## 4. Agent 1 in detail: from a mixed LO list to a tree

Real LOs arrive at several levels of granularity in the same document: broad course-level goals,
mid-level conceptual objectives, and fine-grained atomic objectives. A broad LO like "explain the
core building blocks of a programming language, including variables and functions" is not a peer
of the atomic objectives next to it — it is a summary that several of them together satisfy.
Grouping it by topic similarity is the wrong move; what is needed is a containment judgment: do
these detailed LOs together cover this broad one, so they belong under it?

**Decision:** a course-level LO becomes the **parent node** of the detailed LOs it contains — it
is not discarded.

Agent 1 is an LLM agent that works through these steps. The LLM makes every judgment; code tools
do the mechanical checks.

1. **Normalize (LLM)** every LO into a structured form: action verb, Bloom level (§3), target
   concept, track (conceptual — "explain/describe X" vs. applied — "write/implement X"), and
   scope (atomic, or aggregate/course-level). *Tool:* schema validation of each normalized LO.
2. **Detect containment (LLM)**: for each aggregate LO, decide which atomic LOs it covers; these
   become its children.
3. **Deduplicate (LLM)** atomic LOs, keeping provenance links back to every original LO. Proposed
   rule: only collapse LOs that share both track and Bloom level — "explain iteration"
   (conceptual) and "use iteration to solve a problem" (applied) look similar but are different
   coverage and both survive. (**Open:** confirm this combined rule.) *Tool:* provenance check
   that every raw LO maps to exactly one surviving LO, so nothing is lost.
4. **Infer prerequisites (LLM)** between the deduplicated atomic LOs, each edge with a one-line
   rationale. *Tool:* cycle check on the prerequisite graph; if a cycle is found, it goes back to
   the agent to resolve.
5. **Group into modules (LLM)** by topic, anchored on aggregate LOs where they exist, ordered
   internally by increasing Bloom level. *Tool:* check that no module requires its own later
   content.
6. **Order modules (tool)**: a topological sort of the module dependency graph produces the final
   order; where the graph allows several valid orders, the agent picks one and states why.

The output is a tree: module nodes (headed by a course-level LO where one applies) containing
their atomic LOs, with a record of which original raw LOs fed into each deduplicated one, so
nothing is silently lost.

**Test cases.** Two real, human-built module structures serve as ground truth:

- **Practical Programming with Python (PPP)** — the full syllabus: 129 LOs at mixed levels (5
  course goals, 9 conceptual LOs, 8 project-level LOs, 50 module LOs, 57 per-project LOs) across 8
  modules and 8 projects, with 8-week and 15-week schedule options. First experiment.
- **Udacity AI Programming with Python nanodegree** — second test case, to check Agent 1 isn't
  tuned to one course.

For each, the LOs are flattened and shuffled before being fed in, and the question is whether
Agent 1 recovers something close to the human module structure.

**Working assumption on scale:** Accenture LO sets will be course-sized, similar to PPP (on the
order of 100–150 LOs). At that size the agent can reason over the whole list at once, so Agent 1
uses no embedding or retrieval step. If real LO sets turn out to be much larger (e.g.
catalog-scale), an embedding tool that shortlists candidate duplicates and containment pairs for
the agent to judge is the first thing to add.

## 5. Agent 2 in detail: reconciling the plan against constraints

Agent 2 has four responsibilities:

- **Time allocation** — distribute the total available time across modules and LOs, weighted by
  Bloom level (higher levels generally need more time) and by LO count.
- **Scheduling** — lay modules and LOs onto the timeline (weeks/sessions), preserving Agent 1's
  order, and place checkpoints sensibly (e.g. not before the content they assess).
- **Delivery & grouping** — choose modality, grouping, and cadence per module, conditioned on
  learner level, cohort size, modality preference, and cost.
- **Compression (time cost)** — when time or cost cannot support full coverage, take a contiguous
  span and either merge LOs into a single denser unit or downgrade an LO's Bloom level. Nothing is
  ever silently dropped: every compression is logged (which LOs, level before/after, rationale) so
  a human reviewer can see and veto it.

**Mechanism: an LLM agent with a deterministic checker tool, not reinforcement learning.** This
is a deliberate, data-driven choice: we have a small number of fully worked courses and a couple
of hand-written expert critiques, nowhere near enough labeled "given these constraints, here is
the right trade-off" examples to train a policy. An agent that iterates against a checker needs no
training data:

1. **Draft** — the agent proposes a first-pass plan (allocation, schedule, delivery/grouping, no
   compression).
2. **Check (tool)** — the agent calls the checker, which returns pass/fail and margin per hard
   constraint, plus violations of the structural rules: no reordering of Agent 1's tree,
   compression only on contiguous spans, no LO dropped, no LO below its depth floor.
3. **Revise** — the agent edits the plan to fix what the checker reported (compress a span,
   reallocate hours, change delivery/grouping), logging every compression with its rationale.
4. **Repeat** until the checker passes, or an iteration cap is hit — at which point the plan goes
   to a human reviewer rather than being forced.

**Open:** whether soft quality (the rubric below) also feeds the loop — through the agent
critiquing its own plan, or a separate critic call — or is only used to evaluate final plans.

**Rubric v1 (hand-derived, scored as independent checklist items; where possible, items are
computed by code rather than judged by an LLM):** coverage retained (how much Bloom depth was
given up, and where); pacing (time per LO reasonable for its Bloom level); cognitive progression
(Bloom level builds sensibly across the timeline); checkpoint timing; delivery/grouping fit to
learner level and Bloom level; and cost-scaling risk (e.g. small-group formats multiplied across
a large cohort). Grounded where possible in TEEL's worked critiques of real course schedules. RL
remains a stretch goal, pursued only if the agent shows failure modes a learned policy would fix.

### Modelling cost and time (Open — options under discussion)

The deterministic checker needs computable time and cost. Options I am considering:

- **Option A — Contact-hours model.** Each module gets learner hours and instructor hours.
  Instructor hours are multiplied by grouping (cohort = 1×; small group = number of groups;
  individual = number of learners) and by a delivery rate (in person > online sync > async).
  Budget = total weighted instructor hours (or a dollar rate on top). Most precise; needs rates
  we would have to agree on.
- **Option B — Ordinal cost points.** A small lookup table assigns cost points to each (delivery,
  grouping) combination per hour; budget is a points ceiling. Cheap to set up, still
  deterministic, less realistic.
- **Option C — Time first, cost later.** Implement only duration constraints (total weeks ×
  learner hours/week) in the first version; add cost once the time-only loop works.

## 6. How we'll know it's working

Two separate evaluations, matching the two agents.

**Agent 1** — compared against the human-built module structures of PPP and the Udacity
nanodegree (module assignment agreement, order agreement, dependency violations). Baselines: (a)
embeddings + clustering — group LOs into modules by topic similarity with no LLM judgment, then
order them by a topological sort over the same prerequisite graph — which Agent 1 must beat to
show that LLM judgment adds value; (b) a graph-based curriculum-sequencing method from the
literature, as a further non-LLM comparison.

**Agent 2** — given the *human* module structure (not Agent 1's output, so the two evaluations
don't confound each other), run constraint scenarios:

- Compress PPP's 15-week schedule to 8 weeks (the syllabus's own 8-week version is a useful
  reference point).
- Same content for a beginner vs. an advanced cohort.
- A reduced cost/staffing budget.

Compare the agent against three baselines it must beat to justify its complexity: naive
proportional time trimming, greedy trimming by lowest priority, and a single-shot LLM given every
constraint at once with no checker tool and no iteration. If a cheap baseline does just as well,
that is a useful finding, not a failure.

**Open — possible additions from the literature survey:** an ablation of Agent 2's operations
(compression / allocation / scheduling / delivery-grouping) in the style of Syahputra et al.; a
generalization test on synthetic (LOs, constraints) pairs; a small expert-rating pass using
TEEL's rubric language.

## 7. Implementation roadmap

**Phase 0 — Schema.** Formalize the Learning Objective, Module, Constraint, and Training Plan
schemas, the LO-coverage map, and the depth scale (see *Schema Concepts v1*). Pure design work, no
infrastructure, and the first thing to review.

**Phase 1 — Agent 1.** Build Agent 1's code tools (schema validation, provenance check, cycle
check, module-order check, topological sort), then the agent's prompts for normalization (incl.
Bloom level), containment, deduplication, prerequisites, and module grouping; validate against
PPP, then Udacity.

**Phase 2 — Agent 2's checker tool.** The deterministic constraint and structural-rule checker
(once the cost/time model is chosen), built and tested on its own before any agent uses it.

**Phase 3 — Agent 2.** The agent's prompts for allocation, scheduling, delivery/grouping, and
contiguous-span compression with mandatory logging; the draft → check → revise loop with an
iteration cap; rubric v1.

**Phase 4 — Evaluation.** Constraint scenarios vs. baselines (§6).

**Future.** Constraint drift / plan repair; learned (RL) reconciliation if justified by Phase 4;
connecting to a downstream activity-selection stage.

## 8. Platform and infrastructure (deferred)

The platform is not being decided yet; it will be settled once the schema (Phase 0) and the
Agent 1 design are stable. The LLM and the agent framework are also not chosen yet. Whatever we
choose must provide: access to a capable LLM with tool calling (both agents run on it), storage
for LO/Module/Constraint/Plan data and the prerequisite graph, orchestration for a multi-step
pipeline with an iterative loop and human-review pause points, a place to log every compression
decision for review, and cost tracking for repeated model calls. A candidate AWS/Bedrock stack
was sketched earlier and will be revisited then.

## 9. Data

Through Phase 4, nothing in this plan uses real Accenture data: every phase runs on the PPP
syllabus, the Udacity course materials, and TEEL Lab's own existing materials, so there is no new
data-sharing or compliance question to resolve before work starts. Phase 0 needs no
infrastructure and is the first checkpoint — cheap to revise now, expensive once Phases 1–3 are
built on it.

## 10. Open questions (summary)

1. Confirm Bloom C1–C6 as the depth scale; resolution of ambiguous verbs.
2. Deduplication rule: same track *and* same Bloom level?
3. Contiguous span: may it cross module boundaries? How is contiguity defined across parallel
   conceptual/applied tracks?
4. Cost/time model: Option A, B, or C (§5).
5. Should compression aggressiveness depend on learner level (beginners likely tolerate less)?
6. Which literature-survey evaluation additions stay in scope (§6)?
7. Platform, LLM, and agent framework — deferred until after Phase 0 / Agent 1 design.
8. How PPP relates to the AI Technicians Python course in TEEL's earlier slides — unknown for now.
9. Does soft quality (rubric) feed Agent 2's loop — via self-critique or a separate critic — or
   only evaluate final plans?
10. Scale assumption: Accenture LO sets are course-sized like PPP — to be confirmed once real sets
    are available.
