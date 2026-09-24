"""Rough POC comparison of an Agent 1 output with the human (CSV) structure.

Not the plan's evaluation metrics. Just enough to see whether a model is in the right ballpark:

- grouping: pairwise co-membership precision / recall / F1. A pair of detailed LOs is "together"
  if they share a module (predicted) or a CSV module (ground truth).
- order: over LO pairs that are in different modules in both, the share the model puts in the
  same relative order as the CSV. 0.5 is what a random module order would get.
- merges: which raw LOs were collapsed together, and whether they came from the same CSV module.

Everything is keyed by raw id; the output is in Agent 1 input-id space and is mapped back with
`id_map`. Syllabus broad LOs have no CSV module, so they are reported separately.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any

from aceai.ingest.ground_truth import GroundTruth
from aceai.schemas import SequencerOutput


def _gt_positions(gt: GroundTruth) -> tuple[dict[str, int], list[str]]:
    """raw id -> global CSV module index, plus a label per module index."""
    pos, labels = {}, []
    for unit in gt.units:
        for m in unit.modules:
            for rid in m.lo_ids:
                pos[rid] = len(labels)
            labels.append(f"u{unit.unit_no} {m.module_type.value} {m.module_name.strip()}")
    return pos, labels


def predicted_positions(output: SequencerOutput, id_map: dict[str, str]) -> dict[str, int]:
    """raw id -> index of the predicted module holding the LO it survives as."""
    module_of: dict[str, int] = {}
    for i, m in enumerate(sorted(output.modules, key=lambda m: m.order)):
        for lid in m.lo_ids:
            module_of.setdefault(lid, i)
        if m.aggregate_lo_id:
            module_of.setdefault(m.aggregate_lo_id, i)
    out = {}
    for e in output.provenance:
        rid = id_map.get(e.raw_id)
        if rid is not None and e.lo_id in module_of:
            out.setdefault(rid, module_of[e.lo_id])
    return out


def _prf(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {
        "precision": round(p, 3),
        "recall": round(r, 3),
        "f1": round(f, 3),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def compare(output: SequencerOutput, id_map: dict[str, str], gt: GroundTruth) -> dict[str, Any]:
    gt_pos, gt_labels = _gt_positions(gt)
    pred_pos = predicted_positions(output, id_map)
    detailed = sorted(r for r in gt_pos if r in pred_pos)
    unplaced = sorted(r for r in gt_pos if r not in pred_pos)

    tp = fp = fn = 0
    agree = disagree = 0
    for a, b in combinations(detailed, 2):
        same_gt = gt_pos[a] == gt_pos[b]
        same_pred = pred_pos[a] == pred_pos[b]
        tp += same_gt and same_pred
        fp += same_pred and not same_gt
        fn += same_gt and not same_pred
        if not same_gt and not same_pred:
            if (gt_pos[a] < gt_pos[b]) == (pred_pos[a] < pred_pos[b]):
                agree += 1
            else:
                disagree += 1

    # Merges: surviving LOs with more than one raw source.
    inv = {v: k for k, v in id_map.items()}  # raw -> input id (for display)
    by_lo: dict[str, list[str]] = {}
    for e in output.provenance:
        if e.raw_id in id_map:
            by_lo.setdefault(e.lo_id, []).append(id_map[e.raw_id])
    merges = []
    for lo_id, raws in sorted(by_lo.items()):
        if len(raws) > 1:
            mods = sorted({gt_labels[gt_pos[r]] if r in gt_pos else "syllabus" for r in raws})
            merges.append(
                {
                    "lo_id": lo_id,
                    "raw_ids": sorted(raws),
                    "gt_modules": mods,
                    "same_gt_module": len(mods) == 1,
                }
            )

    # Broad (syllabus) LOs: how many children the model attached.
    los = {lo.id: lo for lo in output.los}
    children: dict[str, int] = {}
    for lo in output.los:
        if lo.parent_id:
            children[lo.parent_id] = children.get(lo.parent_id, 0) + 1
    broad = []
    for level, rids in gt.broad_lo_ids.items():
        for rid in rids:
            lid = next((e.lo_id for e in output.provenance if id_map.get(e.raw_id) == rid), None)
            lo = los.get(lid) if lid else None
            broad.append(
                {
                    "raw_id": rid,
                    "level": level.value,
                    "lo_id": lid,
                    "scope": lo.scope if lo else None,
                    "children": children.get(lid, 0),
                }
            )

    return {
        "n_detailed": len(gt_pos),
        "n_detailed_placed": len(detailed),
        "unplaced": [{"raw_id": r, "input_id": inv.get(r)} for r in unplaced],
        "n_gt_modules": len({gt_pos[r] for r in gt_pos}),
        "n_pred_modules": len(output.modules),
        "n_surviving_los": len(output.los),
        "grouping": _prf(tp, fp, fn),
        "order": {
            "pairs": agree + disagree,
            "agreement": round(agree / (agree + disagree), 3) if agree + disagree else None,
        },
        "merges": merges,
        "broad": broad,
    }
