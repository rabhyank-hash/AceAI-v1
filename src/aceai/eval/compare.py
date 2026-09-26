"""Rough POC comparison of an Agent 1 output with the human (CSV) structure.

Not the plan's evaluation metrics. Just enough to see whether a model is in the right ballpark:

- grouping: BCubed precision / recall / F1 (per-LO averages; the primary grouping score),
  pairwise co-membership precision / recall / F1, and the adjusted Rand index (ARI:
  0 = chance, 1 = identical partitions). A pair of detailed LOs is "together" if they share a
  predicted module, or a CSV module / CSV unit (ground truth).
- order: over LO pairs that are apart in both, the share the model puts in the same relative order
  as the CSV. 0.5 is what a random module order would get.

Each is reported at two ground-truth levels. "module" is the CSV module; "unit" is the CSV unit,
which is fairer for topic grouping because a unit's CONCEPT, PRIMER and PROJECT modules share a
topic, so a model that merges them is not wrong. The CSV order is one valid order, not the only
one, so order agreement below 1 is not necessarily an error (dependency violations need labeled
prerequisites, which the CSV does not have).
- merges: which raw LOs were collapsed together, and whether they came from the same CSV module.
- coverage: how many detailed LOs ended up in a module. Agent 1 must place every LO, so anything
  below 100% makes the run invalid. Unplaced LOs are not dropped from scoring: for grouping each
  counts as a module of its own (lowering recall); order agreement cannot rank them and reports
  how many pairs were skipped.

Everything is keyed by raw id; the output is in Agent 1 input-id space and is mapped back with
`id_map`. Syllabus broad LOs have no CSV module, so they are reported separately.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any

from aceai.ingest.ground_truth import GroundTruth
from aceai.schemas import SequencerOutput


def _gt_positions(gt: GroundTruth) -> tuple[dict[str, int], dict[str, int], list[str]]:
    """raw id -> global CSV module index, raw id -> CSV unit index, and a label per module."""
    mod_pos, unit_pos, labels = {}, {}, []
    for u_idx, unit in enumerate(gt.units):
        for m in unit.modules:
            for rid in m.lo_ids:
                mod_pos[rid] = len(labels)
                unit_pos[rid] = u_idx
            labels.append(f"u{unit.unit_no} {m.module_type.value} {m.module_name.strip()}")
    return mod_pos, unit_pos, labels


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


def _pairs(n: int) -> int:
    return n * (n - 1) // 2


def adjusted_rand_index(a: list[int], b: list[int]) -> float:
    """ARI of two labelings of the same items (Hubert & Arabie). 1 = identical, ~0 = chance."""
    n = len(a)
    if n < 2:
        return 1.0
    cells: dict[tuple[int, int], int] = {}
    rows: dict[int, int] = {}
    cols: dict[int, int] = {}
    for x, y in zip(a, b, strict=True):
        cells[x, y] = cells.get((x, y), 0) + 1
        rows[x] = rows.get(x, 0) + 1
        cols[y] = cols.get(y, 0) + 1
    index = sum(_pairs(c) for c in cells.values())
    sum_rows = sum(_pairs(c) for c in rows.values())
    sum_cols = sum(_pairs(c) for c in cols.values())
    expected = sum_rows * sum_cols / _pairs(n)
    best = (sum_rows + sum_cols) / 2
    if best == expected:  # both labelings trivial (all together or all apart)
        return 1.0
    return round((index - expected) / (best - expected), 3)


def bcubed(truth: list[int], pred: list[int]) -> dict[str, float]:
    """BCubed precision / recall / F1 (Amigó et al., 2009), averaged over items. For each item:
    precision = share of its predicted module that shares its true group; recall = share of its
    true group that shares its predicted module. Unlike pairwise counts, large modules do not
    dominate."""
    n = len(truth)
    if n == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    p_sum = r_sum = 0.0
    for i in range(n):
        same_pred = [j for j in range(n) if pred[j] == pred[i]]
        same_true = [j for j in range(n) if truth[j] == truth[i]]
        both = sum(1 for j in same_pred if truth[j] == truth[i])
        p_sum += both / len(same_pred)
        r_sum += both / len(same_true)
    p, r = p_sum / n, r_sum / n
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": round(p, 3), "recall": round(r, 3), "f1": round(f, 3)}


def _score_level(
    ids: list[str], gt: dict[str, int], pred: dict[str, int], placed: set[str]
) -> tuple[dict, dict]:
    """Grouping and order scores of `pred` against one ground-truth level. `pred` must label every
    id; unplaced ids carry a label of their own, and are skipped for order."""
    tp = fp = fn = agree = disagree = skipped = 0
    for a, b in combinations(ids, 2):
        same_gt, same_pred = gt[a] == gt[b], pred[a] == pred[b]
        tp += same_gt and same_pred
        fp += same_pred and not same_gt
        fn += same_gt and not same_pred
        if not same_gt and not same_pred:
            if a not in placed or b not in placed:
                skipped += 1
            elif (gt[a] < gt[b]) == (pred[a] < pred[b]):
                agree += 1
            else:
                disagree += 1
    grouping = _prf(tp, fp, fn)
    grouping["ari"] = adjusted_rand_index([gt[i] for i in ids], [pred[i] for i in ids])
    grouping["bcubed"] = bcubed([gt[i] for i in ids], [pred[i] for i in ids])
    order = {
        "pairs": agree + disagree,
        "agreement": round(agree / (agree + disagree), 3) if agree + disagree else None,
        "skipped_unplaced_pairs": skipped,
    }
    return grouping, order


def compare(output: SequencerOutput, id_map: dict[str, str], gt: GroundTruth) -> dict[str, Any]:
    gt_pos, gt_unit, gt_labels = _gt_positions(gt)
    pred_pos = predicted_positions(output, id_map)
    detailed = sorted(gt_pos)
    placed = {r for r in detailed if r in pred_pos}
    unplaced = [r for r in detailed if r not in placed]
    # Each unplaced LO gets a module label of its own (negative, so it never collides).
    labels = dict(pred_pos)
    labels.update({r: -1 - i for i, r in enumerate(unplaced)})

    grouping, order = {}, {}
    for level, truth in (("module", gt_pos), ("unit", gt_unit)):
        grouping[level], order[level] = _score_level(detailed, truth, labels, placed)

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
        "coverage": {
            "placed": len(placed),
            "total": len(detailed),
            "complete": not unplaced,
        },
        "n_detailed": len(detailed),
        "n_detailed_placed": len(placed),
        "unplaced": [{"raw_id": r, "input_id": inv.get(r)} for r in unplaced],
        "n_gt_modules": len(set(gt_pos.values())),
        "n_gt_units": len(set(gt_unit.values())),
        "n_pred_modules": len(output.modules),
        "n_surviving_los": len(output.los),
        "grouping": grouping,
        "order": order,
        "merges": merges,
        "broad": broad,
    }
