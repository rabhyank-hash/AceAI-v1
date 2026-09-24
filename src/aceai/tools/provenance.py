"""Tool 2: every raw LO maps to exactly one surviving LO."""

from __future__ import annotations

from collections.abc import Iterable

from pydantic import BaseModel

from aceai.schemas import SequencerOutput
from aceai.tools.results import IssueLog, ToolResult


class ProvenanceResult(ToolResult):
    missing: list[str] = []  # raw ids not mapped to any LO
    duplicated: dict[str, list[str]] = {}  # raw id -> the several LO ids it maps to
    unknown_raw: list[str] = []  # raw ids in the provenance that are not in the input
    unknown_lo: list[str] = []  # LO ids in the provenance that are not in output.los


def _raw_id(item: str | BaseModel) -> str:
    """Accept plain ids, RawLO (`raw_id`), or Agent 1 InputLO (`id`)."""
    if isinstance(item, str):
        return item
    for attr in ("raw_id", "id"):
        if hasattr(item, attr):
            return getattr(item, attr)
    raise TypeError(f"cannot get a raw id from {type(item).__name__}")


def check_provenance(
    raw_los: Iterable[str | BaseModel], output: SequencerOutput
) -> ProvenanceResult:
    """Check `output.provenance` against the input ids.

    Also checks consistency with each LO's `source_ids`: the provenance entry raw -> L must agree
    with L listing raw in its source_ids, and vice versa.
    """
    log = IssueLog()
    raw_ids = sorted({_raw_id(x) for x in raw_los})
    raw_set = set(raw_ids)
    lo_by_id = {lo.id: lo for lo in output.los}
    mapping = output.provenance_map()

    missing = [r for r in raw_ids if r not in mapping]
    duplicated = {r: los for r, los in sorted(mapping.items()) if len(los) > 1}
    unknown_raw = sorted(r for r in mapping if r not in raw_set)
    unknown_lo = sorted({e.lo_id for e in output.provenance if e.lo_id not in lo_by_id})

    if missing:
        log.error(
            "missing",
            f"{len(missing)} input LOs are not mapped to any surviving LO; map each one to the "
            "LO that now covers it.",
            missing,
        )
    for r, los in duplicated.items():
        log.error(
            "duplicated",
            f"Input LO {r} is mapped to {len(los)} LOs {los}; it must map to exactly one.",
            [r, *los],
        )
    if unknown_raw:
        log.error("unknown_raw", "Provenance mentions ids that are not input LOs.", unknown_raw)
    if unknown_lo:
        log.error("unknown_lo", "Provenance points to LOs that are not in the output.", unknown_lo)

    # Cross-check with source_ids
    for r, los in sorted(mapping.items()):
        for lid in los:
            lo = lo_by_id.get(lid)
            if lo is not None and r not in lo.source_ids:
                log.error(
                    "source_ids_mismatch",
                    f"Provenance maps {r} to {lid}, but {lid}.source_ids does not list {r}.",
                    [r, lid],
                )
    for lo in output.los:
        for r in lo.source_ids:
            if lo.id not in mapping.get(r, []):
                log.error(
                    "source_ids_mismatch",
                    f"{lo.id}.source_ids lists {r}, but provenance does not map {r} to {lo.id}.",
                    [r, lo.id],
                )

    return log.finish(
        ProvenanceResult(
            missing=missing, duplicated=duplicated, unknown_raw=unknown_raw, unknown_lo=unknown_lo
        )
    )
