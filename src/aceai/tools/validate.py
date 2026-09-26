"""Tool 1: schema and reference validation for LOs and a whole SequencerOutput."""

from __future__ import annotations

from collections import Counter
from typing import Any

from pydantic import ValidationError

from aceai.schemas import LearningObjective, SequencerOutput
from aceai.tools.results import IssueLog, ToolResult


class ValidationResult(ToolResult):
    pass


def _item_id(data: Any, loc: tuple) -> str | None:
    """The `id` of the list item a schema error points into (e.g. ("los", 25, "verb") -> the id
    of los[25]), so the message can name the item instead of a list position."""
    if len(loc) >= 2 and isinstance(data, dict) and isinstance(loc[1], int):
        items = data.get(loc[0])
        if isinstance(items, list) and 0 <= loc[1] < len(items) and isinstance(items[loc[1]], dict):
            item_id = items[loc[1]].get("id")
            return item_id if isinstance(item_id, str) else None
    return None


def _schema_errors(log: IssueLog, exc: ValidationError, what: str, data: Any = None) -> None:
    for err in exc.errors():
        loc = ".".join(str(p) for p in err["loc"]) or what
        item_id = _item_id(data, err["loc"])
        if item_id:
            field = ".".join(str(p) for p in err["loc"][2:]) or "(item)"
            log.error(
                "schema",
                f"{what}: {err['loc'][0]} item {item_id}: {field}: {err['msg']}",
                [item_id],
            )
        else:
            log.error("schema", f"{what}: {loc}: {err['msg']}")


def _parse(model: type[Any], data: Any, log: IssueLog, what: str) -> Any | None:
    if isinstance(data, model):
        return data
    try:
        return model.model_validate(data)
    except ValidationError as exc:
        _schema_errors(log, exc, what, data)
        return None


def _lo_checks(lo: LearningObjective, log: IssueLog) -> None:
    if lo.depth_floor is not None and lo.depth_floor > lo.bloom_level:
        log.error(
            "depth_floor_above_level",
            f"LO {lo.id}: depth_floor {lo.depth_floor} is above bloom_level {lo.bloom_level}.",
            [lo.id],
        )


def validate_lo(lo: LearningObjective | dict[str, Any]) -> ValidationResult:
    """Schema/enum validation of one LO plus checks that need only the LO itself."""
    log = IssueLog()
    parsed = _parse(LearningObjective, lo, log, "LO")
    if parsed is not None:
        _lo_checks(parsed, log)
    return log.finish(ValidationResult())


def validate_output(output: SequencerOutput | dict[str, Any]) -> ValidationResult:
    """Validate a whole Agent 1 output: schema, id uniqueness, references, and tree shape.

    Errors: duplicate ids; unknown parent / dependency / module member / module head ids (a
    dependency on an id that was merged into another LO names the surviving LO); a
    parent that is not aggregate; an aggregate LO with no children; an LO that is its own
    ancestor; an atomic LO in no module; an LO in more than one module; modules not listed in
    ascending, unique `order`.
    Warnings: a module head whose children sit in other modules.
    Cycles, provenance, and ordering have their own tools.
    """
    log = IssueLog()
    out = _parse(SequencerOutput, output, log, "output")
    if out is None:
        return log.finish(ValidationResult())

    for lo in out.los:
        _lo_checks(lo, log)

    lo_ids = [lo.id for lo in out.los]
    mod_ids = [m.id for m in out.modules]
    for kind, ids in (("LO", lo_ids), ("module", mod_ids)):
        for dup, n in sorted(Counter(ids).items()):
            if n > 1:
                code = f"duplicate_{kind.lower()}_id"
                log.error(code, f"{kind} id {dup} is used {n} times.", [dup])
    by_id = {lo.id: lo for lo in out.los}
    # Source ids that no longer exist as LOs because they were merged into another LO.
    merged_into = {src: lo.id for lo in out.los for src in lo.source_ids if src not in by_id}
    mod_set = set(mod_ids)

    # Parents, children, dependencies
    children: dict[str, list[str]] = {}
    for lo in out.los:
        if lo.parent_id is None:
            continue
        parent = by_id.get(lo.parent_id)
        if parent is None:
            log.error(
                "unknown_parent", f"LO {lo.id} has unknown parent_id {lo.parent_id}.", [lo.id]
            )
            continue
        if parent.scope != "aggregate":
            log.error(
                "parent_not_aggregate",
                f"LO {lo.id} has parent {parent.id}, which is atomic; only aggregate LOs "
                "can have children.",
                [lo.id, parent.id],
            )
        children.setdefault(parent.id, []).append(lo.id)
    for lo in out.los:
        if lo.scope == "aggregate" and lo.id not in children:
            log.error(
                "aggregate_without_children",
                f"Aggregate LO {lo.id} has no children; attach the atomic LOs it covers "
                "or mark it atomic.",
                [lo.id],
            )
        for dep in lo.depends_on:
            if dep in by_id:
                continue
            if dep in merged_into:
                log.error(
                    "dependency_on_merged_lo",
                    f"LO {lo.id} depends on {dep}, which was merged into {merged_into[dep]}; "
                    f"depend on {merged_into[dep]} instead.",
                    [lo.id, dep],
                )
            else:
                log.error("unknown_dependency", f"LO {lo.id} depends on unknown LO {dep}.", [lo.id])

    # Ancestor cycles: walk up parent links.
    reported: set[frozenset[str]] = set()
    for lo in out.los:
        path, cur = [lo.id], lo.parent_id
        while cur is not None and cur in by_id:
            if cur in path:
                loop = path[path.index(cur) :]
                if frozenset(loop) not in reported:
                    reported.add(frozenset(loop))
                    log.error(
                        "own_ancestor",
                        "Parent links form a loop: " + " -> ".join(loop + [cur]) + ".",
                        loop,
                    )
                break
            path.append(cur)
            cur = by_id[cur].parent_id

    # Module membership
    placed: dict[str, list[str]] = {}
    for m in out.modules:
        members = list(m.lo_ids)
        if m.aggregate_lo_id is not None:
            head = by_id.get(m.aggregate_lo_id)
            if head is None:
                log.error(
                    "unknown_module_head",
                    f"Module {m.id} is headed by unknown LO {m.aggregate_lo_id}.",
                    [m.id],
                )
            elif head.scope != "aggregate":
                log.error(
                    "module_head_not_aggregate",
                    f"Module {m.id} is headed by {head.id}, which is atomic.",
                    [m.id, head.id],
                )
            if m.aggregate_lo_id not in members:
                members.append(m.aggregate_lo_id)
        for lid in members:
            if lid not in by_id:
                if lid != m.aggregate_lo_id:
                    log.error(
                        "unknown_module_member", f"Module {m.id} lists unknown LO {lid}.", [m.id]
                    )
                continue
            placed.setdefault(lid, []).append(m.id)
        for dep in m.depends_on:
            if dep not in mod_set:
                log.error(
                    "unknown_module_dependency",
                    f"Module {m.id} depends on unknown module {dep}.",
                    [m.id],
                )
        if m.aggregate_lo_id in by_id:
            stray = [c for c in children.get(m.aggregate_lo_id, []) if c not in m.lo_ids]
            if stray:
                log.warn(
                    "head_children_elsewhere",
                    f"Module {m.id} is headed by {m.aggregate_lo_id}, but its children "
                    f"{stray} are not in the module.",
                    [m.id, *stray],
                )
    for lid, mods in placed.items():
        if len(mods) > 1:
            log.error("lo_in_many_modules", f"LO {lid} is in several modules: {mods}.", [lid])
    for lo in out.los:
        if lo.scope == "atomic" and lo.id not in placed:
            log.error("atomic_lo_unplaced", f"Atomic LO {lo.id} is not in any module.", [lo.id])

    # Module order field
    orders = [m.order for m in out.modules]
    if len(set(orders)) != len(orders):
        log.error("duplicate_module_order", f"Module order values repeat: {orders}.")
    elif orders != sorted(orders):
        log.error(
            "modules_not_sorted",
            f"Modules must be listed in ascending `order`; got {orders}.",
            mod_ids,
        )

    return log.finish(ValidationResult())
