"""Course samples for experiments: the first N CSV modules of a course."""

from __future__ import annotations

from aceai.schemas import CsvSource, RawLO


def select_modules(los: list[RawLO], n_modules: int, min_los: int, broad: bool) -> list[RawLO]:
    """The first `n_modules` CSV modules in file order, extended module by module until the
    sample holds at least `min_los` detailed LOs (or the course runs out). `broad` adds the
    course's syllabus LOs."""
    keep: list[RawLO] = []
    seen: list[tuple] = []
    for lo in los:
        if not isinstance(lo.source, CsvSource):
            continue
        key = (lo.source.unit_no, lo.source.module_type, lo.source.module_name)
        if key not in seen:
            if len(seen) >= n_modules and len(keep) >= min_los:
                break
            seen.append(key)
        keep.append(lo)
    if broad:
        keep += [lo for lo in los if not isinstance(lo.source, CsvSource)]
    return keep
