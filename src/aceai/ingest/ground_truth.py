"""Ground-truth structure of a course: units -> modules -> raw LO ids, as the authors wrote it.

Order is the CSV row order (units, modules within a unit, LOs within a module). Names are kept
exactly as written. Syllabus broad LOs are not part of the CSV structure; they are listed
separately by level.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from aceai.ingest.loader import IngestError
from aceai.schemas import CsvSource, ModuleType, RawLO, SyllabusLevel, SyllabusSource


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class GTModule(_Model):
    module_type: ModuleType
    module_name: str
    lo_ids: list[str]


class GTUnit(_Model):
    unit_no: int
    unit_name: str
    modules: list[GTModule]


class GroundTruth(_Model):
    course: str
    units: list[GTUnit]
    broad_lo_ids: dict[SyllabusLevel, list[str]]

    def all_lo_ids(self) -> list[str]:
        detailed = [i for u in self.units for m in u.modules for i in m.lo_ids]
        broad = [i for ids in self.broad_lo_ids.values() for i in ids]
        return detailed + broad


def extract_ground_truth(course: str, los: list[RawLO]) -> GroundTruth:
    """Build the course's structure from its RawLOs, in file order.

    Raises if a module's rows are not contiguous in the file, since the order would be ambiguous.
    """
    units: list[tuple[int, str, list[tuple[ModuleType, str, list[str]]]]] = []
    seen_modules: set[tuple[int, ModuleType, str]] = set()
    broad: dict[SyllabusLevel, list[str]] = {}
    for lo in los:
        if lo.course != course:
            raise IngestError(f"{lo.raw_id} belongs to {lo.course}, not {course}")
        s = lo.source
        if isinstance(s, SyllabusSource):
            broad.setdefault(s.level, []).append(lo.raw_id)
            continue
        assert isinstance(s, CsvSource)
        if not units or units[-1][0] != s.unit_no:
            if any(u[0] == s.unit_no for u in units):
                raise IngestError(f"{course}: unit {s.unit_no} rows are not contiguous")
            units.append((s.unit_no, s.unit_name, []))
        modules = units[-1][2]
        key = (s.unit_no, s.module_type, s.module_name)
        if not modules or (s.unit_no, modules[-1][0], modules[-1][1]) != key:
            if key in seen_modules:
                raise IngestError(f"{course}: module {key} rows are not contiguous")
            seen_modules.add(key)
            modules.append((s.module_type, s.module_name, []))
        modules[-1][2].append(lo.raw_id)
    return GroundTruth(
        course=course,
        units=[
            GTUnit(
                unit_no=no,
                unit_name=name,
                modules=[GTModule(module_type=t, module_name=n, lo_ids=ids) for t, n, ids in mods],
            )
            for no, name, mods in units
        ],
        broad_lo_ids={lvl: broad[lvl] for lvl in SyllabusLevel if lvl in broad},
    )
