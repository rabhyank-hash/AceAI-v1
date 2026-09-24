"""Ground-truth structure of a course: units -> modules -> LOs, as the authors wrote it.

Each LO is stored as its raw id (the key everything joins on) plus its text, for readability.

Order is the CSV row order (units, modules within a unit, LOs within a module). Names are kept
exactly as written. Syllabus broad LOs are not part of the CSV structure; they are listed
separately by level.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from aceai.ingest.loader import IngestError
from aceai.schemas import (
    BloomLevel,
    CsvSource,
    LearningObjective,
    Module,
    ModuleType,
    ProvenanceEntry,
    RawLO,
    SequencerOutput,
    SyllabusLevel,
    SyllabusSource,
)


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class GTLO(_Model):
    """One LO in the ground truth. `id` is the raw id and the key everything joins on; `text` is
    the whitespace-normalized CSV text, stored only so the file can be read by eye."""

    id: str
    text: str


class GTModule(_Model):
    module_type: ModuleType
    module_name: str
    los: list[GTLO]

    @property
    def lo_ids(self) -> list[str]:
        return [lo.id for lo in self.los]


class GTUnit(_Model):
    unit_no: int
    unit_name: str
    modules: list[GTModule]


class GroundTruth(_Model):
    course: str
    units: list[GTUnit]
    broad_los: dict[SyllabusLevel, list[GTLO]]

    @property
    def broad_lo_ids(self) -> dict[SyllabusLevel, list[str]]:
        return {lvl: [lo.id for lo in los] for lvl, los in self.broad_los.items()}

    def all_lo_ids(self) -> list[str]:
        detailed = [i for u in self.units for m in u.modules for i in m.lo_ids]
        broad = [i for ids in self.broad_lo_ids.values() for i in ids]
        return detailed + broad


def extract_ground_truth(course: str, los: list[RawLO]) -> GroundTruth:
    """Build the course's structure from its RawLOs, in file order.

    Raises if a module's rows are not contiguous in the file, since the order would be ambiguous.
    """
    units: list[tuple[int, str, list[tuple[ModuleType, str, list[GTLO]]]]] = []
    seen_modules: set[tuple[int, ModuleType, str]] = set()
    broad: dict[SyllabusLevel, list[GTLO]] = {}
    for lo in los:
        if lo.course != course:
            raise IngestError(f"{lo.raw_id} belongs to {lo.course}, not {course}")
        s = lo.source
        if isinstance(s, SyllabusSource):
            broad.setdefault(s.level, []).append(GTLO(id=lo.raw_id, text=lo.text))
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
        modules[-1][2].append(GTLO(id=lo.raw_id, text=lo.text))
    return GroundTruth(
        course=course,
        units=[
            GTUnit(
                unit_no=no,
                unit_name=name,
                modules=[GTModule(module_type=t, module_name=n, los=gl) for t, n, gl in mods],
            )
            for no, name, mods in units
        ],
        broad_los={lvl: broad[lvl] for lvl in SyllabusLevel if lvl in broad},
    )


def ground_truth_to_output(
    gt: GroundTruth, los: list[RawLO], placeholder_bloom: BloomLevel = BloomLevel.C1
) -> SequencerOutput:
    """The human structure as a `SequencerOutput`, e.g. for smoke-testing tools or as Agent 2 input.

    Only the structure is real: modules in file order (ids `m01`, `m02`, ...), each LO kept under
    its raw id, provenance raw id -> itself. The normalized fields (verb, Bloom level, track,
    target concept) are placeholders, since no normalization has run. Syllabus broad LOs become
    aggregate LOs with no children and no module (the CSV does not say what they contain), and there
    are no prerequisite edges.
    """
    by_id = {lo.raw_id: lo for lo in los}
    modules = []
    for unit in gt.units:
        for m in unit.modules:
            idx = len(modules) + 1
            modules.append(
                Module(id=f"m{idx:02d}", title=m.module_name.strip(), order=idx, lo_ids=m.lo_ids)
            )
    broad = {i for ids in gt.broad_lo_ids.values() for i in ids}
    out_los = []
    for rid in gt.all_lo_ids():
        text = by_id[rid].text
        out_los.append(
            LearningObjective(
                id=rid,
                raw_text=text,
                source_ids=[rid],
                verb=text.split()[0].lower(),
                bloom_level=placeholder_bloom,
                track="conceptual",
                target_concept="(placeholder)",
                scope="aggregate" if rid in broad else "atomic",
            )
        )
    return SequencerOutput(
        modules=modules,
        los=out_los,
        provenance=[ProvenanceEntry(raw_id=r, lo_id=r) for r in gt.all_lo_ids()],
    )
