"""Phase 0 schemas (from "ACE-AI Schema Concepts v1").

Models validate only what they can check on their own (field types, enums, ranges, self-references).
Cross-object checks (ids resolve, provenance is complete, graphs are acyclic) belong to the Agent 1
tools in `aceai.tools`, which report problems as data instead of raising.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

NonEmptyStr = Annotated[str, Field(min_length=1)]


class _Model(BaseModel):
    # Reject unknown fields so a malformed LLM response fails validation instead of being dropped.
    model_config = ConfigDict(extra="forbid", frozen=True)


# --- Bloom levels -------------------------------------------------------------------------------


class BloomLevel(StrEnum):
    """Bloom's cognitive levels. Serialized as "C1".."C6"; ordered by number, not by string."""

    C1 = "C1"
    C2 = "C2"
    C3 = "C3"
    C4 = "C4"
    C5 = "C5"
    C6 = "C6"

    @property
    def number(self) -> int:
        return int(self.value[1:])

    @property
    def label(self) -> str:
        return _BLOOM_LABELS[self]

    @classmethod
    def from_number(cls, n: int) -> BloomLevel:
        if not 1 <= n <= 6:
            raise ValueError(f"Bloom level number must be 1-6, got {n}")
        return cls(f"C{n}")

    # StrEnum inherits str comparisons, which would also accept plain strings. Restrict ordering to
    # BloomLevel and compare by number. (Returning NotImplemented is not enough: Python would fall
    # back to the reflected str comparison.)
    def _other_number(self, other: object) -> int:
        if not isinstance(other, BloomLevel):
            raise TypeError(f"cannot order BloomLevel against {type(other).__name__}")
        return other.number

    def __lt__(self, other: object) -> bool:
        return self.number < self._other_number(other)

    def __le__(self, other: object) -> bool:
        return self.number <= self._other_number(other)

    def __gt__(self, other: object) -> bool:
        return self.number > self._other_number(other)

    def __ge__(self, other: object) -> bool:
        return self.number >= self._other_number(other)


_BLOOM_LABELS = {
    BloomLevel.C1: "Remember",
    BloomLevel.C2: "Understand",
    BloomLevel.C3: "Apply",
    BloomLevel.C4: "Analyze",
    BloomLevel.C5: "Evaluate",
    BloomLevel.C6: "Create",
}

Track = Literal["conceptual", "applied"]
Scope = Literal["atomic", "aggregate"]


def _check_no_dupes(name: str, values: list[str]) -> None:
    dupes = sorted({v for v in values if values.count(v) > 1})
    if dupes:
        raise ValueError(f"{name} contains duplicate ids: {dupes}")


# --- Learning objectives ------------------------------------------------------------------------


class LearningObjective(_Model):
    id: NonEmptyStr
    raw_text: NonEmptyStr
    canonical_text: str | None = None
    source_ids: list[NonEmptyStr]
    verb: NonEmptyStr
    bloom_level: BloomLevel
    track: Track
    target_concept: NonEmptyStr
    scope: Scope
    parent_id: str | None = None
    depends_on: list[NonEmptyStr] = Field(default_factory=list)
    priority: Annotated[int, Field(ge=1, le=5)] | None = None  # 1 = highest
    depth_floor: BloomLevel | None = None

    @model_validator(mode="after")
    def _self_consistent(self) -> LearningObjective:
        if self.parent_id == self.id:
            raise ValueError(f"LO {self.id} is its own parent")
        if self.id in self.depends_on:
            raise ValueError(f"LO {self.id} depends on itself")
        _check_no_dupes("source_ids", self.source_ids)
        _check_no_dupes("depends_on", self.depends_on)
        return self


# --- Raw ingestion records ----------------------------------------------------------------------


class ModuleType(StrEnum):
    CONCEPT = "CONCEPT"
    PRIMER = "PRIMER"
    PROJECT = "PROJECT"


class SyllabusLevel(StrEnum):
    COURSE_GOAL = "course_goal"
    COURSE_CONCEPTUAL = "course_conceptual"
    COURSE_PROJECT = "course_project"


class CsvSource(_Model):
    """Ground-truth position of a detailed LO in a course CSV. Never shown to Agent 1."""

    kind: Literal["csv"] = "csv"
    unit_no: Annotated[int, Field(ge=0)]
    unit_name: str  # as written in the CSV, including casing/whitespace quirks
    module_type: ModuleType
    module_name: str  # as written in the CSV
    lo_no: Annotated[int, Field(ge=1)]


class SyllabusSource(_Model):
    """Position of a course-level broad LO in a syllabus file. Never shown to Agent 1."""

    kind: Literal["syllabus"] = "syllabus"
    level: SyllabusLevel
    lo_no: Annotated[int, Field(ge=1)]


class RawLO(_Model):
    """One ingested LO. All ground-truth structure lives in `source` so it can be stripped whole."""

    raw_id: NonEmptyStr
    course: NonEmptyStr  # short course key, e.g. "PPP"
    text: NonEmptyStr  # whitespace-normalized
    original_text: str  # exactly as in the file
    source: Annotated[CsvSource | SyllabusSource, Field(discriminator="kind")]


# --- Agent 1 output -----------------------------------------------------------------------------


class Module(_Model):
    id: NonEmptyStr
    title: NonEmptyStr
    order: Annotated[int, Field(ge=0)]
    aggregate_lo_id: str | None = None
    lo_ids: list[NonEmptyStr]
    depends_on: list[NonEmptyStr] = Field(default_factory=list)

    @model_validator(mode="after")
    def _self_consistent(self) -> Module:
        if self.id in self.depends_on:
            raise ValueError(f"Module {self.id} depends on itself")
        _check_no_dupes("lo_ids", self.lo_ids)
        _check_no_dupes("depends_on", self.depends_on)
        return self


class ProvenanceEntry(_Model):
    raw_id: NonEmptyStr
    lo_id: NonEmptyStr


class SequencerOutput(_Model):
    """Agent 1 output: ordered modules, all surviving LOs, and raw_id -> surviving LO provenance.

    Provenance is a list of entries rather than a JSON object: an LLM that maps one raw id twice
    would otherwise have the duplicate silently collapsed by the JSON parser, and `check_provenance`
    must be able to see and report it.
    """

    modules: list[Module]
    los: list[LearningObjective]
    provenance: list[ProvenanceEntry]

    def provenance_map(self) -> dict[str, list[str]]:
        """raw_id -> every LO id it was mapped to (more than one means a duplicated mapping)."""
        out: dict[str, list[str]] = {}
        for entry in self.provenance:
            out.setdefault(entry.raw_id, []).append(entry.lo_id)
        return out


# --- Agent 2 (stubs; not in scope yet) ----------------------------------------------------------


class Constraints(BaseModel):
    """Stub. Time, cost, and learner-level constraints for Agent 2."""

    model_config = ConfigDict(extra="allow")


class TrainingPlan(BaseModel):
    """Stub. Agent 2 output: time allocation, schedule, delivery/grouping, compression log."""

    model_config = ConfigDict(extra="allow")
