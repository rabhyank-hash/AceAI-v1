"""Load LO CSVs from `data/raw/` into `RawLO` records.

Two file shapes are recognized by header:
- course LO files: see `COURSE_HEADER`
- syllabus broad-LO files: see `SYLLABUS_HEADER`

Files may or may not start with a UTF-8 BOM, may use CRLF line endings, and may quote fields
containing commas.
Text is whitespace-normalized into `RawLO.text`; the exact cell is kept in `RawLO.original_text`.
Unit and module names are kept exactly as written (quirks are reported by the profile, not fixed).
Nothing is dropped: a row that cannot be parsed raises `IngestError` naming the file and line.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

from aceai.schemas import CsvSource, ModuleType, RawLO, SyllabusLevel, SyllabusSource

COURSE_HEADER = [
    "course name",
    "Unit no",
    "Unit Name",
    "module type",
    "Module name",
    "LO no",
    "Learning Objective",
]
SYLLABUS_HEADER = ["course name", "level", "LO no", "Learning Objective"]

_COURSE_FILE_SUFFIX = "_learning_objectives_"
_SYLLABUS_FILE_SUFFIX = "_syllabus_"


class IngestError(ValueError):
    pass


def normalize_ws(text: str) -> str:
    """Collapse all whitespace runs (including NBSP and newlines) to one space and strip."""
    return " ".join(text.split())


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def course_key(path: Path) -> str:
    """Course key from the file name: `PPP_learning_objectives_20260916.csv` -> `PPP`."""
    name = path.name
    for marker in (_COURSE_FILE_SUFFIX, _SYLLABUS_FILE_SUFFIX):
        if marker in name:
            return name.split(marker)[0]
    raise IngestError(f"{path.name}: cannot derive course key from file name")


def _read_rows(path: Path) -> tuple[list[str], list[tuple[int, list[str]]]]:
    """Return (header, [(line_no, row), ...]). utf-8-sig strips a BOM if present; newline=''
    lets the csv module handle CRLF and quoted fields."""
    with path.open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.reader(fh)
        try:
            header = next(reader)
        except StopIteration:
            raise IngestError(f"{path.name}: empty file") from None
        rows = []
        for row in reader:
            if not any(cell.strip() for cell in row):
                continue  # blank line (e.g. trailing newline); carries no data
            rows.append((reader.line_num, row))
    return [h.strip() for h in header], rows


def _int(path: Path, line: int, field: str, value: str) -> int:
    try:
        return int(value.strip())
    except ValueError:
        raise IngestError(f"{path.name}:{line}: {field} is not an integer: {value!r}") from None


def _text(path: Path, line: int, value: str) -> str:
    text = normalize_ws(value)
    if not text:
        raise IngestError(f"{path.name}:{line}: empty Learning Objective")
    return text


def load_course_csv(path: Path) -> list[RawLO]:
    header, rows = _read_rows(path)
    if header != COURSE_HEADER:
        raise IngestError(f"{path.name}: unexpected header {header}")
    course = course_key(path)
    prefix = slugify(course)
    out = []
    for line, row in rows:
        if len(row) != len(COURSE_HEADER):
            raise IngestError(f"{path.name}:{line}: expected 7 fields, got {len(row)}")
        _, unit_no, unit_name, module_type, module_name, lo_no, text = row
        try:
            mtype = ModuleType(module_type.strip())
        except ValueError:
            raise IngestError(f"{path.name}:{line}: unknown module type {module_type!r}") from None
        source = CsvSource(
            unit_no=_int(path, line, "Unit no", unit_no),
            unit_name=unit_name,
            module_type=mtype,
            module_name=module_name,
            lo_no=_int(path, line, "LO no", lo_no),
        )
        raw_id = (
            f"{prefix}-u{source.unit_no:02d}-{mtype.value.lower()}"
            f"-{slugify(module_name)}-lo{source.lo_no:02d}"
        )
        out.append(
            RawLO(
                raw_id=raw_id,
                course=course,
                text=_text(path, line, text),
                original_text=text,
                source=source,
            )
        )
    return out


def load_syllabus_csv(path: Path) -> list[RawLO]:
    header, rows = _read_rows(path)
    if header != SYLLABUS_HEADER:
        raise IngestError(f"{path.name}: unexpected header {header}")
    course = course_key(path)
    prefix = slugify(course)
    out = []
    for line, row in rows:
        if len(row) != len(SYLLABUS_HEADER):
            raise IngestError(f"{path.name}:{line}: expected 4 fields, got {len(row)}")
        _, level, lo_no, text = row
        try:
            lvl = SyllabusLevel(level.strip())
        except ValueError:
            raise IngestError(f"{path.name}:{line}: unknown syllabus level {level!r}") from None
        source = SyllabusSource(level=lvl, lo_no=_int(path, line, "LO no", lo_no))
        out.append(
            RawLO(
                raw_id=f"{prefix}-syllabus-{slugify(lvl.value)}-lo{source.lo_no:02d}",
                course=course,
                text=_text(path, line, text),
                original_text=text,
                source=source,
            )
        )
    return out


def load_csv(path: Path) -> list[RawLO]:
    """Load one file, choosing the parser from its header."""
    header, _ = _read_rows(path)
    if header == COURSE_HEADER:
        return load_course_csv(path)
    if header == SYLLABUS_HEADER:
        return load_syllabus_csv(path)
    raise IngestError(f"{path.name}: unrecognized header {header}")


def load_all(raw_dir: Path) -> dict[str, list[RawLO]]:
    """Load every CSV in `raw_dir`, grouped by course key (sorted), in file order within a course.

    Raises on duplicate raw_ids, which would otherwise make provenance ambiguous.
    """
    by_course: dict[str, list[RawLO]] = {}
    for path in sorted(raw_dir.glob("*.csv")):
        for lo in load_csv(path):
            by_course.setdefault(lo.course, []).append(lo)
    seen: dict[str, str] = {}
    for los in by_course.values():
        for lo in los:
            if lo.raw_id in seen:
                raise IngestError(f"duplicate raw_id {lo.raw_id!r}")
            seen[lo.raw_id] = lo.course
    return dict(sorted(by_course.items()))
