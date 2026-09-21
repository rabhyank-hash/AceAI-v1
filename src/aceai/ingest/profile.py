"""Data profile of the raw LO files: counts, naming quirks, duplicates, logistics candidates.

Everything here is a report. Nothing is removed or rewritten.
"""

from __future__ import annotations

import difflib
import itertools
import re
from collections import Counter, defaultdict
from dataclasses import dataclass

from aceai.schemas import CsvSource, ModuleType, RawLO, SyllabusSource

NEAR_DUP_THRESHOLD = 0.8

# Pattern -> reason. Heuristic: flags LOs that look like course logistics rather than learning.
LOGISTICS_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"sail\(\)", re.I), "mentions the Sail() platform"),
    (re.compile(r"\binline activit", re.I), "inline activities"),
    (re.compile(r"\b(grading|grader|auto-grader|submission|submit)\b", re.I), "grading/submission"),
    (re.compile(r"\bcreate (an?|a new) \w+ account\b", re.I), "account creation"),
    (
        re.compile(r"\b(account|subscription) setup\b|\bset up an? \w+ account\b", re.I),
        "account setup",
    ),
    (re.compile(r"\bpersonal access token\b", re.I), "access-token setup"),
    (re.compile(r"\bstarter code\b", re.I), "starter code"),
    (re.compile(r"\bpurchase\b", re.I), "purchase"),
    (re.compile(r"\binstall\b", re.I), "tool installation"),
]

EMBEDDED_LO_MARKER = re.compile(r"\(LO ?(\d+)\)")
_MIXED_CASE_WORD = re.compile(r"\b[A-Za-z]*[a-z][A-Z][A-Za-z]*\b|\b[A-Z]{2,}[a-z]+\w*\b")
_SHOUTED_WORD = re.compile(r"\b[A-Z]{5,}\b")


def dedup_key(text: str) -> str:
    """Casefold, drop punctuation, collapse whitespace. Equal keys = exact duplicates."""
    return " ".join(re.sub(r"[^\w\s]", " ", text.casefold()).split())


def similarity(a: str, b: str) -> float:
    """min(token Jaccard, character ratio) on dedup keys. Both must be high to count as near."""
    ka, kb = dedup_key(a), dedup_key(b)
    ta, tb = set(ka.split()), set(kb.split())
    jaccard = len(ta & tb) / len(ta | tb) if ta | tb else 1.0
    if jaccard < NEAR_DUP_THRESHOLD:
        return jaccard
    return min(jaccard, difflib.SequenceMatcher(None, ka, kb).ratio())


def word_diff(a: str, b: str) -> tuple[list[str], list[str]]:
    ta, tb = set(dedup_key(a).split()), set(dedup_key(b).split())
    return sorted(ta - tb), sorted(tb - ta)


def logistics_reasons(text: str) -> list[str]:
    return [reason for pat, reason in LOGISTICS_PATTERNS if pat.search(text)]


def name_quirks(name: str) -> list[str]:
    quirks = []
    if name != name.strip():
        quirks.append("leading/trailing whitespace")
    if "  " in name:
        quirks.append("repeated spaces")
    if name.strip() and name.strip()[0].islower():
        quirks.append("starts lowercase")
    if _SHOUTED_WORD.search(name):
        quirks.append("ALL-CAPS words")
    mixed = _MIXED_CASE_WORD.findall(name)
    if mixed:
        quirks.append("mixed-case words: " + ", ".join(mixed))
    return quirks


def location(lo: RawLO) -> str:
    s = lo.source
    if isinstance(s, CsvSource):
        return f"u{s.unit_no} {s.module_type.value} “{s.module_name.strip()}” #{s.lo_no}"
    return f"syllabus {s.level.value} #{s.lo_no}"


@dataclass(frozen=True)
class DupPair:
    a: RawLO
    b: RawLO
    score: float  # 1.0 for exact


def find_duplicates(los: list[RawLO]) -> tuple[list[list[RawLO]], list[DupPair]]:
    """Exact duplicate groups (same dedup key) and near-duplicate pairs (not exact)."""
    groups: dict[str, list[RawLO]] = defaultdict(list)
    for lo in los:
        groups[dedup_key(lo.text)].append(lo)
    exact = [g for g in groups.values() if len(g) > 1]
    near = []
    for a, b in itertools.combinations(los, 2):
        if dedup_key(a.text) == dedup_key(b.text):
            continue
        s = similarity(a.text, b.text)
        if s >= NEAR_DUP_THRESHOLD:
            near.append(DupPair(a, b, s))
    near.sort(key=lambda p: (-p.score, p.a.raw_id, p.b.raw_id))
    return exact, near


# --- Markdown rendering -------------------------------------------------------------------------


def _md_escape(text: str) -> str:
    return text.replace("|", "\\|")


def _dup_rows(near: list[DupPair], show_course: bool) -> list[str]:
    head = "| score | A | B | only in A | only in B |"
    rows = [head, "|---|---|---|---|---|"]
    for p in near:
        only_a, only_b = word_diff(p.a.text, p.b.text)
        loc_a = (f"{p.a.course} " if show_course else "") + location(p.a)
        loc_b = (f"{p.b.course} " if show_course else "") + location(p.b)
        rows.append(
            f"| {p.score:.2f} | {_md_escape(loc_a)}: {_md_escape(p.a.text)} "
            f"| {_md_escape(loc_b)}: {_md_escape(p.b.text)} "
            f"| {' '.join(only_a) or '—'} | {' '.join(only_b) or '—'} |"
        )
    return rows


def _course_section(course: str, los: list[RawLO]) -> list[str]:
    detailed = [lo for lo in los if isinstance(lo.source, CsvSource)]
    broad = [lo for lo in los if isinstance(lo.source, SyllabusSource)]
    out = [f"## {course}", ""]
    out.append(f"- Detailed LOs: **{len(detailed)}**; syllabus broad LOs: **{len(broad)}**")

    # Counts by unit x module type
    units: dict[int, str] = {}
    counts: Counter[tuple[int, ModuleType]] = Counter()
    modules: dict[tuple[int, ModuleType, str], int] = {}
    for lo in detailed:
        s = lo.source
        assert isinstance(s, CsvSource)
        units.setdefault(s.unit_no, s.unit_name)
        counts[(s.unit_no, s.module_type)] += 1
        modules[(s.unit_no, s.module_type, s.module_name)] = (
            modules.get((s.unit_no, s.module_type, s.module_name), 0) + 1
        )
    all_units = sorted(units)
    missing = [u for u in range(min(all_units), max(all_units) + 1) if u not in units]
    out.append(
        f"- Units: {len(all_units)} ({', '.join(map(str, all_units))})"
        + (f"; gaps in numbering: {missing}" if missing else "")
    )
    out.append(f"- Modules: {len(modules)}")
    if broad:
        levels = Counter(lo.source.level.value for lo in broad)  # type: ignore[union-attr]
        out.append("- Syllabus levels: " + ", ".join(f"{k} {v}" for k, v in sorted(levels.items())))
    out += ["", "### LO count by unit and module type", ""]
    types = list(ModuleType)
    out.append("| unit | name | " + " | ".join(t.value for t in types) + " | total |")
    out.append("|---|---|" + "---|" * (len(types) + 1))
    for u in all_units:
        row = [counts[(u, t)] for t in types]
        out.append(
            f"| {u} | {_md_escape(repr(units[u]))} | "
            + " | ".join(str(c) for c in row)
            + f" | {sum(row)} |"
        )

    out += ["", "### Modules", ""]
    out.append("| unit | type | module name (as written) | LOs | name quirks |")
    out.append("|---|---|---|---|---|")
    for (u, t, name), n in modules.items():
        q = "; ".join(name_quirks(name)) or ""
        out.append(f"| {u} | {t.value} | {_md_escape(repr(name))} | {n} | {q} |")
    unit_quirks = [(u, units[u], name_quirks(units[u])) for u in all_units]
    unit_quirks = [x for x in unit_quirks if x[2]]
    if unit_quirks:
        out += ["", "Unit-name quirks:", ""]
        out += [f"- unit {u} {name!r}: {'; '.join(q)}" for u, name, q in unit_quirks]
    same_name = defaultdict(list)
    for u, t, name in modules:
        same_name[name.strip().casefold()].append(f"u{u} {t.value}")
    shared = {k: v for k, v in same_name.items() if len(v) > 1}
    if shared:
        out += ["", "Module names used by more than one module in this course:", ""]
        out += [f"- {k!r}: {', '.join(v)}" for k, v in sorted(shared.items())]

    ws = [lo for lo in los if lo.original_text != lo.text]
    markers = [lo for lo in los if EMBEDDED_LO_MARKER.search(lo.text)]
    out += ["", "### Text quirks", ""]
    out.append(f"- LOs whose text changed under whitespace normalization: {len(ws)}")
    for lo in ws:
        out.append(f"  - `{lo.raw_id}`: {_md_escape(repr(lo.original_text))}")
    out.append(f"- LOs containing an embedded `(LOn)` marker: {len(markers)}")
    mismatched = [
        lo
        for lo in markers
        if isinstance(lo.source, CsvSource)
        and int(EMBEDDED_LO_MARKER.search(lo.text).group(1)) != lo.source.lo_no  # type: ignore[union-attr]
    ]
    if markers:
        out.append(f"  - of which the marker differs from the CSV `LO no`: {len(mismatched)}")
        for lo in mismatched:
            out.append(f"    - `{lo.raw_id}` ({location(lo)}): {_md_escape(lo.text)}")

    exact, near = find_duplicates(los)
    out += ["", "### Exact duplicates within the course", ""]
    if not exact:
        out.append("None.")
    for g in exact:
        out.append(f"- “{_md_escape(g[0].text)}”")
        out += [f"  - `{lo.raw_id}` — {location(lo)}" for lo in g]
    out += ["", f"### Near-duplicates within the course (similarity ≥ {NEAR_DUP_THRESHOLD})", ""]
    if near:
        out += _dup_rows(near, show_course=False)
    else:
        out.append("None.")

    logistics = [(lo, logistics_reasons(lo.text)) for lo in los]
    logistics = [(lo, r) for lo, r in logistics if r]
    out += ["", "### Candidate logistics-only LOs (heuristic; not removed)", ""]
    if not logistics:
        out.append("None.")
    for lo, reasons in logistics:
        out.append(f"- `{lo.raw_id}` [{', '.join(reasons)}]: {_md_escape(lo.text)}")
    out.append("")
    return out


def _summary(by_course: dict[str, list[RawLO]]) -> list[str]:
    out = ["## Summary", ""]
    out.append(
        "| course | detailed LOs | broad LOs | units | modules | exact-dup groups "
        "| near-dup pairs | logistics candidates | `(LOn)` markers | ws-normalized |"
    )
    out.append("|---|---|---|---|---|---|---|---|---|---|")
    for course, los in by_course.items():
        detailed = [lo for lo in los if isinstance(lo.source, CsvSource)]
        units = {lo.source.unit_no for lo in detailed}  # type: ignore[union-attr]
        modules = {
            (lo.source.unit_no, lo.source.module_type, lo.source.module_name)  # type: ignore[union-attr]
            for lo in detailed
        }
        exact, near = find_duplicates(los)
        logi = sum(1 for lo in los if logistics_reasons(lo.text))
        markers = sum(1 for lo in los if EMBEDDED_LO_MARKER.search(lo.text))
        ws = sum(1 for lo in los if lo.original_text != lo.text)
        out.append(
            f"| {course} | {len(detailed)} | {len(los) - len(detailed)} | {len(units)} "
            f"| {len(modules)} | {len(exact)} | {len(near)} | {logi} | {markers} | {ws} |"
        )
    out.append("")
    return out


def render_profile(by_course: dict[str, list[RawLO]]) -> str:
    lines = [
        "# Data profile — raw LO files",
        "",
        "Generated by `scripts/profile_data.py`. Report only: nothing has been removed or changed.",
        f"Near-duplicate similarity = min(token Jaccard, character ratio) after casefolding and "
        f"removing punctuation; pairs ≥ {NEAR_DUP_THRESHOLD} are listed with the words that "
        "differ. "
        "High similarity often means *parallel* LOs (e.g. list vs. set), not duplicates.",
        "Logistics candidates come from keyword rules (reasons in brackets) and need human review.",
        "Courses are profiled independently: Agent 1 only ever sees one course's LOs, so overlap "
        "between courses is not reported.",
        "",
    ]
    lines += _summary(by_course)
    for course, los in by_course.items():
        lines += _course_section(course, los)
    return "\n".join(lines).rstrip() + "\n"
