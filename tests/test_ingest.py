from pathlib import Path

import pytest

from aceai.config import DATA_RAW
from aceai.ingest import IngestError, load_all, load_course_csv, load_csv, load_syllabus_csv
from aceai.ingest.profile import (
    dedup_key,
    find_duplicates,
    logistics_reasons,
    name_quirks,
    render_profile,
)
from aceai.schemas import CsvSource, ModuleType, RawLO, SyllabusLevel, SyllabusSource

HEADER = "course name,Unit no,Unit Name,module type,Module name,LO no,Learning Objective"
SYL_HEADER = "course name,level,LO no,Learning Objective"


def write(tmp_path: Path, name: str, lines: list[str], *, bom=True, crlf=True) -> Path:
    sep = "\r\n" if crlf else "\n"
    data = sep.join(lines) + sep
    path = tmp_path / name
    path.write_bytes((b"\xef\xbb\xbf" if bom else b"") + data.encode("utf-8"))
    return path


# --- CSV quirks ---------------------------------------------------------------------------------


@pytest.mark.parametrize("bom", [True, False])
@pytest.mark.parametrize("crlf", [True, False])
def test_bom_and_line_endings(tmp_path, bom, crlf):
    path = write(
        tmp_path,
        "X_learning_objectives_1.csv",
        [HEADER, "X Course,0,getting ready,CONCEPT,Intro,1,Explain things."],
        bom=bom,
        crlf=crlf,
    )
    [lo] = load_course_csv(path)
    assert lo.course == "X"
    assert lo.text == "Explain things."  # no stray \r
    assert lo.source.unit_name == "getting ready"


def test_quoted_commas_and_embedded_newline(tmp_path):
    path = write(
        tmp_path,
        "X_learning_objectives_1.csv",
        [
            HEADER,
            'X,3,"Extraction, Transformation, and Loading (ETL)",PROJECT,'
            '"Docker, Kubernetes",2,"Inspect, query, and clean data."',
            'X,3,"Extraction, Transformation, and Loading (ETL)",PROJECT,'
            '"Docker, Kubernetes",3,"Line one\r\nline two"',
        ],
    )
    a, b = load_course_csv(path)
    assert a.source.unit_name == "Extraction, Transformation, and Loading (ETL)"
    assert a.source.module_name == "Docker, Kubernetes"
    assert a.text == "Inspect, query, and clean data."
    assert b.text == "Line one line two"
    assert b.original_text == "Line one\r\nline two"


def test_whitespace_normalized_but_original_and_names_kept(tmp_path):
    path = write(
        tmp_path,
        "X_learning_objectives_1.csv",
        [
            HEADER,
            "X,2,Introduction TO DAta Engineering ,PRIMER,Module  A ,1,"
            '"  Discuss   data\u00a0security. "',
        ],
    )
    [lo] = load_course_csv(path)
    assert lo.text == "Discuss data security."
    assert lo.original_text == "  Discuss   data\u00a0security. "
    # Names are kept exactly as written; the profile reports quirks.
    assert lo.source.unit_name == "Introduction TO DAta Engineering "
    assert lo.source.module_name == "Module  A "
    assert lo.raw_id == "x-u02-primer-module-a-lo01"


def test_raw_id_format(tmp_path):
    path = write(
        tmp_path,
        "PPP_learning_objectives_1.csv",
        [HEADER, "P,3,data structures,CONCEPT,Data Structures,2,Describe lists."],
    )
    [lo] = load_course_csv(path)
    assert lo.raw_id == "ppp-u03-concept-data-structures-lo02"
    assert lo.source.module_type is ModuleType.CONCEPT


def test_blank_trailing_lines_ignored(tmp_path):
    path = write(
        tmp_path,
        "X_learning_objectives_1.csv",
        [HEADER, "X,0,u,CONCEPT,m,1,Explain.", "", ",,,,,,"],
    )
    assert len(load_course_csv(path)) == 1


@pytest.mark.parametrize(
    "row, match",
    [
        ("X,0,u,LAB,m,1,Explain.", "unknown module type"),
        ("X,zero,u,CONCEPT,m,1,Explain.", "Unit no is not an integer"),
        ("X,0,u,CONCEPT,m,1,   ", "empty Learning Objective"),
        ("X,0,u,CONCEPT,m,1,Explain.,extra", "expected 7 fields"),
    ],
)
def test_bad_rows_raise_with_location(tmp_path, row, match):
    path = write(tmp_path, "X_learning_objectives_1.csv", [HEADER, row])
    with pytest.raises(IngestError, match=match) as exc:
        load_course_csv(path)
    assert "X_learning_objectives_1.csv:2" in str(exc.value)


def test_unrecognized_header(tmp_path):
    path = write(tmp_path, "X_learning_objectives_1.csv", ["a,b,c", "1,2,3"])
    with pytest.raises(IngestError, match="unrecognized header"):
        load_csv(path)


def test_syllabus_file(tmp_path):
    path = write(
        tmp_path,
        "PPP_syllabus_broad_LOs.csv",
        [SYL_HEADER, 'P,course_goal,2,"Read data in CSV, JSON, and XML."'],
        bom=False,
    )
    [lo] = load_syllabus_csv(path)
    assert lo.raw_id == "ppp-syllabus-course-goal-lo02"
    assert lo.course == "PPP"
    assert isinstance(lo.source, SyllabusSource)
    assert lo.source.level is SyllabusLevel.COURSE_GOAL
    assert lo.text == "Read data in CSV, JSON, and XML."


def test_load_all_dispatches_and_rejects_duplicate_ids(tmp_path):
    write(tmp_path, "P_learning_objectives_1.csv", [HEADER, "P,0,u,CONCEPT,m,1,Explain."])
    write(tmp_path, "P_syllabus_broad_LOs.csv", [SYL_HEADER, "P,course_goal,1,Explain all."])
    out = load_all(tmp_path)
    assert list(out) == ["P"]
    assert [type(lo.source) for lo in out["P"]] == [CsvSource, SyllabusSource]

    write(tmp_path, "P_learning_objectives_2.csv", [HEADER, "P,0,u,CONCEPT,m,1,Again."])
    with pytest.raises(IngestError, match="duplicate raw_id"):
        load_all(tmp_path)


# --- Real data ----------------------------------------------------------------------------------

EXPECTED_DETAILED = {
    "AI_Practitioner": 125,
    "CloudAdmin": 197,
    "CloudDevOps": 171,
    "CloudNative": 166,
    "DataEng": 57,
    "PPP": 202,
}


@pytest.fixture(scope="module")
def real_data() -> dict[str, list[RawLO]]:
    if not all(
        (DATA_RAW / f"{c}_learning_objectives_20260916.csv").exists() for c in EXPECTED_DETAILED
    ):
        pytest.skip("course CSVs not present in data/raw/")
    return load_all(DATA_RAW)


def test_real_data_loads_all_courses(real_data):
    detailed = {
        c: sum(isinstance(lo.source, CsvSource) for lo in los) for c, los in real_data.items()
    }
    assert detailed == EXPECTED_DETAILED
    broad = [lo for lo in real_data["PPP"] if isinstance(lo.source, SyllabusSource)]
    assert len(broad) == 22
    levels = [lo.source.level for lo in broad]
    assert levels.count(SyllabusLevel.COURSE_GOAL) == 5
    assert levels.count(SyllabusLevel.COURSE_CONCEPTUAL) == 9
    assert levels.count(SyllabusLevel.COURSE_PROJECT) == 8


def test_real_data_known_quirks_preserved(real_data):
    units = {lo.source.unit_name for lo in real_data["DataEng"]}
    assert "Introduction TO DAta Engineering" in units
    ppp = {lo.raw_id: lo for lo in real_data["PPP"]}
    assert ppp["ppp-u00-concept-programming-and-python-lo05"].text == (
        "Attempt Sail() inline activities."
    )


# --- Profile helpers ----------------------------------------------------------------------------


def raw(raw_id: str, text: str, course="X") -> RawLO:
    return RawLO(
        raw_id=raw_id,
        course=course,
        text=text,
        original_text=text,
        source=CsvSource(unit_no=0, unit_name="u", module_type="CONCEPT", module_name="m", lo_no=1),
    )


def test_dedup_key_ignores_case_punctuation_space():
    assert dedup_key("Describe the  Features, of X.") == dedup_key("describe the features of x")


def test_find_duplicates_exact_and_near():
    los = [
        raw("a", "Explain the concept of RDDs and their roles in the execution of a Spark job."),
        raw("b", "Explain the concept of RDDs and their roles in the execution of Spark job."),
        raw("c", "Compare NoSQL databases to relational databases."),
        raw("d", "compare nosql databases to relational databases"),
        raw("e", "Write a haiku."),
    ]
    exact, near = find_duplicates(los)
    assert [[lo.raw_id for lo in g] for g in exact] == [["c", "d"]]
    assert [(p.a.raw_id, p.b.raw_id) for p in near] == [("a", "b")]


def test_logistics_reasons():
    assert logistics_reasons("Attempt Sail() inline activities.") == [
        "mentions the Sail() platform",
        "inline activities",
    ]
    assert logistics_reasons("Explain what a computer program is.") == []


def test_name_quirks():
    assert name_quirks("Introduction TO DAta Engineering") == ["mixed-case words: DAta"]
    assert "leading/trailing whitespace" in name_quirks("CI/CD Pipelines ")
    assert "starts lowercase" in name_quirks("getting ready")
    assert "ALL-CAPS words" in name_quirks("SERVERLESS & COMMUNICATION PATTERNS")
    assert name_quirks("Data Modeling") == []


def test_render_profile_smoke():
    md = render_profile({"X": [raw("a", "Attempt Sail() inline activities."), raw("b", "x")]})
    assert "## Summary" in md and "## X" in md
    assert "Across courses" not in md
    assert "`a` [mentions the Sail() platform, inline activities]" in md
