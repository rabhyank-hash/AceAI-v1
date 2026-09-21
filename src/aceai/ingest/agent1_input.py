"""Build Agent 1's input for one course: LO texts only, structure removed, seeded shuffle.

Raw ids encode the ground truth (`ppp-u03-concept-data-structures-lo02` names the unit, module
type, module, and position), so the agent sees opaque input ids instead. The id map back to raw ids
and the log of text edits are returned alongside the payload and must not be sent to the agent.
"""

from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from aceai.config import DATA_RAW
from aceai.ingest.loader import IngestError, load_all
from aceai.schemas import RawLO

# Trailing "(LO3)" markers in some PPP texts equal the CSV `LO no`, i.e. they leak module order.
_LO_MARKER = re.compile(r"\s*\(LO ?\d+\)\s*$")
_ID_HEX = 6


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class InputLO(_Model):
    id: str
    text: str


class Agent1Input(_Model):
    """The only object that may be shown to Agent 1."""

    course: str
    los: list[InputLO]


@dataclass(frozen=True)
class TextEdit:
    raw_id: str
    before: str
    after: str
    reason: str


@dataclass(frozen=True)
class PreparedInput:
    payload: Agent1Input  # send this
    seed: int
    id_map: dict[str, str]  # input id -> raw_id; keep out of the prompt
    edits: list[TextEdit]  # every change made to an LO text, for the run log


def input_id(raw_id: str) -> str:
    """Opaque, stable id: independent of seed and of the LO's position in the course."""
    return "LO-" + hashlib.sha256(raw_id.encode()).hexdigest()[:_ID_HEX]


def strip_lo_marker(text: str) -> str:
    return _LO_MARKER.sub("", text)


def make_agent1_input(
    course: str, seed: int, los: list[RawLO] | None = None, raw_dir: Path = DATA_RAW
) -> PreparedInput:
    """Agent 1 input for `course`: its detailed LOs plus its syllabus broad LOs (PPP only, for now),
    mixed together and shuffled with `seed`. Pass `los` to skip reading `raw_dir`."""
    if los is None:
        by_course = load_all(raw_dir)
        if course not in by_course:
            raise IngestError(f"unknown course {course!r}; have {sorted(by_course)}")
        los = by_course[course]
    los = [lo for lo in los if lo.course == course]
    if not los:
        raise IngestError(f"no LOs for course {course!r}")

    items: list[InputLO] = []
    id_map: dict[str, str] = {}
    edits: list[TextEdit] = []
    for lo in los:
        iid = input_id(lo.raw_id)
        if iid in id_map:
            raise IngestError(f"input id collision: {lo.raw_id} and {id_map[iid]}")
        id_map[iid] = lo.raw_id
        text = strip_lo_marker(lo.text)
        if text != lo.text:
            edits.append(TextEdit(lo.raw_id, lo.text, text, "removed trailing (LOn) marker"))
        items.append(InputLO(id=iid, text=text))

    # Sort first so the result depends only on the seed, not on file order.
    items.sort(key=lambda x: x.id)
    random.Random(seed).shuffle(items)
    return PreparedInput(
        payload=Agent1Input(course=course, los=items), seed=seed, id_map=id_map, edits=edits
    )
