"""Structured results shared by the Agent 1 tools.

Every tool returns a `ToolResult` subclass: `ok` is False iff there are errors; warnings never flip
`ok`. Results serialize to JSON with `model_dump_json()` so an LLM agent can read and act on them.
"""

from __future__ import annotations

from typing import TypeVar

from pydantic import BaseModel, ConfigDict, Field


class Issue(BaseModel):
    model_config = ConfigDict(frozen=True)

    code: str  # stable, machine-readable, e.g. "unknown_parent"
    message: str  # one sentence the agent can act on
    ids: list[str] = Field(default_factory=list)  # LO / module / raw ids involved


class ToolResult(BaseModel):
    ok: bool = True
    errors: list[Issue] = Field(default_factory=list)
    warnings: list[Issue] = Field(default_factory=list)


R = TypeVar("R", bound=ToolResult)


class IssueLog:
    """Collects issues while a tool runs; `finish` stamps them onto the result."""

    def __init__(self) -> None:
        self.errors: list[Issue] = []
        self.warnings: list[Issue] = []

    def error(self, code: str, message: str, ids: list[str] | None = None) -> None:
        self.errors.append(Issue(code=code, message=message, ids=ids or []))

    def warn(self, code: str, message: str, ids: list[str] | None = None) -> None:
        self.warnings.append(Issue(code=code, message=message, ids=ids or []))

    def finish(self, result: R) -> R:
        result.errors = list(result.errors) + self.errors
        result.warnings = list(result.warnings) + self.warnings
        result.ok = not result.errors
        return result
