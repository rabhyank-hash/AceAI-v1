"""Thin, provider-agnostic chat client over the `openai` SDK.

- Any OpenAI-compatible provider from `aceai.config.PROVIDERS` (base URL + key from `.env`).
- JSON mode (`json_mode=True`) and tool calling (`tools=[...]`).
- Retries HTTP 429 honoring `retry-after` (or Groq's `x-ratelimit-reset-*`), with exponential
  backoff otherwise; also retries 5xx and connection errors.
- On-disk cache keyed by (provider, model, messages, params): a rerun with identical inputs costs
  nothing. Pass `use_cache=False` to force a fresh call.
- Every call (cached or not) is appended to `call_log` so runs can save it.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import openai
from pydantic import BaseModel, Field

from aceai.config import DEFAULT_PROVIDER, LLM_CACHE_DIR, ProviderConfig, get_provider

_RATE_HEADER_PREFIXES = ("x-ratelimit-", "retry-after")
_DURATION_PART = re.compile(r"(\d+(?:\.\d+)?)(ms|h|m|s)")
_UNIT_SECONDS = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}


class LLMError(RuntimeError):
    pass


class RequestTooLarge(LLMError):
    """The request cannot fit the model's per-minute token budget; batch the input."""


class InvalidJSONReply(LLMError):
    """The provider rejected the model's reply in JSON mode (Groq: `json_validate_failed`).

    `failed_generation` is the rejected text (may be empty, e.g. if max_tokens ran out first).
    """

    def __init__(self, message: str, failed_generation: str) -> None:
        super().__init__(message)
        self.failed_generation = failed_generation


class LLMResponse(BaseModel):
    provider: str
    model: str
    content: str | None
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    finish_reason: str | None = None
    usage: dict[str, Any] = Field(default_factory=dict)
    rate_limit_headers: dict[str, str] = Field(default_factory=dict)
    cached: bool = False
    attempts: int = 1
    cache_key: str = ""

    def parse_json(self) -> Any:
        """Parse `content` as JSON (raises LLMError with the text if it is not)."""
        try:
            return json.loads(self.content or "")
        except json.JSONDecodeError as e:
            raise LLMError(f"model reply is not valid JSON ({e}): {self.content!r:.500}") from e


def parse_duration(value: str | None) -> float | None:
    """Seconds from `retry-after` ("7", "7.5") or Groq reset headers ("2m59.56s", "250ms")."""
    if not value:
        return None
    value = value.strip()
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    parts = _DURATION_PART.findall(value)
    if not parts or "".join(n + u for n, u in parts) != value:
        return None
    return sum(float(n) * _UNIT_SECONDS[u] for n, u in parts)


def estimate_tokens(messages: list[dict[str, Any]]) -> int:
    """Rough token count (~4 chars/token), for budget checks only."""
    return sum(len(json.dumps(m, ensure_ascii=False)) for m in messages) // 4 + 4 * len(messages)


def cache_key(provider: str, model: str, messages: list[dict[str, Any]], params: dict) -> str:
    blob = json.dumps(
        {"provider": provider, "model": model, "messages": messages, "params": params},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(blob.encode()).hexdigest()


class LLMClient:
    def __init__(
        self,
        provider: str | ProviderConfig = DEFAULT_PROVIDER,
        model: str | None = None,
        *,
        cache_dir: Path | None = LLM_CACHE_DIR,
        max_retries: int = 6,
        base_backoff: float = 2.0,
        max_backoff: float = 90.0,
        sdk: Any = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.provider = provider if isinstance(provider, ProviderConfig) else get_provider(provider)
        self.model = model or self.provider.default_model
        self.limits = self.provider.limits_for(self.model)
        self.cache_dir = cache_dir
        self.max_retries = max_retries
        self.base_backoff = base_backoff
        self.max_backoff = max_backoff
        self._sleep = sleep
        self._sdk = sdk
        self.call_log: list[dict[str, Any]] = []

    @property
    def sdk(self) -> Any:
        if self._sdk is None:
            # SDK retries off: this class owns retry so it can log attempts and honor headers.
            self._sdk = openai.OpenAI(
                api_key=self.provider.api_key(), base_url=self.provider.base_url, max_retries=0
            )
        return self._sdk

    # --- public ----------------------------------------------------------------------------------

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        json_mode: bool = False,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        seed: int | None = None,
        use_cache: bool = True,
        label: str = "",
    ) -> LLMResponse:
        params: dict[str, Any] = {
            **self.provider.model_params.get(self.model, {}),
            "temperature": temperature,
        }
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        if seed is not None:
            params["seed"] = seed
        if json_mode:
            params["response_format"] = {"type": "json_object"}
        if tools:
            params["tools"] = tools
            if tool_choice is not None:
                params["tool_choice"] = tool_choice

        key = cache_key(self.provider.name, self.model, messages, params)
        self._check_budget(messages, max_tokens)

        response = self._read_cache(key) if use_cache else None
        if response is None:
            response = self._call(messages, params)
            response.cache_key = key
            self._write_cache(key, response)
        self.call_log.append(
            {
                "label": label,
                "provider": self.provider.name,
                "model": self.model,
                "params": params,
                "messages": messages,
                "response": response.model_dump(),
            }
        )
        return response

    # --- internals -------------------------------------------------------------------------------

    def _check_budget(self, messages: list[dict[str, Any]], max_tokens: int | None) -> None:
        tpm = self.limits.tokens_per_minute
        if tpm is None:
            return
        need = estimate_tokens(messages) + (max_tokens or 0)
        if need > tpm:
            raise RequestTooLarge(
                f"~{need} tokens (prompt estimate + max_tokens) exceeds {self.model}'s "
                f"{tpm} tokens/minute on {self.provider.name}; batch the input or lower max_tokens"
            )

    def _call(self, messages: list[dict[str, Any]], params: dict[str, Any]) -> LLMResponse:
        attempt = 0
        while True:
            attempt += 1
            try:
                raw = self.sdk.chat.completions.with_raw_response.create(
                    model=self.model, messages=messages, **params
                )
            except openai.RateLimitError as e:
                self._retry_or_raise(attempt, e, self._rate_limit_wait(e, attempt))
                continue
            except (openai.InternalServerError, openai.APIConnectionError) as e:
                self._retry_or_raise(attempt, e, self._backoff(attempt))
                continue
            except openai.BadRequestError as e:
                err = e.body.get("error", e.body) if isinstance(e.body, dict) else {}
                if isinstance(err, dict) and err.get("code") == "json_validate_failed":
                    raise InvalidJSONReply(
                        f"provider rejected the reply as invalid JSON: {err.get('message')}",
                        err.get("failed_generation") or "",
                    ) from e
                raise
            completion = raw.parse()
            choice = completion.choices[0]
            msg = choice.message
            return LLMResponse(
                provider=self.provider.name,
                model=self.model,
                content=msg.content,
                tool_calls=[tc.model_dump() for tc in (msg.tool_calls or [])],
                finish_reason=choice.finish_reason,
                usage=completion.usage.model_dump() if completion.usage else {},
                rate_limit_headers=_rate_headers(raw.headers),
                attempts=attempt,
            )

    def _retry_or_raise(self, attempt: int, exc: Exception, wait: float) -> None:
        if attempt > self.max_retries:
            raise LLMError(f"giving up after {attempt} attempts: {exc}") from exc
        self._sleep(wait)

    def _backoff(self, attempt: int) -> float:
        return min(self.max_backoff, self.base_backoff * 2 ** (attempt - 1))

    def _rate_limit_wait(self, exc: openai.RateLimitError, attempt: int) -> float:
        headers = exc.response.headers if exc.response is not None else {}
        for h in ("retry-after", "x-ratelimit-reset-tokens", "x-ratelimit-reset-requests"):
            wait = parse_duration(headers.get(h))
            if wait is not None:
                return min(self.max_backoff, wait + 0.5)
        return self._backoff(attempt)

    def _cache_path(self, key: str) -> Path | None:
        return None if self.cache_dir is None else self.cache_dir / f"{key}.json"

    def _read_cache(self, key: str) -> LLMResponse | None:
        path = self._cache_path(key)
        if path is None or not path.exists():
            return None
        response = LLMResponse.model_validate_json(path.read_text())
        return response.model_copy(update={"cached": True})

    def _write_cache(self, key: str, response: LLMResponse) -> None:
        path = self._cache_path(key)
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(response.model_dump_json(indent=2))
        tmp.replace(path)


def _rate_headers(headers: Any) -> dict[str, str]:
    return {
        k.lower(): v
        for k, v in dict(headers).items()
        if k.lower().startswith(_RATE_HEADER_PREFIXES)
    }
