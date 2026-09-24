"""Project paths and LLM provider settings.

Providers are OpenAI-compatible endpoints; adding one is a new `PROVIDERS` entry plus its key in
`.env`. Rate limits are the free-tier values shown in each provider's console. They change, so
re-check them there and edit this file rather than trusting the numbers below.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
RUNS_DIR = PROJECT_ROOT / "runs"
CACHE_DIR = PROJECT_ROOT / ".cache"
LLM_CACHE_DIR = CACHE_DIR / "llm"

DEFAULT_PROVIDER = "groq"


@dataclass(frozen=True)
class RateLimits:
    """Per-model limits. None = unknown or not enforced; checks skip it."""

    requests_per_minute: int | None = None
    requests_per_day: int | None = None
    tokens_per_minute: int | None = None
    tokens_per_day: int | None = None


@dataclass(frozen=True)
class ProviderConfig:
    name: str
    base_url: str
    api_key_env: str
    default_model: str
    limits: dict[str, RateLimits]  # model -> limits
    # model -> extra request params sent on every call (part of the cache key)
    model_params: dict[str, dict[str, Any]] = field(default_factory=dict)

    def api_key(self) -> str:
        load_dotenv(PROJECT_ROOT / ".env")
        key = os.environ.get(self.api_key_env, "").strip()
        if not key:
            raise RuntimeError(
                f"{self.api_key_env} is not set; add it to {PROJECT_ROOT / '.env'} "
                "(see .env.example)"
            )
        return key

    def limits_for(self, model: str) -> RateLimits:
        return self.limits.get(model, RateLimits())


PROVIDERS: dict[str, ProviderConfig] = {
    "groq": ProviderConfig(
        name="groq",
        base_url="https://api.groq.com/openai/v1",
        api_key_env="GROQ_API_KEY",
        default_model="openai/gpt-oss-120b",
        # Read from the x-ratelimit-limit-* response headers on 2026-09-23 (free tier). The headers
        # do not report daily tokens; check https://console.groq.com/settings/limits.
        # llama-3.3-70b-versatile (the model CLAUDE.md names) is no longer offered on this key.
        limits={
            "openai/gpt-oss-120b": RateLimits(requests_per_day=1_000, tokens_per_minute=8_000),
            "qwen/qwen3.8-27b": RateLimits(requests_per_day=1_000, tokens_per_minute=8_000),
        },
        # gpt-oss is a reasoning model; its hidden reasoning tokens count against the per-minute
        # budget, so keep effort low.
        model_params={"openai/gpt-oss-120b": {"reasoning_effort": "low"}},
    ),
    "openrouter": ProviderConfig(
        name="openrouter",
        base_url="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
        default_model="meta-llama/llama-3.3-70b-instruct:free",
        limits={},
    ),
    "mistral": ProviderConfig(
        name="mistral",
        base_url="https://api.mistral.ai/v1",
        api_key_env="MISTRAL_API_KEY",
        default_model="mistral-large-latest",
        limits={},
    ),
    "deepseek": ProviderConfig(
        name="deepseek",
        base_url="https://api.deepseek.com/v1",
        api_key_env="DEEPSEEK_API_KEY",
        default_model="deepseek-chat",
        limits={},
    ),
}


def get_provider(name: str = DEFAULT_PROVIDER) -> ProviderConfig:
    if name not in PROVIDERS:
        raise KeyError(f"unknown provider {name!r}; have {sorted(PROVIDERS)}")
    return PROVIDERS[name]
