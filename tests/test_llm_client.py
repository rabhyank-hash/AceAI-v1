import pytest

from aceai.config import ProviderConfig, RateLimits
from aceai.llm.client import (
    LLMClient,
    LLMError,
    RequestTooLarge,
    cache_key,
    parse_duration,
)
from fakes import FakeSDK, rate_limit_error

PROVIDER = ProviderConfig(
    name="fake",
    base_url="http://fake/v1",
    api_key_env="FAKE_KEY",
    default_model="m",
    limits={"small": RateLimits(tokens_per_minute=50)},
)
MSGS = [{"role": "user", "content": "hi"}]


def make(script, tmp_path, **kw):
    sleeps = []
    sdk = FakeSDK(script)
    client = LLMClient(PROVIDER, cache_dir=tmp_path, sdk=sdk, sleep=sleeps.append, **kw)
    return client, sdk, sleeps


@pytest.mark.parametrize(
    "value,expected",
    [
        ("7", 7.0),
        ("0.5", 0.5),
        ("2m59.5s", 179.5),
        ("250ms", 0.25),
        ("1h", 3600.0),
        ("", None),
        (None, None),
        ("soon", None),
    ],
)
def test_parse_duration(value, expected):
    assert parse_duration(value) == expected


def test_success_keeps_rate_headers_and_params(tmp_path):
    client, sdk, _ = make(['{"a": 1}'], tmp_path)
    r = client.chat(MSGS, json_mode=True, max_tokens=10, seed=1)
    assert r.parse_json() == {"a": 1}
    assert r.rate_limit_headers == {"x-ratelimit-remaining-tokens": "11000"}
    assert sdk.calls[0]["response_format"] == {"type": "json_object"}
    assert sdk.calls[0]["model"] == "m" and sdk.calls[0]["seed"] == 1
    assert not r.cached and r.attempts == 1


def test_429_honors_retry_after(tmp_path):
    client, sdk, sleeps = make([rate_limit_error({"retry-after": "7"}), "ok"], tmp_path)
    r = client.chat(MSGS)
    assert r.content == "ok" and r.attempts == 2
    assert sleeps == [7.5]


def test_429_falls_back_to_groq_reset_header_then_backoff(tmp_path):
    script = [
        rate_limit_error({"x-ratelimit-reset-tokens": "1m2s"}),
        rate_limit_error({}),
        "ok",
    ]
    client, _, sleeps = make(script, tmp_path, base_backoff=2.0)
    client.chat(MSGS)
    assert sleeps == [62.5, 4.0]  # header wait; then exponential backoff for attempt 2


def test_retry_gives_up(tmp_path):
    client, sdk, sleeps = make([rate_limit_error({})] * 3, tmp_path, max_retries=2)
    with pytest.raises(LLMError, match="giving up after 3"):
        client.chat(MSGS)
    assert len(sdk.calls) == 3 and len(sleeps) == 2


def test_cache_hit_skips_api_and_is_logged(tmp_path):
    client, sdk, _ = make(["first"], tmp_path)
    client.chat(MSGS)
    again = client.chat(MSGS)
    assert again.cached and again.content == "first"
    assert len(sdk.calls) == 1
    assert [c["response"]["cached"] for c in client.call_log] == [False, True]


def test_cache_key_depends_on_params_and_messages(tmp_path):
    client, sdk, _ = make(["a", "b", "c"], tmp_path)
    client.chat(MSGS)
    client.chat(MSGS, temperature=0.5)
    client.chat([{"role": "user", "content": "other"}])
    assert len(sdk.calls) == 3
    assert cache_key("p", "m", MSGS, {}) != cache_key("p", "m2", MSGS, {})


def test_use_cache_false_forces_call(tmp_path):
    client, sdk, _ = make(["a", "b"], tmp_path)
    client.chat(MSGS)
    assert client.chat(MSGS, use_cache=False).content == "b"


def test_request_too_large_is_refused_before_calling(tmp_path):
    sdk = FakeSDK([])
    client = LLMClient(PROVIDER, model="small", cache_dir=tmp_path, sdk=sdk)
    with pytest.raises(RequestTooLarge):
        client.chat(MSGS, max_tokens=100)
    assert sdk.calls == []


def test_invalid_json_reply_raises_with_text(tmp_path):
    client, _, _ = make(["not json"], tmp_path)
    with pytest.raises(LLMError, match="not json"):
        client.chat(MSGS).parse_json()
