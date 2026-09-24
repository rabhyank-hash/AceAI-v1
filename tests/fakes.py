"""Fake OpenAI SDK for offline tests: no test may call a real API."""

from types import SimpleNamespace

import httpx2
import openai


def completion(content="ok"):
    msg = SimpleNamespace(content=content, tool_calls=None)
    usage = SimpleNamespace(model_dump=lambda: {"total_tokens": 3})
    choice = SimpleNamespace(message=msg, finish_reason="stop")
    return SimpleNamespace(choices=[choice], usage=usage)


def rate_limit_error(headers):
    req = httpx2.Request("POST", "http://fake/v1/chat/completions")
    resp = httpx2.Response(429, headers=headers, request=req)
    return openai.RateLimitError("rate limited", response=resp, body=None)


class FakeSDK:
    """Replays `script`: each item is an exception to raise or a content string to return."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(with_raw_response=SimpleNamespace(create=self._create))
        )

    def _create(self, **kw):
        self.calls.append(kw)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return SimpleNamespace(
            headers={"x-ratelimit-remaining-tokens": "11000", "content-type": "json"},
            parse=lambda: completion(item),
        )
