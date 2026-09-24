"""Send one tiny request to check the key, model, and rate-limit headers.

python scripts/llm_smoke.py [--provider groq] [--model NAME]
"""

from __future__ import annotations

import argparse
import json

from aceai.config import DEFAULT_PROVIDER
from aceai.llm.client import LLMClient


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default=DEFAULT_PROVIDER)
    ap.add_argument("--model", default=None)
    args = ap.parse_args()
    client = LLMClient(args.provider, args.model, cache_dir=None)
    r = client.chat(
        [{"role": "user", "content": 'Reply with the JSON object {"ok": true} and nothing else.'}],
        json_mode=True,
        max_tokens=20,
    )
    print(f"{client.provider.name} / {client.model}: {r.content!r} (attempts {r.attempts})")
    print("usage:", r.usage)
    print("rate-limit headers:", json.dumps(r.rate_limit_headers, indent=2))


if __name__ == "__main__":
    main()
