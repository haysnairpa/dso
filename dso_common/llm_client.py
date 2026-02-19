from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from openai import OpenAI


@dataclass(frozen=True)
class LLMConfig:
    provider: str
    model: str
    base_url: Optional[str] = None


def build_llm_client(*, provider: str) -> OpenAI:
    p = (provider or "").strip().lower()
    if p == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required when LLM_PROVIDER=openai")
        return OpenAI(api_key=api_key)

    if p == "fireworks":
        api_key = os.environ.get("FIREWORKS_API_KEY")
        if not api_key:
            raise ValueError("FIREWORKS_API_KEY is required when LLM_PROVIDER=fireworks")
        base_url = os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1")
        return OpenAI(api_key=api_key, base_url=base_url)

    raise ValueError(f"Unknown LLM provider: {provider}. Use 'openai' or 'fireworks'.")


def chat_json(
    *,
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.0,
    max_tokens: int | None = None,
) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
    }
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens

    resp = client.chat.completions.create(**kwargs)
    content = resp.choices[0].message.content
    if not content:
        raise ValueError("LLM returned empty content")

    import json

    return json.loads(content)
