"""The thin LLM interface for stages 9-10: one prompt in, text out, no tools."""

from __future__ import annotations

import json
import re
from typing import Any, Literal, Protocol, runtime_checkable

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


@runtime_checkable
class LlmProvider(Protocol):
    name: str  # "qwen3-8b-4bit-mlx", "bedrock-...", ...
    locality: Literal["local", "cloud"]

    def complete(self, system: str, user: str) -> str:
        """Deterministic (greedy) completion. Callers parse the text; the provider never sees a schema."""
        ...


def parse_json_object(text: str) -> dict[str, Any]:
    """The first-to-last brace span of a reply, after dropping any reasoning block and code fences.

    Raises ValueError when there is no JSON object, so callers can reprompt.
    """
    text = _THINK.sub("", text)
    a, b = text.find("{"), text.rfind("}")
    if a < 0 or b < a:
        raise ValueError("no JSON object in the reply")
    try:
        doc = json.loads(text[a : b + 1])
    except json.JSONDecodeError as e:
        raise ValueError(f"reply is not valid JSON: {e.msg}") from e
    if not isinstance(doc, dict):
        raise ValueError("reply is not a JSON object")
    return doc
