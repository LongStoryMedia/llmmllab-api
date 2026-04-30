"""Lightweight helpers for workflow cache key generation.

Kept in a separate module to avoid circular imports when testing.
"""

import hashlib
from typing import Any, List


def _tools_cache_key(tools: List[Any]) -> str:
    """Produce a short, stable hash from a list of tool definitions.

    Accepts LangChain ``BaseTool`` instances (have a ``.name`` attribute)
    as well as raw OpenAI-format dicts (``{"type":"function","function":{"name":...}}``).
    """
    names: list[str] = []
    for t in tools:
        if isinstance(t, dict):
            # OpenAI dict format: {"type": "function", "function": {"name": "..."}}
            fn = t.get("function", {})
            names.append(fn.get("name", "") if isinstance(fn, dict) else "")
        elif hasattr(t, "name"):
            names.append(t.name)
    return hashlib.md5(",".join(sorted(names)).encode()).hexdigest()[:12]
