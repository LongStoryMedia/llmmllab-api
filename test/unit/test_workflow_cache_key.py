"""
Unit tests for the workflow cache key logic in core/service.py.

Validates that ``_tools_cache_key`` produces stable, deterministic hashes
from tool definitions (both OpenAI dict format and LangChain BaseTool
objects) and that ``ComposerService.compose_workflow`` correctly includes
tools in the cache key.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from graph.cache_utils import _tools_cache_key


class TestToolsCacheKey:

    def test_empty_list(self):
        """Empty tools list produces a deterministic key."""
        key = _tools_cache_key([])
        assert isinstance(key, str)
        assert len(key) == 12

    def test_deterministic_for_same_tools(self):
        """Same tool set always produces the same key."""
        tools = [
            {"type": "function", "function": {"name": "read_file"}},
            {"type": "function", "function": {"name": "write_file"}},
        ]
        k1 = _tools_cache_key(tools)
        k2 = _tools_cache_key(tools)
        assert k1 == k2

    def test_order_independent(self):
        """Tool order does not affect the cache key."""
        tools_a = [
            {"type": "function", "function": {"name": "read_file"}},
            {"type": "function", "function": {"name": "write_file"}},
        ]
        tools_b = [
            {"type": "function", "function": {"name": "write_file"}},
            {"type": "function", "function": {"name": "read_file"}},
        ]
        assert _tools_cache_key(tools_a) == _tools_cache_key(tools_b)

    def test_different_tools_different_key(self):
        """Different tool sets produce different keys."""
        tools_a = [{"type": "function", "function": {"name": "read_file"}}]
        tools_b = [{"type": "function", "function": {"name": "delete_file"}}]
        assert _tools_cache_key(tools_a) != _tools_cache_key(tools_b)

    def test_langchain_basetool_objects(self):
        """Handles objects with a ``.name`` attribute (LangChain BaseTool)."""
        tool_a = MagicMock()
        tool_a.name = "read_file"
        tool_b = MagicMock()
        tool_b.name = "write_file"

        key = _tools_cache_key([tool_a, tool_b])
        assert isinstance(key, str)
        assert len(key) == 12

    def test_mixed_dict_and_object(self):
        """Handles a mixture of dict and BaseTool-like objects."""
        tool_dict = {"type": "function", "function": {"name": "read_file"}}
        tool_obj = MagicMock()
        tool_obj.name = "write_file"

        key = _tools_cache_key([tool_dict, tool_obj])
        assert isinstance(key, str)
        assert len(key) == 12

    def test_many_tools_produces_stable_key(self):
        """49 tools (typical Claude Code session) still produces a 12-char key."""
        tools = [
            {"type": "function", "function": {"name": f"tool_{i}"}}
            for i in range(49)
        ]
        key = _tools_cache_key(tools)
        assert len(key) == 12
        # Calling again gives the same key
        assert _tools_cache_key(tools) == key

    def test_superset_different_from_subset(self):
        """Adding a tool changes the key."""
        tools_base = [
            {"type": "function", "function": {"name": "read_file"}},
            {"type": "function", "function": {"name": "write_file"}},
        ]
        tools_extra = tools_base + [
            {"type": "function", "function": {"name": "delete_file"}},
        ]
        assert _tools_cache_key(tools_base) != _tools_cache_key(tools_extra)
