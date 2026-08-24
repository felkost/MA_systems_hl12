"""`agents._allowlist` is the one place a factory checks what it was handed.

Under stage 3's dependency-inversion design (`docs/specs/stage-3.md`, D3.2),
an agent factory never imports `tools.py` -- it receives `tools` from its
caller. That makes a caller's mistake a real risk a Planner handed
`read_url` would silently gain a capability its architecture row denies it
(D3.5). This module is the guard, shared by all three factories rather than
duplicated three times.
"""

from __future__ import annotations

import pytest
from langchain_core.tools import tool

from agents._allowlist import ToolAllowlistError, assert_allowlist


@tool("web_search")
def _web_search(query: str) -> str:
    """Fake web search."""
    return query


@tool("read_url")
def _read_url(url: str) -> str:
    """Fake read_url."""
    return url


@tool("knowledge_search")
def _knowledge_search(query: str) -> str:
    """Fake knowledge search."""
    return query


def test_assert_allowlist_accepts_an_exact_match() -> None:
    assert_allowlist(
        [_web_search, _knowledge_search], ("web_search", "knowledge_search"), "planner"
    )


def test_assert_allowlist_refuses_an_unexpected_tool() -> None:
    with pytest.raises(ToolAllowlistError, match="read_url"):
        assert_allowlist(
            [_web_search, _read_url, _knowledge_search],
            ("web_search", "knowledge_search"),
            "planner",
        )


def test_assert_allowlist_refuses_a_missing_expected_tool() -> None:
    with pytest.raises(ToolAllowlistError, match="knowledge_search"):
        assert_allowlist([_web_search], ("web_search", "knowledge_search"), "planner")


def test_assert_allowlist_error_names_the_agent() -> None:
    with pytest.raises(ToolAllowlistError, match="planner"):
        assert_allowlist([_read_url], ("web_search",), "planner")
