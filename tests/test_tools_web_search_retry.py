"""Stage 9d D9d.3: `web_search` retries a transient DuckDuckGo failure.

The failure path **returns** an error string rather than raising, so
`ToolRetryMiddleware` -- which retries exceptions -- never sees it. Every
DDG rate-limit was therefore one attempt and done, measured across the
stage-9a and stage-9c live runs (9 of 15 e2e runs hit
`ERROR: Web search is temporarily unavailable` in stage 9a alone). The
retry lives in the tool because that is the only layer that can see this
failure at all.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import SecretStr

import tools
from config import Settings


@pytest.fixture(autouse=True)
def _fast_and_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """No real sleeping, no real settings read (CI has no API key)."""
    monkeypatch.setattr(tools.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(
        tools,
        "load_settings",
        lambda: Settings(openrouter_api_key=SecretStr("test-key")),
    )


class _FlakyDDGS:
    """Fails `failures` times, then returns one result."""

    calls = 0

    def __init__(self, failures: int) -> None:
        self._failures = failures

    def text(self, *args: Any, **kwargs: Any) -> list[dict[str, str]]:
        type(self).calls += 1
        if type(self).calls <= self._failures:
            raise RuntimeError("ratelimited")
        return [{"title": "T", "href": "https://example.com", "body": "B"}]


def _install(monkeypatch: pytest.MonkeyPatch, failures: int) -> type[_FlakyDDGS]:
    _FlakyDDGS.calls = 0
    monkeypatch.setattr(tools, "DDGS", lambda: _FlakyDDGS(failures))
    return _FlakyDDGS


def test_web_search_retries_a_transient_failure_and_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ddgs = _install(monkeypatch, failures=2)
    result = tools.web_search.func("agent architectures")  # type: ignore[attr-defined]
    assert isinstance(result, list)
    assert result[0]["url"] == "https://example.com"
    assert ddgs.calls == 3


def test_web_search_gives_up_after_three_attempts_with_the_same_error_string(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ddgs = _install(monkeypatch, failures=99)
    result = tools.web_search.func("agent architectures")  # type: ignore[attr-defined]
    assert result == "ERROR: Web search is temporarily unavailable."
    assert ddgs.calls == 3


def test_web_search_does_not_retry_an_empty_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ddgs = _install(monkeypatch, failures=0)
    result = tools.web_search.func("   ")  # type: ignore[attr-defined]
    assert result == "ERROR: Search query cannot be empty."
    assert ddgs.calls == 0
