"""Stage-4 spec additions to `Settings`: `composer_prompt_version` and the
derived `resolved_supervisor_max_tool_calls` (D4.4, D4.13).
"""

from __future__ import annotations

from typing import Any

from pydantic import SecretStr

from config import Settings


def _settings(**overrides: Any) -> Settings:
    return Settings(openrouter_api_key=SecretStr("test-key"), **overrides)


def test_composer_prompt_defaults_to_w1() -> None:
    assert _settings().composer_prompt_version == "w1"


def test_supervisor_max_tool_calls_defaults_to_none() -> None:
    assert _settings().supervisor_max_tool_calls is None


def test_resolved_supervisor_max_tool_calls_is_derived_from_max_revisions() -> None:
    # 1 (plan) + 2*(max_revisions+1) (research+critique) + 1 (save_report) + 3
    assert _settings(max_revisions=1).resolved_supervisor_max_tool_calls() == 9
    assert _settings(max_revisions=2).resolved_supervisor_max_tool_calls() == 11
    assert _settings(max_revisions=3).resolved_supervisor_max_tool_calls() == 13


def test_resolved_supervisor_max_tool_calls_respects_an_explicit_override() -> None:
    settings = _settings(max_revisions=2, supervisor_max_tool_calls=50)
    assert settings.resolved_supervisor_max_tool_calls() == 50
