"""`prompts.PROMPT_NAMES` -- the name/label registry, not prompt text.

Before this project's prompt-store rewrite, this file asserted on prompt
*text* registered in `prompts.py` directly (hl11's `PLANNER_PROMPTS`/...
registries). That text now lives only in Langfuse; nothing in this project
reads it back out of a `.py` file, so nothing here can assert on it either
-- see `tests/test_no_hardcoded_prompts.py` for the structural guard that
a prompt string never reappears in code.
"""

from __future__ import annotations

from config import Settings
from prompts import CRITIC_PROMPT_VARIABLES, PROMPT_NAMES


def test_prompt_names_cover_every_agent_role() -> None:
    for role in Settings.ROLES:
        assert role in PROMPT_NAMES, f"no Langfuse prompt name registered for {role!r}"


def test_prompt_names_also_cover_the_composer_and_critic_verification() -> None:
    # Not in Settings.ROLES: `composer` is the orchestrator path's own
    # report-writing prompt (no dedicated model role -- it reuses the
    # "supervisor" role's model), and `critic_verification` is
    # `CriticVerificationMiddleware`'s retry instruction, not an agent's
    # own system prompt at all.
    assert "composer" in PROMPT_NAMES
    assert "critic_verification" in PROMPT_NAMES


def test_prompt_names_are_unique() -> None:
    names = list(PROMPT_NAMES.values())
    assert len(names) == len(set(names)), f"duplicate Langfuse prompt name: {names}"


def test_every_prompt_name_is_hl12_prefixed() -> None:
    # A stray hl11-prefixed (or unprefixed) name here would fetch a
    # different project's prompt -- or nothing -- from the shared Langfuse
    # organisation. Every prompt name this project uses must be hl12-prefixed.
    for role, name in PROMPT_NAMES.items():
        assert name.startswith("hl12-"), f"{role!r}: {name!r} is not hl12-prefixed"


def test_critic_prompt_variables_is_exactly_today() -> None:
    assert CRITIC_PROMPT_VARIABLES == frozenset({"today"})
