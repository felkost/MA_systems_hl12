"""hl8's registry mechanism carrying hl10's prompt text
(`docs/specs/stage-3.md`, "Why the prompt text must be hl10's").

hl8's own prompt text is wrong for this project on two counts checked here:
it instructs the Researcher to call a tool this project removed
(`graph_search`), and its Supervisor prompt has no out-of-scope path for
D3.3's `in_scope` field to be read by. Both are pinned as standing rules,
not one-time review comments.
"""

from __future__ import annotations

from datetime import date

import pytest

from prompts import (
    build_composer_prompt,
    build_critic_prompt,
    build_planner_prompt,
    build_researcher_prompt,
    build_supervisor_prompt,
)


def test_build_planner_prompt_looks_up_a_registered_version() -> None:
    text = build_planner_prompt("p1")
    assert "Planner" in text


def test_build_planner_prompt_raises_on_unknown_version() -> None:
    with pytest.raises(KeyError, match="p1"):
        build_planner_prompt("p99")


def test_planner_prompt_p1_and_p2_both_registered() -> None:
    p1 = build_planner_prompt("p1")
    p2 = build_planner_prompt("p2")
    assert p1 != p2


def test_planner_prompt_p2_adds_the_incoherence_criterion() -> None:
    # D9e.8 (`p2`): the only genuinely new content over p1 is treating an
    # incoherent request (gibberish, no identifiable topic) as out of scope
    # -- "judging scope is a decision to make immediately" is already in p1
    # verbatim, so it is not what distinguishes p2.
    p1 = " ".join(build_planner_prompt("p1").split())
    p2 = " ".join(build_planner_prompt("p2").split())
    assert "not coherent enough to research" not in p1
    assert "not coherent enough to research" in p2


def test_planner_prompt_p2_no_longer_names_the_golden_recipe_example() -> None:
    # D9e.9: the borscht example is generalised away -- it matched
    # edge-out-of-scope-recipe's own subject verbatim, which is teaching to
    # the test rather than illustrating the rule.
    p1 = build_planner_prompt("p1")
    p2 = build_planner_prompt("p2")
    assert "borscht" in p1
    assert "borscht" not in p2


def test_planner_prompt_p2_still_judges_scope_immediately() -> None:
    text = " ".join(build_planner_prompt("p2").split())
    assert "decision to make immediately" in text


def test_build_researcher_prompt_raises_on_unknown_version() -> None:
    with pytest.raises(KeyError):
        build_researcher_prompt("r99")


def test_build_critic_prompt_raises_on_unknown_version() -> None:
    with pytest.raises(KeyError):
        build_critic_prompt("c99", today=date(2026, 8, 22))


def test_build_supervisor_prompt_raises_on_unknown_version() -> None:
    with pytest.raises(KeyError):
        build_supervisor_prompt("s99")


def test_build_critic_prompt_injects_the_date() -> None:
    text = build_critic_prompt("c2", today=date(2026, 8, 22))
    assert "2026-08-22" in text


def test_researcher_prompt_never_mentions_graph_search() -> None:
    text = build_researcher_prompt("r1")
    assert "graph_search" not in text


def test_supervisor_prompt_has_an_out_of_scope_path() -> None:
    text = build_supervisor_prompt("s1")
    assert "out of scope" in text.lower() or "in_scope" in text.lower()


def test_supervisor_prompt_s1_and_s2_both_registered() -> None:
    s1 = build_supervisor_prompt("s1")
    s2 = build_supervisor_prompt("s2")
    assert s1 != s2


def test_supervisor_prompt_s2_adds_the_no_fabricated_findings_rule() -> None:
    # Pre-phase-4 finding (`insights.md`, `docs/specs/stage-9e.md`): a real
    # run showed the Supervisor receive an obviously broken research result
    # ("INJECTED") and silently invent a plausible-sounding replacement
    # rather than flagging it. `s2` adds the rule against that.
    s1 = " ".join(build_supervisor_prompt("s1").split())
    s2 = " ".join(build_supervisor_prompt("s2").split())
    assert "Never author findings of your own" not in s1
    assert "Never author findings of your own" in s2


def test_supervisor_prompt_s2_still_has_an_out_of_scope_path() -> None:
    text = build_supervisor_prompt("s2")
    assert "out of scope" in text.lower() or "in_scope" in text.lower()


def test_critic_prompt_c1_and_c2_both_registered() -> None:
    c1 = build_critic_prompt("c1", today=date(2026, 8, 22))
    c2 = build_critic_prompt("c2", today=date(2026, 8, 22))
    assert c1 != c2


def test_critic_prompt_c3_adds_the_well_evidenced_absence_rule() -> None:
    # D9e.7 (`c3`): the only new content over c2 is the rule that a
    # well-evidenced absence counts as completeness, not a gap -- discounted
    # by the second clause ("name a source"), already present in c2.
    c2 = " ".join(build_critic_prompt("c2", today=date(2026, 8, 22)).split())
    c3 = " ".join(build_critic_prompt("c3", today=date(2026, 8, 22)).split())
    assert "well-evidenced absence is completeness" not in c2
    assert "well-evidenced absence is completeness" in c3


def test_critic_prompt_c3_keeps_c2s_verdict_boolean_coupling_rule() -> None:
    # c3 must not silently drop the rule Critique Quality's own GEval steps
    # measure against (D3.4) while adding the absence-is-completeness rule.
    c3 = build_critic_prompt("c3", today=date(2026, 8, 22))
    assert "never an APPROVE alongside an unresolved gap" in c3


def test_build_composer_prompt_raises_on_unknown_version() -> None:
    with pytest.raises(KeyError):
        build_composer_prompt("w99")


def test_researcher_prompt_r1_and_r2_both_registered() -> None:
    r1 = build_researcher_prompt("r1")
    r2 = build_researcher_prompt("r2")
    assert r1 != r2


def test_researcher_prompt_r2_warns_against_filling_gaps_with_guesses() -> None:
    r1 = build_researcher_prompt("r1")
    r2 = build_researcher_prompt("r2")
    assert "does not address part of the question" not in r1
    assert "does not address part of the question" in r2


def test_composer_prompt_is_a_separate_registry_from_the_supervisor() -> None:
    # D4.4: a composer prompt must not be reachable through
    # `build_supervisor_prompt` -- registering it in `s*` would let
    # `SUPERVISOR_PROMPT_VERSION="w1"` hand the agent-as-tool Supervisor a
    # composer prompt with no delegation rules.
    composer = build_composer_prompt("w1")
    assert "plan, research, critique" not in composer
    with pytest.raises(KeyError):
        build_supervisor_prompt("w1")
