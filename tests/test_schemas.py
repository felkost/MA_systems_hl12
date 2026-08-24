"""`ResearchPlan.in_scope` -- adopted from `MA_systems_hl10`, not hl8.

hl10 measured that a plain Python default made the Planner set this field
correctly in only 1 of 6 real out-of-scope runs, which is why it carries no
schema-level default and instead uses a `model_validator(mode="before")`
that only fills the field in for old-style Python construction, never for
the wire schema OpenAI's strict mode reads.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from schemas import (
    CritiqueResult,
    ReportDraft,
    ResearchPlan,
    render_critique,
    render_plan,
)


def test_research_plan_in_scope_defaults_true_for_python_construction() -> None:
    # `model_validate` on a plain dict, not the keyword constructor: the
    # default-injecting validator only fires for dict-style construction
    # (e.g. a caller passing parsed JSON), which is also the only call shape
    # `in_scope` being absent from Python's own required-argument checking
    # would otherwise mask.
    plan = ResearchPlan.model_validate(
        {
            "goal": "Compare A and B",
            "search_queries": ["A vs B"],
            "sources_to_check": ["web"],
            "output_format": "narrative",
        }
    )
    assert plan.in_scope is True


def test_research_plan_out_of_scope_allows_empty_queries() -> None:
    plan = ResearchPlan(
        goal="Not a research question",
        in_scope=False,
        sources_to_check=[],
        output_format="",
    )
    assert plan.search_queries == []


def test_research_plan_in_scope_requires_at_least_one_query() -> None:
    with pytest.raises(ValidationError, match="search_queries"):
        ResearchPlan(
            goal="Compare A and B",
            in_scope=True,
            search_queries=[],
            sources_to_check=["web"],
            output_format="narrative",
        )


def test_render_plan_out_of_scope_is_a_refusal_block() -> None:
    plan = ResearchPlan(
        goal="This asks for a recipe, not research",
        in_scope=False,
        sources_to_check=[],
        output_format="",
    )
    rendered = render_plan(plan)
    assert "Out of scope" in rendered
    assert "Search queries" not in rendered


def test_render_plan_in_scope_shows_the_normal_layout() -> None:
    plan = ResearchPlan(
        goal="Compare A and B",
        in_scope=True,
        search_queries=["A vs B"],
        sources_to_check=["web"],
        output_format="narrative",
    )
    rendered = render_plan(plan)
    assert "Goal:" in rendered
    assert "Search queries:" in rendered


def test_report_draft_carries_filename_and_content() -> None:
    draft = ReportDraft(filename="report.md", content="# Report")
    assert draft.filename == "report.md"
    assert draft.content == "# Report"


def test_render_critique_lists_empty_sections_as_none() -> None:
    critique = CritiqueResult(
        verdict="APPROVE",
        is_fresh=True,
        is_complete=True,
        is_well_structured=True,
        strengths=["clear citations"],
        gaps=[],
        revision_requests=[],
    )
    rendered = render_critique(critique)
    assert "Verdict:** APPROVE" in rendered
    assert "none" in rendered
