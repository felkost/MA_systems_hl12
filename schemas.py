"""Structured outputs shared by the Planner, the Critic and the report write
path.

Field sets for `ResearchPlan` and `CritiqueResult` are copied verbatim from
the assignment, not designed from scratch -- `sources_to_check` stays
`list[str]` rather than a `Literal`, because its own description allows the
model to answer "both", which a `Literal["knowledge_base", "web"]` would
reject.

`in_scope` is ported from `MA_systems_hl10`, not the original hl8 shape
(`docs/specs/stage-3.md`, D3.3): a research assistant that only ever
produces a plan needs a way to refuse a request that has nothing to
research, and stage 6's golden dataset already commits to measuring that
refusal as a `failure_case`.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# OpenAI's strict structured-output mode refuses any schema that does not
# supply `additionalProperties: false`, and a plain Pydantic model omits the
# key entirely. `extra="forbid"` is what emits it. Strict mode matters
# because without it the provider treats `required` as a hint: a
# `CritiqueResult` really did come back with no `verdict` at all.
_STRUCTURED_OUTPUT_CONFIG = ConfigDict(extra="forbid")


class ResearchPlan(BaseModel):
    """A decomposed research request, produced by the Planner.

    `in_scope` carries **no Python-level default**, on purpose: a pydantic
    field with a default is excluded from the JSON schema's `required` list,
    and OpenAI's strict structured-output mode does not reliably ask the
    model to fill in a field it is free to omit -- hl10 measured the Planner
    setting this correctly in only 1 of 6 real out-of-scope runs while it
    carried a plain default. It stays required in the wire schema; the
    `model_validator` below injects the backward-compatible default of
    `True` for Python-level construction before the required-field check
    runs. When `in_scope` is `False`, `goal` carries the reason the request
    falls outside this system's research-assistant purpose.
    """

    model_config = _STRUCTURED_OUTPUT_CONFIG

    goal: str = Field(description="What we are trying to answer")
    in_scope: bool = Field(
        description=(
            "False when the request is unrelated to this system's "
            "research-assistant purpose -- a personal request, a recipe, "
            "or anything the knowledge base and web search cannot "
            "meaningfully research. When False, `goal` states why. This "
            "field is required: an explicit true or false, decided before "
            "anything else, never left to a default."
        ),
    )
    search_queries: list[str] = Field(
        default_factory=list, description="Specific queries to execute"
    )
    sources_to_check: list[str] = Field(description="'knowledge_base', 'web', or both")
    output_format: str = Field(description="What the final report should look like")

    @model_validator(mode="before")
    @classmethod
    def _default_in_scope_to_true_for_python_construction(cls, data: object) -> object:
        """Runs before field validation, so it does not affect the JSON
        schema's `required` list -- only Python-level construction that
        omits the key defaults to `True` here."""
        if isinstance(data, dict) and "in_scope" not in data:
            return {**data, "in_scope": True}
        return data

    @model_validator(mode="after")
    def _in_scope_plan_names_at_least_one_query(self) -> "ResearchPlan":
        """The `min_length=1` this field used to carry unconditionally --
        now conditioned on `in_scope`, since an out-of-scope plan has
        nothing to search for and forcing a dummy query would be worse than
        allowing none."""
        if self.in_scope and not self.search_queries:
            raise ValueError(
                "search_queries must name at least one query when in_scope "
                "is True -- leave it empty only when in_scope is False"
            )
        return self


class CritiqueResult(BaseModel):
    """The Critic's verdict on one round of research findings."""

    model_config = _STRUCTURED_OUTPUT_CONFIG

    verdict: Literal["APPROVE", "REVISE"]
    is_fresh: bool = Field(
        description="Is the data up-to-date and based on recent sources?"
    )
    is_complete: bool = Field(
        description="Does the research fully cover the user's original request?"
    )
    is_well_structured: bool = Field(
        description="Are findings logically organized and ready for a report?"
    )
    strengths: list[str] = Field(description="What is good about the research")
    gaps: list[str] = Field(
        description="What is missing, outdated, or poorly structured"
    )
    revision_requests: list[str] = Field(
        description="Specific things to fix if verdict is REVISE"
    )


class ReportDraft(BaseModel):
    """A report ready to hand to `save_report`. Ported from hl8 -- absent
    from hl10's `schemas.py` (`docs/specs/stage-3.md`, plan-audit table)."""

    filename: str = Field(description="Name for the saved report file")
    content: str = Field(description="Markdown content of the report")


# Shared by both coordination paths' Researcher input: a coordinator's own
# paraphrase of the user's request drops whatever it did not think mattered
# -- measured in hl8: a coordinator dropping context in five of six
# paraphrases. Forwarding the user's own wording alongside the current task
# text does not depend on either coordinator remembering to.
RESEARCH_INPUT_TEMPLATE = (
    "Original request: {request}\n\nYour task for this round:\n{task}"
)


def render_plan(plan: ResearchPlan) -> str:
    """Render a `ResearchPlan` as compact markdown for the Supervisor.

    Parameters
    ----------
    plan : ResearchPlan
        The Planner's structured response.

    Returns
    -------
    str
        Markdown text. The Supervisor reads this instead of
        `result["structured_response"]` directly, so it never sees raw JSON
        -- which is why an out-of-scope plan renders as an unmistakable
        refusal block instead of a normal layout with empty queries: the
        Supervisor cannot infer refusal from an absence it never sees.
    """
    if not plan.in_scope:
        return f"**Out of scope:** {plan.goal}"
    queries = "\n".join(f"- {query}" for query in plan.search_queries)
    sources = ", ".join(plan.sources_to_check)
    return (
        f"**Goal:** {plan.goal}\n\n"
        f"**Search queries:**\n{queries}\n\n"
        f"**Sources to check:** {sources}\n\n"
        f"**Output format:** {plan.output_format}"
    )


def render_critique(critique: CritiqueResult) -> str:
    """Render a `CritiqueResult` as compact markdown for the Supervisor.

    Parameters
    ----------
    critique : CritiqueResult
        The Critic's structured response.

    Returns
    -------
    str
        Markdown text, with empty `gaps`/`revision_requests` spelled out as
        "none" rather than left as an empty section.
    """
    strengths = _render_list(critique.strengths)
    gaps = _render_list(critique.gaps)
    revision_requests = _render_list(critique.revision_requests)
    return (
        f"**Verdict:** {critique.verdict}\n\n"
        f"**Fresh:** {critique.is_fresh} · "
        f"**Complete:** {critique.is_complete} · "
        f"**Well-structured:** {critique.is_well_structured}\n\n"
        f"**Strengths:**\n{strengths}\n\n"
        f"**Gaps:**\n{gaps}\n\n"
        f"**Revision requests:**\n{revision_requests}"
    )


def _render_list(items: list[str]) -> str:
    if not items:
        return "- none"
    return "\n".join(f"- {item}" for item in items)
