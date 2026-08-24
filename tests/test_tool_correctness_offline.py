"""Gate-tier validation of `ToolCorrectnessMetric`'s actual scoring
behaviour (stage 8, D8.8).

Same reasoning as the component-metric definition tests: a metric whose
semantics are only ever exercised inside a paid run is a metric nobody has
checked. `ToolCorrectnessMetric` is deterministic when `available_tools` is
not passed -- no judge call is made at all -- so its exact scoring is
checkable offline, for free, and pinned here.

Two of these numbers are genuinely surprising and are pinned so a later
session meets them as a red test rather than as a mystery inside a live run:
a reversed expected pair scores 0.5 rather than 0, and a **duplicated**
expected entry scores 0.5 rather than 1.0 -- the metric matches calls into a
`set()` of value-compared `ToolCall`s, so two field-identical entries
collide and the second can never be credited.
"""

from __future__ import annotations

from deepeval.metrics import ToolCorrectnessMetric
from deepeval.test_case import LLMTestCase, ToolCall

from evals.deepeval_model import OpenRouterModel


def _fake_model() -> OpenRouterModel:
    return OpenRouterModel("openai/gpt-4.1-mini", api_key="sk-fake-not-real")


def _calls(names: list[str]) -> list[ToolCall]:
    # `input_parameters` carries a default, but pydantic's mypy plugin
    # treats an aliased field as required, so it is passed explicitly.
    return [ToolCall(name=name, input_parameters=None) for name in names]


def _score(called: list[str], expected: list[str], *, ordering: bool = False) -> float:
    metric = ToolCorrectnessMetric(
        threshold=0.5,
        model=_fake_model(),
        should_consider_ordering=ordering,
    )
    case = LLMTestCase(
        input="q",
        actual_output="a",
        tools_called=_calls(called),
        expected_tools=_calls(expected),
    )
    # `.measure()` returns None; the score lands on the metric itself.
    metric.measure(case)
    assert metric.score is not None
    return metric.score


def test_constructs_with_an_explicit_model() -> None:
    # Every deepeval metric, including this deterministic one, raises at
    # construction with no `model=` -- which is why the tests above build
    # theirs inside a function rather than at module scope.
    metric = ToolCorrectnessMetric(threshold=0.5, model=_fake_model())
    assert metric.threshold == 0.5


def test_score_is_the_matched_fraction_so_one_of_two_is_exactly_half() -> None:
    # This is what makes threshold=0.5 over a two-tool expectation mean
    # "either of these", which is the brief's own "web_search and/or
    # knowledge_search" expressed as a number.
    assert _score(["knowledge_search"], ["web_search", "knowledge_search"]) == 0.5
    assert (
        _score(["web_search", "knowledge_search"], ["web_search", "knowledge_search"])
        == 1.0
    )


def test_a_missing_expected_tool_scores_zero() -> None:
    assert _score(["web_search"], ["knowledge_search"]) == 0.0


def test_empty_expectations_fail_rather_than_passing_vacuously() -> None:
    assert _score(["web_search"], []) == 0.0


def test_ordering_tolerates_unrelated_calls_around_the_expected_pair() -> None:
    # The Supervisor calls plan and research either side of the pair under
    # test; an ordered-subsequence check must not be confused by them.
    assert (
        _score(
            ["plan", "research", "critique", "save_report"],
            ["critique", "save_report"],
            ordering=True,
        )
        == 1.0
    )


def test_reversed_expected_pair_scores_half_not_zero() -> None:
    assert (
        _score(
            ["plan", "research", "save_report", "critique"],
            ["critique", "save_report"],
            ordering=True,
        )
        == 0.5
    )


def test_duplicate_expected_entries_cannot_both_be_credited() -> None:
    # Value-compared ToolCalls collide in the metric's own matched-set, so a
    # repeated expectation is unsatisfiable however many times the tool ran.
    # No expected_tools list in this project contains a repeated name; this
    # test is what keeps that a decision rather than an accident.
    assert _score(["critique", "critique"], ["critique", "critique"]) == 0.5
