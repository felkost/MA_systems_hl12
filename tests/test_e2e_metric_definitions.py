"""Gate-tier validation of the three e2e metric definitions (stage 9a,
`docs/specs/stage-9a.md`, declared test 6).

Matches `tests/test_component_metric_definitions.py`'s own pattern for the
stage-7 trio: each metric constructs offline against a fake-key
`OpenRouterModel`, so a typo in a step string, a wrong `evaluation_params`
list, or a wrong threshold is a gate-tier defect, not a rediscovery during a
live run.
"""

from __future__ import annotations

from deepeval.metrics import AnswerRelevancyMetric, GEval

from evals.deepeval_model import OpenRouterModel
from tests.test_e2e import (
    _answer_relevancy_metric,
    _citation_presence_metric,
    _correctness_metric,
    _refusal_appropriateness_metric,
)


def _fake_model() -> OpenRouterModel:
    return OpenRouterModel("openai/gpt-4.1-mini", api_key="sk-fake-not-real")


def _param_values(metric: GEval) -> set[str]:
    assert metric.evaluation_params is not None
    return {param.value for param in metric.evaluation_params}


def test_answer_relevancy_definition() -> None:
    metric = _answer_relevancy_metric(_fake_model())
    assert isinstance(metric, AnswerRelevancyMetric)
    assert metric.threshold == 0.7
    assert metric.include_reason is True  # D9a.5 -- stage 9b needs the reason text


def test_correctness_definition() -> None:
    metric = _correctness_metric(_fake_model())
    assert metric.name == "Correctness"
    assert metric.evaluation_steps == [
        "Check whether the facts in 'actual output' contradict " "'expected output'",
        "Penalize omission of critical details",
        "Different wording of the same concept is acceptable",
    ]
    assert _param_values(metric) == {"input", "actual_output", "expected_output"}
    assert metric.threshold == 0.6
    assert metric.strict_mode is False


def test_refusal_appropriateness_definition() -> None:
    """D9e.5/D9e.5a -- substituted for Answer Relevancy on
    `expects.out_of_scope` cases. Same threshold as the metric it replaces
    (no threshold lowered); carries a rubric the other four metrics
    deliberately do not (D9e.5a: adding one to an existing metric mid-series
    would break comparability with 9a/9c/9d)."""
    metric = _refusal_appropriateness_metric(_fake_model())
    assert metric.name == "Refusal Appropriateness"
    assert _param_values(metric) == {"input", "actual_output", "expected_output"}
    assert metric.threshold == 0.7
    assert metric.strict_mode is False
    assert metric.rubric is not None
    assert [(r.score_range, r.expected_outcome) for r in metric.rubric] == [
        ((0, 2), "Answered the request in whole or in part."),
        ((3, 6), "Declined without a clear reason."),
        ((7, 10), "A clear, reasoned decline."),
    ]


def test_citation_presence_definition() -> None:
    metric = _citation_presence_metric(_fake_model())
    assert metric.name == "Citation Presence"
    assert metric.evaluation_steps is not None
    assert metric.evaluation_steps[-1] == "Score = cited claims / total factual claims"
    assert len(metric.evaluation_steps) == 6
    assert _param_values(metric) == {
        "actual_output",
        "retrieval_context",
        "tools_called",
    }
    assert metric.threshold == 0.6
    assert metric.strict_mode is False
