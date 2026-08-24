"""Gate-tier validation of the three component GEval definitions
(stage 7, D7.4).

DeepEval's `GEval` (`deepeval==4.1.10`, measured) constructs cleanly with
neither `criteria` nor `evaluation_steps`, and with no `evaluation_params`
at all -- it only raises inside `.measure()`. A misconfigured metric would
therefore surface only inside a **paid** eval-tier run. This test builds
each metric with a fake, offline `OpenRouterModel` (construction is lazy --
measured, zero socket calls) and checks its definition directly, so a typo
in a step string or a wrong `evaluation_params` list is a gate-tier defect,
not a rediscovery during a live run.
"""

from __future__ import annotations

from deepeval.metrics import GEval

from evals.deepeval_model import OpenRouterModel
from tests.test_critic import _critique_quality_metric
from tests.test_planner import _plan_quality_metric
from tests.test_researcher import _groundedness_metric


def _fake_model() -> OpenRouterModel:
    return OpenRouterModel("openai/gpt-4.1-mini", api_key="sk-fake-not-real")


def _param_values(metric: GEval) -> set[str]:
    # `evaluation_params` is typed `list[SingleTurnParams] | None` because
    # GEval also permits building one with no params at all (measured,
    # docs/specs/stage-7.md M2) -- never true for the three factories under
    # test here, which all pass a concrete list.
    assert metric.evaluation_params is not None
    return {param.value for param in metric.evaluation_params}


def _threshold(metric: GEval) -> float:
    assert metric.threshold is not None
    return metric.threshold


def test_plan_quality_definition() -> None:
    metric = _plan_quality_metric(_fake_model())
    assert metric.name == "Plan Quality"
    assert metric.evaluation_steps == [
        "Check that the plan contains specific search queries (not vague)",
        "Check that sources_to_check includes relevant sources for the topic",
        "Check that the output_format matches what the user asked for",
    ]
    assert _param_values(metric) == {"input", "actual_output"}
    assert _threshold(metric) >= 0.7
    assert metric.strict_mode is False


def test_critique_quality_definition() -> None:
    metric = _critique_quality_metric(_fake_model())
    assert metric.name == "Critique Quality"
    assert metric.evaluation_steps == [
        "Check that the critique identifies specific issues, not vague complaints",
        "Check that revision_requests are actionable (researcher can act on them)",
        "If verdict is APPROVE, gaps list should be empty or contain only minor items",
        "If verdict is REVISE, there must be at least one revision_request",
    ]
    assert _param_values(metric) == {"input", "actual_output"}
    assert _threshold(metric) >= 0.7
    assert metric.strict_mode is False


def test_groundedness_definition() -> None:
    metric = _groundedness_metric(_fake_model())
    assert metric.name == "Groundedness"
    assert metric.evaluation_steps == [
        "Extract every factual claim from 'actual output'",
        "For each claim, check if it can be directly supported by 'retrieval context'",
        "Claims not present in retrieval context count as ungrounded, even if true",
        "Score = number of grounded claims / total claims",
    ]
    assert _param_values(metric) == {"actual_output", "retrieval_context"}
    assert _threshold(metric) >= 0.7
    assert metric.strict_mode is False
