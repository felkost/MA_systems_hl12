"""Offline tests for `evals.deepeval_model.OpenRouterModel`.

No network call: every test injects an `httpx.MockTransport` through the
wrapper's `transport=` constructor parameter, so this suite runs in the gate
tier. The one live call this module needs (a real OpenRouter round trip) is a
separate, explicitly-run smoke check -- never part of
`pytest -m "not smoke and not eval"`.
"""

from __future__ import annotations

import json

import httpx
import pytest
from pydantic import BaseModel

from evals.deepeval_model import OpenRouterModel


class _Verdict(BaseModel):
    verdict: str
    score: float


def _model(
    response_content: str,
    usage: dict[str, int],
    *,
    reasoning_effort: str | None = None,
    captured_payload: dict | None = None,
) -> OpenRouterModel:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer test-key"
        if captured_payload is not None:
            captured_payload.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": response_content}}],
                "usage": usage,
            },
        )

    return OpenRouterModel(
        model_name="openai/gpt-4.1-mini",
        api_key="test-key",
        transport=httpx.MockTransport(handler),
        reasoning_effort=reasoning_effort,
    )


def test_generate_returns_message_content() -> None:
    model = _model(
        "plain text answer",
        {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
    )

    result = model.generate("say something")

    assert result == "plain text answer"


def test_generate_records_usage_because_deepeval_will_not() -> None:
    """DeepEval never calls `_accrue_cost` for a custom (non-native) model --
    `initialize_model` marks it `using_native_model=False` -- so this wrapper
    is the only place that ever sees the provider's real token counts."""
    model = _model(
        "ok", {"prompt_tokens": 123, "completion_tokens": 45, "total_tokens": 168}
    )

    model.generate("anything")

    assert model.last_usage == {
        "prompt_tokens": 123,
        "completion_tokens": 45,
        "total_tokens": 168,
    }


def test_generate_with_schema_returns_validated_instance() -> None:
    payload = json.dumps({"verdict": "PASS", "score": 0.9})
    model = _model(
        payload, {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10}
    )

    result = model.generate_with_schema("judge this", schema=_Verdict)

    assert isinstance(result, _Verdict)
    assert result.verdict == "PASS"
    assert result.score == 0.9


@pytest.mark.asyncio
async def test_a_generate_with_schema_returns_validated_instance() -> None:
    payload = json.dumps({"verdict": "FAIL", "score": 0.1})
    model = _model(
        payload, {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10}
    )

    result = await model.a_generate_with_schema("judge this", schema=_Verdict)

    assert isinstance(result, _Verdict)
    assert result.verdict == "FAIL"


def test_get_model_name_returns_the_configured_id() -> None:
    model = OpenRouterModel(model_name="openai/gpt-4.1-mini", api_key="test-key")

    assert model.get_model_name() == "openai/gpt-4.1-mini"


def test_default_timeout_is_generous_enough_for_a_full_report_judge_call() -> None:
    """Stage 9d D9d.4, raised again at stage 9e phase 1b. A judge call
    carrying a saved report plus a full retrieval context errored at 60s in
    the 9a/9c runs and again at 120s in the 9e phase-1b run
    (`adversarial-direct-jailbreak`, `httpx.ReadTimeout`) -- an errored
    metric carries no score at all, so the case is lost, not merely slow.
    Each raise followed a measured timeout, never a precaution."""
    model = OpenRouterModel("openai/gpt-4.1-mini", api_key="k")
    assert model._timeout == 180.0


def test_client_timeout_leaves_slack_under_deepevals_per_task_budget() -> None:
    """D9e.1's invariant, pinned as arithmetic rather than left in prose:
    `PER_TASK >= n_sequential_judge_calls * httpx_timeout + slack`.
    `AnswerRelevancyMetric` makes 3 sequential calls, so raising either
    number without the other silently reintroduces the cancellation this
    pair exists to prevent."""
    from config import Settings

    per_task = Settings.model_fields["deepeval_per_task_timeout_seconds"].default
    client_timeout = OpenRouterModel("openai/gpt-4.1-mini", api_key="k")._timeout
    sequential_answer_relevancy_calls = 3

    assert per_task > sequential_answer_relevancy_calls * client_timeout


# -- Stage 9e, D9e.16: the judge's thinking budget, controllable and off --


def test_payload_carries_no_reasoning_key_when_unset() -> None:
    """The default (`reasoning_effort=None`) must keep today's payload
    byte-for-byte reproducible -- no `reasoning` key at all."""
    captured: dict = {}
    model = _model(
        "ok",
        {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        captured_payload=captured,
    )

    model.generate("anything")

    assert "reasoning" not in captured


def test_payload_carries_the_exact_reasoning_effort_when_set() -> None:
    captured: dict = {}
    model = _model(
        "ok",
        {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        reasoning_effort="low",
        captured_payload=captured,
    )

    model.generate("anything")

    assert captured["reasoning"] == {"effort": "low"}
