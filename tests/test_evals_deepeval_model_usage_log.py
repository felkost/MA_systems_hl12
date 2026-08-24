"""Offline tests for `OpenRouterModel`'s optional usage accumulator (stage
9a, `docs/specs/stage-9a.md` D9a.7).

No network call -- every test injects an `httpx.MockTransport`, matching
`tests/test_deepeval_model.py`'s own pattern.
"""

from __future__ import annotations

import httpx

from evals.deepeval_model import OpenRouterModel


def _model(usage: dict[str, int], **kwargs: object) -> OpenRouterModel:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "ok"}}], "usage": usage},
        )

    return OpenRouterModel(
        model_name="openai/gpt-4.1-mini",
        api_key="test-key",
        transport=httpx.MockTransport(handler),
        **kwargs,  # type: ignore[arg-type]
    )


def test_usage_log_defaults_to_none_and_last_usage_is_unaffected() -> None:
    model = _model({"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14})

    model.generate("first call")

    assert model.usage_log is None
    assert model.last_usage == {
        "prompt_tokens": 10,
        "completion_tokens": 4,
        "total_tokens": 14,
    }


def test_usage_log_accumulates_across_calls_in_order_when_given() -> None:
    shared_log: list[dict[str, int]] = []
    model = _model(
        {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
        usage_log=shared_log,
    )

    model.generate("first call")
    model.generate("second call")

    assert shared_log == [
        {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
        {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
    ]
    assert model.usage_log is shared_log
    # last_usage keeps its own per-call overwrite behaviour, unaffected by
    # the accumulator existing alongside it.
    assert model.last_usage == {
        "prompt_tokens": 10,
        "completion_tokens": 4,
        "total_tokens": 14,
    }


def test_usage_log_is_shared_across_two_model_instances_given_the_same_list() -> None:
    """The design this stage relies on (D9a.7): one instance is reused across
    45 metric measurements, but the log itself is just a plain list a caller
    can pass to more than one model if it ever needed to."""
    shared_log: list[dict[str, int]] = []
    first = _model(
        {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        usage_log=shared_log,
    )
    second = _model(
        {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4},
        usage_log=shared_log,
    )

    first.generate("a")
    second.generate("b")

    assert shared_log == [
        {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4},
    ]
