"""`grounding.py` -- stage 9e, D9e.7a's extractor and D9e.7b's run-scoped
nudge (`docs/specs/stage-9e.md`).

D9e.7a's own false-positive gate needs a **real** saved report, not a
synthetic sentence: a v1 heuristic (any title-case run) was measured to
fire 15-74 times per real report with no separation between passing and
failing cases. `tests/fixtures/report-core-agent-vs-rag-boundary.md` and
`tests/fixtures/report-core-single-vs-multi-agent.md` are two real
`actualOutput` strings pulled from `runs/4fe7beab-.../eval-results.json`
(phase-1b's own live checkpoint) -- the spec names both explicitly, since a
single fixture is not representative at the measured 4x per-case spread.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from langchain.agents.middleware.types import (
    AgentMiddleware,
    ExtendedModelResponse,
    ModelRequest,
    ModelResponse,
)
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from grounding import UnsupportedClaimMiddleware, extract_candidates

_FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


def _token_attribute(span: Any, name: str) -> int:
    """One integer span attribute, narrowed for mypy.

    `ReadableSpan.attributes` is `Mapping[str, <wide union>] | None`, so
    neither the subscript nor the arithmetic below type-checks without
    stating what this project itself always writes there
    (`TracingMiddleware._record_usage`: a plain int).
    """
    assert span.attributes is not None
    return int(span.attributes[name])


def _fixture_text(name: str) -> str:
    return (_FIXTURES_DIR / name).read_text(encoding="utf-8")


# -- D9e.7a: extract_candidates ----------------------------------------


def test_explicit_version_token_is_a_candidate() -> None:
    assert extract_candidates("Released as v0.4 last month.") == ["v0.4"]


def test_four_digit_year_is_a_candidate() -> None:
    assert extract_candidates("Published in 2026.") == ["2026"]


def test_capitalised_run_with_a_digit_token_qualifies() -> None:
    assert extract_candidates("The AG2 framework won.") == ["AG2"]
    assert extract_candidates("Built on MemGPT-2 internally.") == ["MemGPT-2"]


def test_capitalised_run_that_is_entirely_allcaps_qualifies() -> None:
    assert extract_candidates("Exposed over an SDK.") == ["SDK"]
    assert extract_candidates("Uses AI throughout.") == ["AI"]


def test_bare_title_case_token_alone_does_not_qualify() -> None:
    # The literal example named in docs/specs/stage-9e.md: "OpenAI alone
    # out" -- no digit, not all-caps.
    assert extract_candidates("Built by OpenAI last year.") == []


def test_bare_title_case_run_does_not_qualify() -> None:
    assert extract_candidates("A Comparative Analysis Overview follows.") == []


def test_fenced_code_block_is_skipped_entirely() -> None:
    text = "Before.\n```\nAG2 v9.9 2099\n```\nAfter, no OpenAI here."
    assert extract_candidates(text) == []


def test_markdown_heading_line_is_skipped() -> None:
    text = "# AG2 Overview\n\nBody text with no candidate here."
    assert extract_candidates(text) == []


def test_inline_code_span_is_unwrapped_not_skipped() -> None:
    # Models put fabricated version strings inside inline code -- D9e.7a's
    # own reason for unwrapping rather than skipping.
    assert extract_candidates("Pin the version: `v0.4`.") == ["v0.4"]


def test_bullet_and_bold_and_table_rows_are_not_skipped() -> None:
    text = (
        "- **AG2**: a multi-agent framework.\n"
        "| Tool | Version |\n"
        "|------|---------|\n"
        "| Widget | v1.2 |\n"
    )
    candidates = extract_candidates(text)
    assert "AG2" in candidates
    assert "v1.2" in candidates


def test_duplicates_collapse_preserving_first_seen_order() -> None:
    text = "AG2 appeared. Later, AG2 appeared again, alongside SDK."
    assert extract_candidates(text) == ["AG2", "SDK"]


@pytest.mark.parametrize(
    "fixture_name",
    [
        "report-core-agent-vs-rag-boundary.md",
        "report-core-single-vs-multi-agent.md",
    ],
)
def test_false_positive_ceiling_on_a_real_saved_report(fixture_name: str) -> None:
    # D9e.7a's own measured bound for the revised extractor: 4-17 candidates
    # across real reports, against 15-74 for the rejected v1 heuristic. A
    # generous ceiling, not the exact count -- the point is "not dozens",
    # not reproducing the original session's own script bit for bit.
    candidates = extract_candidates(_fixture_text(fixture_name))
    assert len(candidates) < 30


# -- D9e.7b: UnsupportedClaimMiddleware ----------------------------------


def _model_request(**overrides: Any) -> ModelRequest[Any]:
    messages = overrides.pop("messages", [HumanMessage("Research this")])
    defaults: dict[str, Any] = {
        "model": None,
        "messages": messages,
        "state": {"messages": messages, "grounding_nudge_count": 0},
        "model_settings": {},
    }
    return ModelRequest(**{**defaults, **overrides})


def _draft(text: str) -> ModelResponse[Any]:
    return ModelResponse(result=[AIMessage(content=text)])


def _tool_call_response(name: str = "web_search") -> ModelResponse[Any]:
    return ModelResponse(
        result=[
            AIMessage(
                content="",
                tool_calls=[{"name": name, "args": {"query": "x"}, "id": "c1"}],
            )
        ]
    )


def test_defines_both_hook_variants() -> None:
    # D3.1b: a missing async variant inherits AgentMiddleware's own
    # NotImplementedError, laundered by ToolErrorMiddleware into a
    # plausible-looking failure.
    assert (
        UnsupportedClaimMiddleware.wrap_model_call
        is not AgentMiddleware.wrap_model_call
    )
    assert (
        UnsupportedClaimMiddleware.awrap_model_call
        is not AgentMiddleware.awrap_model_call
    )


def test_nudges_once_when_the_draft_names_an_unsupported_candidate() -> None:
    calls = 0

    def handler(request: ModelRequest[Any]) -> ModelResponse[Any]:
        nonlocal calls
        calls += 1
        if calls == 1:
            return _draft("The AG2 framework handles this.")
        return _draft("Removed the unverified claim.")

    request = _model_request()
    result = UnsupportedClaimMiddleware().wrap_model_call(request, handler)

    assert calls == 2
    assert isinstance(result, ExtendedModelResponse)
    assert result.model_response.result[0].content == "Removed the unverified claim."
    assert result.command is not None
    assert result.command.update == {"grounding_nudge_count": 1}


def test_no_nudge_when_the_candidate_appears_in_a_tool_result() -> None:
    calls = 0
    tool_call = AIMessage(
        content="",
        tool_calls=[{"name": "web_search", "args": {"query": "AG2"}, "id": "c1"}],
    )
    tool_result = ToolMessage(
        content="AG2 is a real framework.", tool_call_id="c1", name="web_search"
    )

    def handler(request: ModelRequest[Any]) -> ModelResponse[Any]:
        nonlocal calls
        calls += 1
        return _draft("The AG2 framework handles this.")

    request = _model_request(
        messages=[HumanMessage("Research this"), tool_call, tool_result]
    )
    result = UnsupportedClaimMiddleware().wrap_model_call(request, handler)

    assert calls == 1
    assert not isinstance(result, ExtendedModelResponse)


def test_no_nudge_when_the_response_calls_a_tool() -> None:
    calls = 0

    def handler(request: ModelRequest[Any]) -> ModelResponse[Any]:
        nonlocal calls
        calls += 1
        return _tool_call_response()

    result = UnsupportedClaimMiddleware().wrap_model_call(_model_request(), handler)

    assert calls == 1
    assert not isinstance(result, ExtendedModelResponse)


def test_no_nudge_when_the_draft_has_no_candidates() -> None:
    calls = 0

    def handler(request: ModelRequest[Any]) -> ModelResponse[Any]:
        nonlocal calls
        calls += 1
        return _draft("A plain summary with nothing distinctive.")

    result = UnsupportedClaimMiddleware().wrap_model_call(_model_request(), handler)

    assert calls == 1
    assert not isinstance(result, ExtendedModelResponse)


def test_cap_stops_nudging_once_the_run_scoped_limit_is_reached() -> None:
    calls = 0

    def handler(request: ModelRequest[Any]) -> ModelResponse[Any]:
        nonlocal calls
        calls += 1
        return _draft("The AG2 framework handles this.")

    request = _model_request(
        state={
            "messages": [HumanMessage("Research this")],
            "grounding_nudge_count": 2,
        }
    )
    result = UnsupportedClaimMiddleware().wrap_model_call(request, handler)

    assert calls == 1  # no retry once the cap is already reached
    assert not isinstance(result, ExtendedModelResponse)


# -- Stage 9e, phase 3 R.2: same shape as CriticVerificationMiddleware's own
# regression in tests/test_middleware.py -- the grounding nudge's discarded
# first response must still get its own `model.<role>` span.


def test_grounding_nudge_retry_does_not_lose_the_first_calls_token_usage() -> None:
    from types import SimpleNamespace

    from langchain_core.messages.ai import UsageMetadata
    from opentelemetry import trace as otel_trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )
    from opentelemetry.util._once import Once

    import middleware

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    otel_trace.set_tracer_provider(provider)
    try:
        first_response = AIMessage(
            content="The AG2 framework handles this.",
            usage_metadata=UsageMetadata(
                input_tokens=100, output_tokens=50, total_tokens=150
            ),
        )
        second_response = AIMessage(
            content="Removed the unverified claim.",
            usage_metadata=UsageMetadata(
                input_tokens=200, output_tokens=80, total_tokens=280
            ),
        )
        scripted = iter([first_response, second_response])
        calls = 0

        def inner_handler(request: ModelRequest[Any]) -> ModelResponse[Any]:
            nonlocal calls
            calls += 1
            return ModelResponse(result=[next(scripted)])

        request = _model_request(
            model=SimpleNamespace(model_name="openai/gpt-4.1-mini"),
        )
        nudger = UnsupportedClaimMiddleware()

        def traced_handler(r: ModelRequest[Any]) -> ModelResponse[Any]:
            # Mirrors langchain's own `_chain_model_call_handlers` inner
            # wrapper (`factory.py`'s `inner_handler`): an
            # `ExtendedModelResponse` is always unwrapped to its plain
            # `ModelResponse` before an outer middleware's own `handler()`
            # call returns it -- `TracingMiddleware` never sees the wrapper
            # type in the real composed graph.
            result = nudger.wrap_model_call(r, inner_handler)
            if isinstance(result, ExtendedModelResponse):
                return result.model_response
            return result

        middleware.TracingMiddleware(role="researcher").wrap_model_call(
            request, traced_handler
        )

        assert calls == 2  # both real, billed model calls happened

        model_spans = [
            s for s in exporter.get_finished_spans() if s.name == "model.researcher"
        ]
        total_input = sum(
            _token_attribute(s, "gen_ai.usage.input_tokens") for s in model_spans
        )
        total_output = sum(
            _token_attribute(s, "gen_ai.usage.output_tokens") for s in model_spans
        )
        assert total_input == 300
        assert total_output == 130
    finally:
        otel_trace._TRACER_PROVIDER = None
        otel_trace._TRACER_PROVIDER_SET_ONCE = Once()


@pytest.mark.asyncio
async def test_async_variant_mirrors_the_sync_one() -> None:
    calls = 0

    async def handler(request: ModelRequest[Any]) -> ModelResponse[Any]:
        nonlocal calls
        calls += 1
        if calls == 1:
            return _draft("The AG2 framework handles this.")
        return _draft("Removed the unverified claim.")

    result = await UnsupportedClaimMiddleware().awrap_model_call(
        _model_request(), handler
    )

    assert calls == 2
    assert isinstance(result, ExtendedModelResponse)
    assert result.command is not None
    assert result.command.update == {"grounding_nudge_count": 1}
