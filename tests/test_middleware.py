"""`middleware.py` -- ported from `MA_systems_hl10` with two corrections
(`docs/specs/stage-3.md`): D3.1b's precise hook contract, and D3.7's rename
of the two Supervisor-only classes onto this project's own tool names
(`plan`/`research`/`critique`, not hl10's `delegate_to_*`).
"""

from __future__ import annotations

import json
from typing import Any, cast

import pytest
from langchain.agents.middleware import (
    ModelCallLimitMiddleware,
    ModelRetryMiddleware,
    ToolCallLimitMiddleware,
    ToolErrorMiddleware,
    ToolRetryMiddleware,
)
from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
)
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest

import middleware
from tests.fakes import FakeToolCallingModel


def _middleware_classes() -> list[type[AgentMiddleware[Any, Any, Any]]]:
    return [
        obj
        for obj in vars(middleware).values()
        if isinstance(obj, type)
        and issubclass(obj, AgentMiddleware)
        and obj is not AgentMiddleware
    ]


# -- D3.1b: override detected by identity, not inspect.iscoroutinefunction --
# Measured: the *base* AgentMiddleware.awrap_tool_call is itself an async def
# that raises NotImplementedError, so iscoroutinefunction(base.awrap_tool_call)
# is True -- it cannot distinguish "overridden" from "inherited". The correct
# primitive is identity comparison against the base class attribute, the same
# one langchain's own factory uses to detect overrides.


@pytest.mark.parametrize("cls", _middleware_classes(), ids=lambda c: c.__name__)
def test_custom_middleware_define_both_hook_variants(
    cls: type[AgentMiddleware[Any, Any, Any]],
) -> None:
    pairs = [
        ("wrap_model_call", "awrap_model_call"),
        ("wrap_tool_call", "awrap_tool_call"),
    ]
    for sync_name, async_name in pairs:
        sync_overridden = getattr(cls, sync_name) is not getattr(
            AgentMiddleware, sync_name
        )
        async_overridden = getattr(cls, async_name) is not getattr(
            AgentMiddleware, async_name
        )
        assert sync_overridden == async_overridden, (
            f"{cls.__name__} overrides {sync_name} ({sync_overridden}) and "
            f"{async_name} ({async_overridden}) inconsistently -- the "
            "missing variant inherits the base's NotImplementedError, which "
            "ToolErrorMiddleware launders into a plausible-looking failure"
        )


# -- Test 11: CriticVerificationMiddleware retries exactly once --
#
# Unit-tested directly against `wrap_model_call`/`awrap_model_call` rather
# than through a full `create_agent` invocation: routing the retry's forced
# tool call through the real graph would execute the actual `web_search`
# tool (a live DuckDuckGo call) and, with no further scripted response to
# terminate the run, loop until `GraphRecursionError` -- measured. The
# middleware's own contract ("the retried response is returned as-is,
# whatever it contains") is exactly what a direct handler-call check proves,
# without needing a real tool node at all.


def _model_request(**overrides: Any) -> ModelRequest[Any]:
    defaults: dict[str, Any] = {
        "model": FakeToolCallingModel(responses=[AIMessage(content="")]),
        "messages": [HumanMessage("Verify these findings")],
        "state": {"messages": [HumanMessage("Verify these findings")]},
        "model_settings": {},
    }
    return ModelRequest(**{**defaults, **overrides})


def test_critic_verification_middleware_retries_once_when_no_tool_was_called() -> None:
    calls = 0

    def handler(request: ModelRequest[Any]) -> ModelResponse[Any]:
        nonlocal calls
        calls += 1
        return ModelResponse(result=[AIMessage(content="no verification call")])

    result = middleware.CriticVerificationMiddleware().wrap_model_call(
        _model_request(), handler
    )
    assert calls == 2
    assert result.result[0].content == "no verification call"


def test_critic_verification_middleware_does_not_retry_when_a_tool_was_called() -> None:
    calls = 0

    def handler(request: ModelRequest[Any]) -> ModelResponse[Any]:
        nonlocal calls
        calls += 1
        return ModelResponse(
            result=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "web_search", "args": {"query": "x"}, "id": "c1"}
                    ],
                )
            ]
        )

    middleware.CriticVerificationMiddleware().wrap_model_call(_model_request(), handler)
    assert calls == 1


def test_critic_verification_middleware_skips_retry_if_already_verified() -> None:
    calls = 0
    prior_ai = AIMessage(
        content="",
        tool_calls=[{"name": "knowledge_search", "args": {"query": "x"}, "id": "c0"}],
    )

    def handler(request: ModelRequest[Any]) -> ModelResponse[Any]:
        nonlocal calls
        calls += 1
        return ModelResponse(result=[AIMessage(content="verdict, no new tool call")])

    request = _model_request(state={"messages": [HumanMessage("q"), prior_ai]})
    middleware.CriticVerificationMiddleware().wrap_model_call(request, handler)
    assert calls == 1


# -- Test 12: agent_middleware() order and retry-on-failure --


def test_agent_middleware_stack_order_and_retry_on_failure() -> None:
    stack = middleware.agent_middleware(tool_call_limit=5, role="test")
    kinds = [type(m) for m in stack]
    assert kinds == [
        ModelCallLimitMiddleware,
        ToolCallLimitMiddleware,
        ToolErrorMiddleware,
        ToolRetryMiddleware,
        ModelRetryMiddleware,
        middleware.TracingMiddleware,
    ]
    retry = next(m for m in stack if isinstance(m, ToolRetryMiddleware))
    assert retry.on_failure == "error"


def test_agent_middleware_respects_the_tool_call_limit_argument() -> None:
    stack = middleware.agent_middleware(tool_call_limit=7, role="test")
    limiter = next(m for m in stack if isinstance(m, ToolCallLimitMiddleware))
    assert limiter.run_limit == 7


def test_agent_middleware_appends_a_tracing_middleware_for_the_given_role() -> None:
    stack = middleware.agent_middleware(tool_call_limit=5, role="researcher")
    tracer = next(m for m in stack if isinstance(m, middleware.TracingMiddleware))
    assert tracer.role == "researcher"


# -- Stage 5, D5.10: TracingMiddleware -- model-call spans carry token usage
# and cost; tool-call spans carry JSON-serialized args, never a raw dict
# (OTel silently drops a dict-valued span attribute, measured against the
# installed opentelemetry-sdk -- docs/specs/stage-5.md, D5.10 first
# correction).


def _recording_provider() -> tuple[Any, Any]:
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider, exporter


def test_tracing_middleware_records_token_usage_and_cost_on_model_call() -> None:
    from types import SimpleNamespace

    from langchain_core.messages.ai import UsageMetadata
    from opentelemetry import trace as otel_trace

    provider, exporter = _recording_provider()
    otel_trace.set_tracer_provider(provider)
    try:
        ai_message = AIMessage(
            content="answer",
            usage_metadata=UsageMetadata(
                input_tokens=1000, output_tokens=500, total_tokens=1500
            ),
        )
        # wrap_model_call reads the model name off `request.model` and never
        # invokes it directly (the handler does) -- a bare stand-in with
        # `model_name` is enough, matching how `ChatOpenAI.model_name`
        # resolves in production (models.py, measured this session).
        request = _model_request(
            model=SimpleNamespace(model_name="openai/gpt-4.1-mini"),
        )
        handler = lambda r: ModelResponse(result=[ai_message])  # noqa: E731
        middleware.TracingMiddleware(role="researcher").wrap_model_call(
            request, handler
        )
        spans = exporter.get_finished_spans()
        model_span = next(s for s in spans if s.name == "model.researcher")
        assert model_span.attributes["gen_ai.usage.input_tokens"] == 1000
        assert model_span.attributes["gen_ai.usage.output_tokens"] == 500
        assert model_span.attributes["gen_ai.usage.cost_usd"] == pytest.approx(
            1000 * 0.0000004 + 500 * 0.0000016
        )
        # D9e.20: Langfuse only populates its own native model/cost columns
        # from its own attribute names (`langfuse/_client/attributes.py`),
        # not from the `gen_ai.*` OTel pair alone -- measured live, stage 9e
        # phase 1b: every model.* span reached Langfuse with the right
        # figures sitting unread in `metadata`, and `providedModelName`/
        # `costDetails`/`totalCost` empty.
        assert model_span.attributes["gen_ai.request.model"] == "openai/gpt-4.1-mini"
        assert (
            model_span.attributes["langfuse.observation.model.name"]
            == "openai/gpt-4.1-mini"
        )
        cost_details = json.loads(
            model_span.attributes["langfuse.observation.cost_details"]
        )
        assert cost_details["total"] == pytest.approx(
            1000 * 0.0000004 + 500 * 0.0000016
        )
    finally:
        otel_trace._TRACER_PROVIDER = None
        from opentelemetry.util._once import Once

        otel_trace._TRACER_PROVIDER_SET_ONCE = Once()


# -- Stage 9e, phase 3 R.2: a retry inside `wrap_model_call` must not erase
# the discarded call's own token usage. `TracingMiddleware`'s own
# `model.<role>` span wraps the *composed* `handler()` call, so when an
# inner middleware (`CriticVerificationMiddleware`, `SaveReportGuardMiddleware`,
# `grounding.UnsupportedClaimMiddleware`) retries by calling `handler()` a
# second time, `TracingMiddleware` only ever sees the final response --
# verified offline this stage with exactly this nesting (`TracingMiddleware`
# outer, a one-shot retry middleware inner, a fake model returning two
# distinct `usage_metadata` payloads), producing one `model.<role>` span
# carrying only the second call's counts (`docs/specs/stage-9e.md`'s dated
# R.2 line, "phase 3, H3a/H3b/H3c probes run"). The fix: the retrying
# middleware itself records the first (discarded) response's usage via
# `middleware.record_superseded_model_call` before retrying, so the two
# real calls together produce two spans.


def test_critic_verification_retry_does_not_lose_the_first_calls_token_usage() -> None:
    from types import SimpleNamespace

    from langchain_core.messages.ai import UsageMetadata
    from opentelemetry import trace as otel_trace

    provider, exporter = _recording_provider()
    otel_trace.set_tracer_provider(provider)
    try:
        first_response = AIMessage(
            content="no verification call",
            usage_metadata=UsageMetadata(
                input_tokens=100, output_tokens=50, total_tokens=150
            ),
        )
        second_response = AIMessage(
            content="",
            tool_calls=[{"name": "web_search", "args": {"query": "x"}, "id": "c1"}],
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
        critic = middleware.CriticVerificationMiddleware()

        def traced_handler(r: ModelRequest[Any]) -> ModelResponse[Any]:
            return critic.wrap_model_call(r, inner_handler)

        middleware.TracingMiddleware(role="critic").wrap_model_call(
            request, traced_handler
        )

        assert calls == 2  # both real, billed model calls happened

        model_spans = [
            s for s in exporter.get_finished_spans() if s.name == "model.critic"
        ]
        total_input = sum(
            s.attributes["gen_ai.usage.input_tokens"] for s in model_spans
        )
        total_output = sum(
            s.attributes["gen_ai.usage.output_tokens"] for s in model_spans
        )
        # Both calls' tokens must be accounted for, not just the second
        # (retried) call's -- 100+200 input, 50+80 output.
        assert total_input == 300
        assert total_output == 130
    finally:
        otel_trace._TRACER_PROVIDER = None
        from opentelemetry.util._once import Once

        otel_trace._TRACER_PROVIDER_SET_ONCE = Once()


def test_tracing_middleware_serializes_tool_args_to_a_json_string_not_a_raw_dict() -> (
    None
):
    from opentelemetry import trace as otel_trace

    provider, exporter = _recording_provider()
    otel_trace.set_tracer_provider(provider)
    try:
        request = _tool_call_request(name="web_search")
        request.tool_call["args"] = {"query": "agent frameworks"}
        middleware.TracingMiddleware(role="researcher").wrap_tool_call(
            request, lambda r: ToolMessage("result", tool_call_id="c1")
        )
        spans = exporter.get_finished_spans()
        tool_span = next(s for s in spans if s.name == "tool.web_search")
        assert isinstance(tool_span.attributes["tool.args"], str)
        assert "agent frameworks" in tool_span.attributes["tool.args"]
    finally:
        otel_trace._TRACER_PROVIDER = None
        from opentelemetry.util._once import Once

        otel_trace._TRACER_PROVIDER_SET_ONCE = Once()


def test_tracing_middleware_defines_both_hook_variants() -> None:
    for sync_name, async_name in [
        ("wrap_model_call", "awrap_model_call"),
        ("wrap_tool_call", "awrap_tool_call"),
    ]:
        sync_overridden = getattr(
            middleware.TracingMiddleware, sync_name
        ) is not getattr(AgentMiddleware, sync_name)
        async_overridden = getattr(
            middleware.TracingMiddleware, async_name
        ) is not getattr(AgentMiddleware, async_name)
        assert sync_overridden == async_overridden


def test_truncate_for_span_marks_truncation_when_text_exceeds_max_length() -> None:
    assert middleware.truncate_for_span("short", 100) == "short"
    result = middleware.truncate_for_span("x" * 50, 10)
    assert result.startswith("x" * 10)
    assert "truncated" in result
    assert len(result) < 50


# -- Test 14: D3.7's renamed tool set, vacuous until stage 4 by design --


def _tool_call_request(
    *, name: str, call_id: str = "c1", state: dict[str, Any] | None = None
) -> ToolCallRequest:
    return ToolCallRequest(
        tool_call={"name": name, "args": {}, "id": call_id},
        tool=None,
        state=state if state is not None else {"messages": []},
        runtime=cast(Any, None),
    )


def _ai_with_calls(*names: str, ids: list[str] | None = None) -> AIMessage:
    call_ids = ids or [f"c{i}" for i in range(len(names))]
    return AIMessage(
        content="",
        tool_calls=[
            {"name": name, "args": {}, "id": call_id}
            for name, call_id in zip(names, call_ids)
        ],
    )


def _tool_result(call_id: str, content: str, *, is_error: bool = False) -> ToolMessage:
    return ToolMessage(
        content=content,
        tool_call_id=call_id,
        name="critique",
        status="error" if is_error else "success",
    )


# -- Stage-4 spec D4.5: RevisionCapMiddleware --


def test_revision_cap_allows_calls_up_to_max_revisions_plus_one() -> None:
    # max_revisions=1 -> limit is 2 critique calls total; the 2nd (this
    # in-flight call, excluded from the prior count) must still be allowed.
    prior_ai = _ai_with_calls("critique", ids=["c0"])
    request = _tool_call_request(
        name="critique",
        call_id="c1",
        state={"messages": [HumanMessage("q"), prior_ai]},
    )
    guard = middleware.RevisionCapMiddleware(max_revisions=1)
    assert guard.wrap_tool_call(request, lambda r: _tool_result("c1", "ok")) == (
        _tool_result("c1", "ok")
    )


def test_revision_cap_refuses_past_max_revisions_plus_one() -> None:
    prior_ai = _ai_with_calls("critique", "critique", ids=["c0", "c1"])
    request = _tool_call_request(
        name="critique",
        call_id="c2",
        state={"messages": [HumanMessage("q"), prior_ai]},
    )
    guard = middleware.RevisionCapMiddleware(max_revisions=1)
    result = guard.wrap_tool_call(
        request, lambda r: _tool_result("c2", "should not run")
    )
    assert isinstance(result, ToolMessage)
    assert result.status == "error"
    assert "revision cap" in str(result.content)


def test_revision_cap_resets_after_a_new_human_message() -> None:
    # Two prior critique calls from a PREVIOUS question, then a new question
    # starts -- the cap must not carry the old count forward.
    old_ai = _ai_with_calls("critique", "critique", ids=["c0", "c1"])
    request = _tool_call_request(
        name="critique",
        call_id="c2",
        state={"messages": [HumanMessage("q1"), old_ai, HumanMessage("q2")]},
    )
    guard = middleware.RevisionCapMiddleware(max_revisions=1)
    result = guard.wrap_tool_call(request, lambda r: _tool_result("c2", "ok"))
    assert result == _tool_result("c2", "ok")


def test_revision_cap_ignores_non_critique_tools() -> None:
    request = _tool_call_request(name="research", state={"messages": []})
    guard = middleware.RevisionCapMiddleware(max_revisions=1)
    result = guard.wrap_tool_call(request, lambda r: _tool_result("c1", "ok"))
    assert result == _tool_result("c1", "ok")


# -- Stage-4 spec D4.18: SaveReportVerdictGuardMiddleware --


def test_verdict_guard_refuses_without_any_completed_critique() -> None:
    request = _tool_call_request(
        name="save_report",
        state={"messages": [HumanMessage("q")], "verdict": "APPROVE"},
    )
    guard = middleware.SaveReportVerdictGuardMiddleware(max_revisions=2)
    result = guard.wrap_tool_call(
        request, lambda r: ToolMessage("saved", tool_call_id="c1")
    )
    assert isinstance(result, ToolMessage)
    assert result.status == "error"


def test_verdict_guard_refuses_a_stale_approve_after_an_errored_critique() -> None:
    """The exact hole a probe-based verification round found: an emitted-only
    gate would let a critique that raised (no Command state write) leave a
    previous question's checkpointed APPROVE unchallenged."""
    critique_ai = _ai_with_calls("critique", ids=["c0"])
    errored = _tool_result("c0", "ERROR: critique failed (RuntimeError)", is_error=True)
    request = _tool_call_request(
        name="save_report",
        call_id="c1",
        state={
            "messages": [HumanMessage("q2"), critique_ai, errored],
            "verdict": "APPROVE",  # stale, from a previous question
        },
    )
    guard = middleware.SaveReportVerdictGuardMiddleware(max_revisions=2)
    result = guard.wrap_tool_call(
        request, lambda r: ToolMessage("saved", tool_call_id="c1")
    )
    assert isinstance(result, ToolMessage)
    assert result.status == "error"


def test_verdict_guard_allows_save_on_approve() -> None:
    critique_ai = _ai_with_calls("critique", ids=["c0"])
    approved = _tool_result("c0", "verdict APPROVE")
    request = _tool_call_request(
        name="save_report",
        call_id="c1",
        state={
            "messages": [HumanMessage("q"), critique_ai, approved],
            "verdict": "APPROVE",
        },
    )
    guard = middleware.SaveReportVerdictGuardMiddleware(max_revisions=2)
    result = guard.wrap_tool_call(
        request, lambda r: ToolMessage("saved", tool_call_id="c1")
    )
    assert result == ToolMessage("saved", tool_call_id="c1")


def test_verdict_guard_refuses_revise_while_rounds_remain() -> None:
    critique_ai = _ai_with_calls("critique", ids=["c0"])
    revised = _tool_result("c0", "verdict REVISE")
    request = _tool_call_request(
        name="save_report",
        call_id="c1",
        state={
            "messages": [HumanMessage("q"), critique_ai, revised],
            "verdict": "REVISE",
        },
    )
    guard = middleware.SaveReportVerdictGuardMiddleware(max_revisions=2)
    result = guard.wrap_tool_call(
        request, lambda r: ToolMessage("saved", tool_call_id="c1")
    )
    assert isinstance(result, ToolMessage)
    assert result.status == "error"
    assert "rounds remain" in str(result.content)


def test_verdict_guard_allows_save_once_the_cap_is_exhausted_on_revise() -> None:
    """A cap-exhausted REVISE run must still be able to save (D4.18) --
    otherwise the run deadlocks against `_S1` rule 5's "stopped revising for
    any reason, save now"."""
    critique_ai = _ai_with_calls(
        "critique", "critique", "critique", ids=["c0", "c1", "c2"]
    )
    results = [
        _tool_result("c0", "verdict REVISE"),
        _tool_result("c1", "verdict REVISE"),
        _tool_result("c2", "verdict REVISE"),
    ]
    request = _tool_call_request(
        name="save_report",
        call_id="c3",
        state={
            "messages": [HumanMessage("q"), critique_ai, *results],
            "verdict": "REVISE",
        },
    )
    guard = middleware.SaveReportVerdictGuardMiddleware(max_revisions=2)
    result = guard.wrap_tool_call(
        request, lambda r: ToolMessage("saved", tool_call_id="c3")
    )
    assert result == ToolMessage("saved", tool_call_id="c3")


def test_verdict_guard_ignores_other_tools() -> None:
    request = _tool_call_request(name="research", state={"messages": []})
    guard = middleware.SaveReportVerdictGuardMiddleware(max_revisions=2)
    result = guard.wrap_tool_call(
        request, lambda r: ToolMessage("ok", tool_call_id="c1")
    )
    assert result == ToolMessage("ok", tool_call_id="c1")


# -- Stage-4 spec D4.15: agent_middleware(tool_exit_behavior=...) --


def test_agent_middleware_default_exit_behavior_is_continue() -> None:
    stack = middleware.agent_middleware(tool_call_limit=5, role="test")
    limiter = next(m for m in stack if isinstance(m, ToolCallLimitMiddleware))
    assert limiter.exit_behavior == "continue"


def test_agent_middleware_accepts_end_exit_behavior_for_the_supervisor() -> None:
    stack = middleware.agent_middleware(
        tool_call_limit=5, tool_exit_behavior="end", role="supervisor"
    )
    limiter = next(m for m in stack if isinstance(m, ToolCallLimitMiddleware))
    assert limiter.exit_behavior == "end"


def test_stability_tools_match_the_supervisor_tool_names() -> None:
    """Pinned against CLAUDE.md's `supervisor.py` row / hl8's `_S1`
    (`plan`, `research`, `critique`, `save_report`), not against
    `supervisor.py` itself -- it does not exist until stage 4. The rule is
    written before the code it constrains, the same shape as
    `test_forbidden_pairs_never_import_each_other`."""
    expected = frozenset({"research", "critique"})
    assert middleware.SUPERVISOR_DELEGATION_TOOLS == expected


# -- Stage-9d D9d.1: SaveReportGuardMiddleware's REVISE-path nudge --
#
# The stage-9c live run produced two cases whose `save_report` was refused by
# `SaveReportVerdictGuardMiddleware` on a standing REVISE while rounds
# remained, after which the Supervisor ended its turn conversationally
# instead of re-entering the loop -- confirmed from those runs' own span
# dumps, not inferred. The refusal string was the only thing pointing the
# model back at `critique`; these tests pin a deterministic nudge instead.


def _guard_state(
    *,
    verdict: str,
    critique_ids: list[str],
    research: bool = True,
    saved_ok: bool = False,
) -> dict[str, Any]:
    messages: list[Any] = [HumanMessage("q")]
    if research:
        messages.append(_ai_with_calls("research", ids=["r0"]))
        messages.append(
            ToolMessage(content="findings", tool_call_id="r0", name="research")
        )
    if critique_ids:
        messages.append(
            _ai_with_calls(*["critique"] * len(critique_ids), ids=critique_ids)
        )
        messages.extend(
            _tool_result(call_id, f"verdict {verdict}") for call_id in critique_ids
        )
    if saved_ok:
        messages.append(_ai_with_calls("save_report", ids=["s0"]))
        messages.append(
            ToolMessage(content="saved to x.md", tool_call_id="s0", name="save_report")
        )
    return {"messages": messages, "verdict": verdict}


def _nudge_probe(state: dict[str, Any], max_revisions: int = 2) -> list[Any]:
    """Run the guard over a no-tool-call response, returning every request
    the handler saw -- length 2 means the nudge fired."""
    seen: list[Any] = []

    def handler(request: ModelRequest[Any]) -> ModelResponse[Any]:
        seen.append(request)
        return ModelResponse(result=[AIMessage(content="I'll stop here.")])

    guard = middleware.SaveReportGuardMiddleware(max_revisions=max_revisions)
    guard.wrap_model_call(_model_request(state=state), handler)
    return seen


def test_save_report_guard_nudges_back_to_critique_on_revise_with_rounds_left() -> None:
    seen = _nudge_probe(_guard_state(verdict="REVISE", critique_ids=["c0"]))
    assert len(seen) == 2
    appended = str(seen[1].messages[-1].content)
    assert "critique" in appended


def test_save_report_guard_nudges_to_save_on_revise_once_the_cap_is_exhausted() -> None:
    seen = _nudge_probe(_guard_state(verdict="REVISE", critique_ids=["c0", "c1", "c2"]))
    assert len(seen) == 2
    appended = str(seen[1].messages[-1].content)
    assert "save_report" in appended


def test_save_report_guard_does_not_nudge_when_a_save_already_executed() -> None:
    seen = _nudge_probe(
        _guard_state(verdict="REVISE", critique_ids=["c0"], saved_ok=True)
    )
    assert len(seen) == 1


def test_save_report_guard_never_nudges_a_run_that_never_reached_research() -> None:
    """`_S1` rule 1a: an out-of-scope plan ends the run with a message and
    no research. Nudging there would force a save on a refusal."""
    seen = _nudge_probe(_guard_state(verdict="REVISE", critique_ids=[], research=False))
    assert len(seen) == 1


def test_save_report_guard_still_nudges_on_approve_unsaved() -> None:
    seen = _nudge_probe(_guard_state(verdict="APPROVE", critique_ids=["c0"]))
    assert len(seen) == 2
    assert "save_report" in str(seen[1].messages[-1].content)


# -- Stage-9e D9e.14: middleware.py:466-467 removed --
#
# `_run_tool_call_ids(messages, "save_report")` used to make the APPROVE
# branch stand down the moment a save_report call was *emitted*, whether or
# not it ever executed. That silenced the guard on exactly the runs
# stage 9c/9d found stuck: a save_report refused by
# SaveReportVerdictGuardMiddleware on a standing REVISE, or one still
# pending (e.g. paused at the HITL gate) with no ToolMessage at all yet.


def test_save_report_guard_nudges_on_approve_when_the_save_was_refused() -> None:
    state = _guard_state(verdict="APPROVE", critique_ids=["c0"])
    state["messages"].append(_ai_with_calls("save_report", ids=["s0"]))
    state["messages"].append(
        ToolMessage(
            content="ERROR: save_report call refused -- the verdict is not " "APPROVE.",
            tool_call_id="s0",
            name="save_report",
            status="error",
        )
    )
    seen = _nudge_probe(state)
    assert len(seen) == 2
    assert "save_report" in str(seen[1].messages[-1].content)


def test_save_report_guard_nudges_on_approve_when_the_save_is_unresolved() -> None:
    """The second case the removed check also covered: a save_report call
    emitted this run with no `ToolMessage` at all yet (e.g. the HITL gate
    has not returned)."""
    state = _guard_state(verdict="APPROVE", critique_ids=["c0"])
    state["messages"].append(_ai_with_calls("save_report", ids=["s0"]))
    seen = _nudge_probe(state)
    assert len(seen) == 2
    assert "save_report" in str(seen[1].messages[-1].content)


@pytest.mark.asyncio
async def test_save_report_guard_revise_nudge_has_an_async_variant() -> None:
    seen: list[Any] = []

    async def handler(request: ModelRequest[Any]) -> ModelResponse[Any]:
        seen.append(request)
        return ModelResponse(result=[AIMessage(content="I'll stop here.")])

    guard = middleware.SaveReportGuardMiddleware(max_revisions=2)
    await guard.awrap_model_call(
        _model_request(state=_guard_state(verdict="REVISE", critique_ids=["c0"])),
        handler,
    )
    assert len(seen) == 2
    assert "critique" in str(seen[1].messages[-1].content)
