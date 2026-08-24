"""`tools_called_for_agent` -- span-derived tool-call extraction (stage 8,
D8.1).

The plan's own stage-8 criterion is that `tools_called` comes from the
offline span dump, never from a mock object, so these tests pin the
extraction against hand-built span trees rather than against whichever real
run happened to exist when the code was written.

Two scope modes, deliberately different:

- a named `agent.<role>` scopes by ancestor walk, the same rule
  `retrieval_context_for_agent` already applies (a span whose chain reaches
  a parent absent from the dump is excluded, never assumed in scope);
- `None` is whole-run scope with **no** ancestor walk at all, which is what
  the Supervisor's own tool calls need -- no `agent.supervisor` span exists
  to scope them by. There is therefore no orphan case in that mode: nothing
  is walked, so nothing can be excluded for an incomplete chain.

Unlike the retrieval-context extraction, repeated calls are **not**
collapsed: a tool called twice is a fact the dump records, and the
extraction reports it faithfully.
"""

from __future__ import annotations

import json

from evals.runner import RunSpans, tools_called_for_agent
from middleware import truncate_for_span


def _span(
    span_id: str,
    parent_span_id: str | None,
    name: str,
    *,
    args: dict[str, object] | None = None,
    raw_args: str | None = None,
) -> dict[str, object]:
    attributes: dict[str, object] = {}
    if args is not None:
        attributes["tool.args"] = json.dumps(args)
    if raw_args is not None:
        attributes["tool.args"] = raw_args
    return {
        "trace_id": "t1",
        "span_id": span_id,
        "parent_span_id": parent_span_id,
        "name": name,
        "start_time": 0,
        "end_time": 1,
        "status": "OK",
        "attributes": attributes,
    }


def _names(run: RunSpans, scope: str | None) -> list[str]:
    return [call.name for call in tools_called_for_agent(run, scope)]


def test_named_agent_scope_selects_only_its_own_tool_calls() -> None:
    spans = [
        _span("root", None, "repl.question"),
        _span("agent-planner", "root", "agent.planner"),
        _span("ws-planner", "agent-planner", "tool.web_search"),
        _span("agent-researcher", "root", "agent.researcher"),
        _span("ks-researcher", "agent-researcher", "tool.knowledge_search"),
        _span("ru-researcher", "agent-researcher", "tool.read_url"),
    ]
    run = RunSpans(run_id="r1", spans=spans)

    assert _names(run, "agent.planner") == ["web_search"]
    assert _names(run, "agent.researcher") == ["knowledge_search", "read_url"]


def test_whole_run_scope_includes_every_tool_span_at_any_depth() -> None:
    # The shape a real Supervisor turn produces (measured, stage-8 spec
    # N11): the four Supervisor tools at the top level, each sub-agent's own
    # calls nested under its agent span. Whole-run scope is every one of
    # them -- "the Supervisor's tools" and "every tool call in the run" are
    # not the same set, and this mode is explicitly the second.
    spans = [
        _span("root", None, "repl.question"),
        _span("plan-tool", "root", "tool.plan"),
        _span("agent-planner", "plan-tool", "agent.planner"),
        _span("ks-planner", "agent-planner", "tool.knowledge_search"),
        _span("critique-tool", "root", "tool.critique"),
        _span("save-tool", "root", "tool.save_report"),
    ]
    run = RunSpans(run_id="r2", spans=spans)

    assert _names(run, None) == [
        "plan",
        "knowledge_search",
        "critique",
        "save_report",
    ]


def test_call_order_is_preserved_and_repeats_are_not_collapsed() -> None:
    spans = [
        _span("root", None, "repl.question"),
        _span("agent-researcher", "root", "agent.researcher"),
        _span("ws-1", "agent-researcher", "tool.web_search"),
        _span("ru-1", "agent-researcher", "tool.read_url"),
        _span("ws-2", "agent-researcher", "tool.web_search"),
    ]
    run = RunSpans(run_id="r3", spans=spans)

    assert _names(run, "agent.researcher") == ["web_search", "read_url", "web_search"]


def test_input_parameters_recovered_from_the_tool_args_json_string() -> None:
    # TracingMiddleware always writes tool.args as a JSON string, never the
    # raw dict -- a raw dict is silently dropped by OTel's own attribute
    # validation (middleware.py's own measured note).
    spans = [
        _span("root", None, "repl.question"),
        _span("agent-planner", "root", "agent.planner"),
        _span(
            "ws",
            "agent-planner",
            "tool.web_search",
            args={"query": "multi-agent architectures"},
        ),
    ]
    run = RunSpans(run_id="r4", spans=spans)

    calls = tools_called_for_agent(run, "agent.planner")
    assert calls[0].input_parameters == {"query": "multi-agent architectures"}


def test_missing_tool_args_yields_a_call_without_input_parameters() -> None:
    # A tool may legitimately take no arguments; an absent attribute is not
    # an error.
    spans = [
        _span("root", None, "repl.question"),
        _span("agent-critic", "root", "agent.critic"),
        _span("ws", "agent-critic", "tool.web_search"),
    ]
    run = RunSpans(run_id="r5", spans=spans)

    calls = tools_called_for_agent(run, "agent.critic")
    assert calls[0].name == "web_search"
    assert calls[0].input_parameters is None


def test_truncated_tool_args_yields_a_call_without_arguments_not_an_error() -> None:
    # Non-regression: the first stage-8 live run raised here on a perfectly
    # healthy Supervisor turn. `TracingMiddleware` caps `tool.args`, and a
    # delegation argument routinely exceeds the cap, so valid JSON reaches
    # the dump with its tail removed. The arguments are unrecoverable, the
    # call itself is not -- and it is the call that tool correctness scores.
    truncated = truncate_for_span(json.dumps({"task": "y" * 500}), 100)
    spans = [
        _span("root", None, "repl.question"),
        _span("critique-tool", "root", "tool.critique", raw_args=truncated),
    ]
    run = RunSpans(run_id="r-truncated", spans=spans)

    calls = tools_called_for_agent(run, None)
    assert [call.name for call in calls] == ["critique"]
    assert calls[0].input_parameters is None


def test_unparsable_tool_args_raises_rather_than_being_swallowed() -> None:
    spans = [
        _span("root", None, "repl.question"),
        _span("agent-planner", "root", "agent.planner"),
        _span("ws", "agent-planner", "tool.web_search", raw_args="{not json"),
    ]
    run = RunSpans(run_id="r6", spans=spans)

    try:
        tools_called_for_agent(run, "agent.planner")
    except ValueError as error:
        assert "tool.args" in str(error)
    else:  # pragma: no cover - the assertion below is the real failure path
        raise AssertionError("unparsable tool.args must raise, not be swallowed")


def test_span_with_missing_parent_is_excluded_from_a_named_agent_scope() -> None:
    spans = [
        _span("agent-researcher", "root-not-in-dump", "agent.researcher"),
        _span("ws-orphan", "some-missing-node", "tool.web_search"),
    ]
    run = RunSpans(run_id="r7", spans=spans)

    assert _names(run, "agent.researcher") == []
    # ...but whole-run scope walks nothing, so the same span counts there.
    assert _names(run, None) == ["web_search"]


def test_non_tool_spans_are_ignored() -> None:
    spans = [
        _span("root", None, "repl.question"),
        _span("model-planner", "root", "model.planner"),
        _span("agent-planner", "root", "agent.planner"),
    ]
    run = RunSpans(run_id="r8", spans=spans)

    assert _names(run, None) == []
