"""`build_llm_test_case` -- one run's span dump composed into a DeepEval
`LLMTestCase`.

The whole-run (`agent_span_name=None`) case carries its own history: an
earlier draft of this function returned an **empty** `retrieval_context` at
that scope. Whole-run is exactly the scope a later, system-wide citation
metric needs, and an empty retrieval context there would let such a metric
pass while checking nothing. Both fields are therefore real extractions at
both scopes, and the whole-run test below pins that directly.
"""

from __future__ import annotations

import json

from deepeval.test_case import ToolCall

from evals.runner import RunSpans, build_llm_test_case


def _span(
    span_id: str,
    parent_span_id: str | None,
    name: str,
    *,
    chunks: list[str] | None = None,
    args: dict[str, object] | None = None,
) -> dict[str, object]:
    attributes: dict[str, object] = {}
    if chunks is not None:
        attributes["retrieval.chunks"] = chunks
    if args is not None:
        attributes["tool.args"] = json.dumps(args)
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


def _run() -> RunSpans:
    return RunSpans(
        run_id="r1",
        spans=[
            _span("root", None, "repl.question"),
            _span("plan-tool", "root", "tool.plan"),
            _span("agent-planner", "plan-tool", "agent.planner"),
            _span(
                "ks-planner",
                "agent-planner",
                "tool.knowledge_search",
                chunks=["P1"],
                args={"query": "planner query"},
            ),
            _span("research-tool", "root", "tool.research"),
            _span("agent-researcher", "research-tool", "agent.researcher"),
            _span(
                "ks-researcher",
                "agent-researcher",
                "tool.knowledge_search",
                chunks=["R1", "R2"],
            ),
            _span("save-tool", "root", "tool.save_report"),
        ],
    )


def test_agent_scope_composes_that_agent_s_own_context_and_calls() -> None:
    case = build_llm_test_case(
        _run(),
        input="q",
        actual_output="a",
        agent_span_name="agent.researcher",
    )

    assert case.input == "q"
    assert case.actual_output == "a"
    assert case.retrieval_context == ["R1", "R2"]
    assert case.tools_called is not None
    assert [call.name for call in case.tools_called] == ["knowledge_search"]


def test_whole_run_scope_keeps_a_real_retrieval_context_not_an_empty_one() -> None:
    case = build_llm_test_case(_run(), input="q", actual_output="a")

    # Both agents' chunks, at whole-run scope -- not [].
    assert case.retrieval_context == ["P1", "R1", "R2"]
    assert case.tools_called is not None
    assert [call.name for call in case.tools_called] == [
        "plan",
        "knowledge_search",
        "research",
        "knowledge_search",
        "save_report",
    ]


def test_expected_tools_and_expected_output_pass_through_unchanged() -> None:
    # `input_parameters` carries a default, but pydantic's mypy plugin
    # treats an aliased field as required, so it is passed explicitly.
    expected = [
        ToolCall(name="critique", input_parameters=None),
        ToolCall(name="save_report", input_parameters=None),
    ]
    case = build_llm_test_case(
        _run(),
        input="q",
        actual_output="a",
        expected_output="e",
        expected_tools=expected,
    )

    assert case.expected_output == "e"
    assert case.expected_tools == expected


def test_omitted_expectations_stay_none_rather_than_becoming_empty_lists() -> None:
    # An empty list is not a neutral default: a metric reading
    # `expected_tools` would treat [] as a real, satisfiable expectation
    # rather than as "this caller is not measuring tool correctness".
    case = build_llm_test_case(_run(), input="q", actual_output="a")

    assert case.expected_tools is None
    assert case.expected_output is None
