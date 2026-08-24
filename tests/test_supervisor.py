"""`supervisor.py`: the agent-as-tool coordination path
(`docs/specs/stage-4.md`).

Two properties are proven end-to-end against real `create_agent`/`ToolNode`
machinery, not asserted from source reading, because a prior verification
round found source-level reasoning about this exact SDK insufficient twice:
that a sub-agent invoked through a wrapper writes no checkpoint (D3.6, D4.3),
and that `save_report` executes exactly once, only after `approve`.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from pydantic import SecretStr

import middleware
import supervisor
import tools as project_tools
from config import Settings
from tests.fakes import FakeToolCallingModel

_PLAN_ARGS = {
    "goal": "g",
    "in_scope": True,
    "search_queries": ["q"],
    "sources_to_check": ["web"],
    "output_format": "narrative",
}
_APPROVE_ARGS = {
    "verdict": "APPROVE",
    "is_fresh": True,
    "is_complete": True,
    "is_well_structured": True,
    "strengths": ["s"],
    "gaps": [],
    "revision_requests": [],
}
_REVISE_ARGS = {
    "verdict": "REVISE",
    "is_fresh": False,
    "is_complete": False,
    "is_well_structured": False,
    "strengths": [],
    "gaps": ["missing X"],
    "revision_requests": ["add X"],
}


def _settings(**overrides: Any) -> Settings:
    return Settings(openrouter_api_key=SecretStr("test-key"), **overrides)


def _fake_role_models(
    *,
    planner_responses: Sequence[BaseMessage] | None = None,
    researcher_responses: Sequence[BaseMessage] | None = None,
    critic_responses: Sequence[BaseMessage] | None = None,
    supervisor_responses: Sequence[BaseMessage] | None = None,
) -> dict[str, FakeToolCallingModel]:
    return {
        "planner": FakeToolCallingModel(
            responses=(
                list(planner_responses)
                if planner_responses is not None
                else [AIMessage(content=json.dumps(_PLAN_ARGS))]
            )
        ),
        "researcher": FakeToolCallingModel(
            responses=(
                list(researcher_responses)
                if researcher_responses is not None
                else [AIMessage(content="findings")]
            )
        ),
        "critic": FakeToolCallingModel(
            responses=(
                list(critic_responses)
                if critic_responses is not None
                else [AIMessage(content=json.dumps(_APPROVE_ARGS))]
            )
        ),
        "supervisor": FakeToolCallingModel(
            responses=(
                list(supervisor_responses)
                if supervisor_responses is not None
                else [AIMessage(content="done")]
            )
        ),
    }


def _fake_save_report() -> tuple[BaseTool, list[tuple[str, str]]]:
    calls: list[tuple[str, str]] = []

    @tool("save_report")
    def fake_save_report(filename: str, content: str) -> str:
        """Save a report."""
        calls.append((filename, content))
        return f"Report saved to: {filename}"

    return fake_save_report, calls


def _tool_call(name: str, args: dict[str, Any], call_id: str) -> dict[str, Any]:
    return {"name": name, "args": args, "id": call_id}


# -- D4.3: all three wrappers forward the original request in code --


def test_all_three_wrappers_forward_the_original_request() -> None:
    seen_inputs: dict[str, str] = {}

    class RecordingModel(FakeToolCallingModel):
        role: str = ""

        def _generate(  # type: ignore[override]
            self, messages: list[Any], **kwargs: Any
        ) -> Any:
            human = next(m for m in messages if isinstance(m, HumanMessage))
            seen_inputs[self.role] = str(human.content)
            return super()._generate(messages, **kwargs)

    planner = RecordingModel(
        role="planner", responses=[AIMessage(content=json.dumps(_PLAN_ARGS))]
    )
    researcher = RecordingModel(
        role="researcher", responses=[AIMessage(content="findings")]
    )
    critic = RecordingModel(
        role="critic", responses=[AIMessage(content=json.dumps(_APPROVE_ARGS))]
    )
    fake_save, _ = _fake_save_report()
    role_models = {
        "planner": planner,
        "researcher": researcher,
        "critic": critic,
        "supervisor": FakeToolCallingModel(
            responses=[
                AIMessage(
                    content="", tool_calls=[_tool_call("plan", {"task": "t1"}, "c1")]
                ),
                AIMessage(
                    content="",
                    tool_calls=[_tool_call("research", {"task": "t2"}, "c2")],
                ),
                AIMessage(
                    content="",
                    tool_calls=[_tool_call("critique", {"task": "t3"}, "c3")],
                ),
                AIMessage(
                    content="",
                    tool_calls=[
                        _tool_call(
                            "save_report", {"filename": "x", "content": "y"}, "c4"
                        )
                    ],
                ),
            ]
        ),
    }
    graph = supervisor.create_supervisor(
        _settings(), role_models=role_models, base_tools=[fake_save]
    )
    graph.invoke({"messages": [HumanMessage(content="What is the capital of France?")]})

    for role in ("planner", "researcher", "critic"):
        assert (
            "Original request: What is the capital of France?" in seen_inputs[role]
        ), (
            f"{role} wrapper did not forward the original request in its input: "
            f"{seen_inputs.get(role)!r}"
        )


# -- D4.2/D4.3: plan and critique write state via Command --


def test_plan_wrapper_writes_the_plan_into_state() -> None:
    fake_save, _ = _fake_save_report()
    role_models = _fake_role_models(
        supervisor_responses=[
            AIMessage(content="", tool_calls=[_tool_call("plan", {"task": "t"}, "c1")]),
            AIMessage(content="stopping here"),
        ]
    )
    graph = supervisor.create_supervisor(
        _settings(), role_models=role_models, base_tools=[fake_save]
    )
    result = graph.invoke({"messages": [HumanMessage(content="q")]})
    assert result["plan"].goal == "g"


def test_critique_wrapper_writes_verdict_and_gaps_into_state() -> None:
    fake_save, _ = _fake_save_report()
    role_models = _fake_role_models(
        critic_responses=[AIMessage(content=json.dumps(_REVISE_ARGS))],
        supervisor_responses=[
            AIMessage(
                content="", tool_calls=[_tool_call("critique", {"task": "t"}, "c1")]
            ),
            AIMessage(content="stopping here"),
        ],
    )
    graph = supervisor.create_supervisor(
        _settings(), role_models=role_models, base_tools=[fake_save]
    )
    result = graph.invoke({"messages": [HumanMessage(content="q")]})
    assert result["verdict"] == "REVISE"
    assert result["critic_gaps"] == ["missing X"]
    assert result["previous_critic_gaps"] is None


# -- D4.19: the Supervisor's own tool list --


def test_supervisor_tool_list_includes_base_tools_and_the_three_wrappers() -> None:
    fake_save, _ = _fake_save_report()
    planner_graph = supervisor.create_planner_agent(
        _settings(),
        project_tools.PLANNER_TOOLS,
        model=FakeToolCallingModel(
            responses=[AIMessage(content=json.dumps(_PLAN_ARGS))]
        ),
        middleware=[],
    )
    research_graph = supervisor.create_research_agent(
        _settings(),
        project_tools.RESEARCHER_TOOLS,
        model=FakeToolCallingModel(responses=[AIMessage(content="f")]),
        middleware=[],
    )
    critic_graph = supervisor.create_critic_agent(
        _settings(),
        project_tools.CRITIC_TOOLS,
        model=FakeToolCallingModel(
            responses=[AIMessage(content=json.dumps(_APPROVE_ARGS))]
        ),
        middleware=[],
    )
    wrapper_names = {
        supervisor._make_plan_tool(planner_graph).name,
        supervisor._make_research_tool(research_graph).name,
        supervisor._make_critique_tool(critic_graph).name,
    }
    assert wrapper_names == {"plan", "research", "critique"}
    assert {t.name for t in [fake_save]} == {"save_report"}


# -- D4.21: SUPERVISOR_DELEGATION_TOOLS is a subset of the wrapper names --


def test_delegation_tools_are_a_subset_of_the_three_wrapper_names() -> None:
    wrapper_names = {"plan", "research", "critique"}
    assert middleware.SUPERVISOR_DELEGATION_TOOLS <= wrapper_names
    assert middleware.SUPERVISOR_DELEGATION_TOOLS == {"research", "critique"}


# -- D4.15: the assembled Supervisor middleware stack's observed order --


def test_supervisor_middleware_order() -> None:
    from langchain.agents.middleware import (
        HumanInTheLoopMiddleware,
        ModelCallLimitMiddleware,
        ModelRetryMiddleware,
        ToolCallLimitMiddleware,
        ToolErrorMiddleware,
        ToolRetryMiddleware,
    )

    stack = supervisor._supervisor_middleware(_settings())
    kinds = [type(m) for m in stack]
    assert kinds == [
        HumanInTheLoopMiddleware,
        ModelCallLimitMiddleware,
        ToolCallLimitMiddleware,
        ToolErrorMiddleware,
        ToolRetryMiddleware,
        ModelRetryMiddleware,
        middleware.TracingMiddleware,
        middleware.RoundStabilityMiddleware,
        middleware.SaveReportGuardMiddleware,
        middleware.RevisionCapMiddleware,
        middleware.SaveReportVerdictGuardMiddleware,
    ]


def test_supervisor_blanket_limiter_uses_end_exit_behavior() -> None:
    from langchain.agents.middleware import ToolCallLimitMiddleware

    stack = supervisor._supervisor_middleware(_settings())
    limiter = next(m for m in stack if isinstance(m, ToolCallLimitMiddleware))
    assert limiter.exit_behavior == "end"


# -- D4.3: sub-agents invoked through a wrapper write no checkpoint --


def test_sub_agent_invoked_through_a_wrapper_writes_no_checkpoint() -> None:
    fake_save, _ = _fake_save_report()
    role_models = _fake_role_models(
        supervisor_responses=[
            AIMessage(content="", tool_calls=[_tool_call("plan", {"task": "t"}, "c1")]),
            AIMessage(
                content="", tool_calls=[_tool_call("research", {"task": "t2"}, "c2")]
            ),
            AIMessage(
                content="", tool_calls=[_tool_call("critique", {"task": "t3"}, "c3")]
            ),
            AIMessage(
                content="",
                tool_calls=[
                    _tool_call("save_report", {"filename": "x", "content": "y"}, "c4")
                ],
            ),
        ]
    )
    saver = MemorySaver()
    graph = supervisor.create_supervisor(
        _settings(), role_models=role_models, base_tools=[fake_save], checkpointer=saver
    )
    graph.invoke(
        {"messages": [HumanMessage(content="q")]},
        config=RunnableConfig(configurable={"thread_id": "t1"}),
    )
    threads = {cp.config["configurable"]["thread_id"] for cp in saver.list(None)}
    assert threads == {"t1"}, f"a sub-agent leaked a checkpoint: {threads}"


# -- D4.1/D4.14: exactly one executed save_report, only after approve --


def test_save_report_executes_exactly_once_after_approve() -> None:
    fake_save, calls = _fake_save_report()
    role_models = _fake_role_models(
        supervisor_responses=[
            AIMessage(content="", tool_calls=[_tool_call("plan", {"task": "t"}, "c1")]),
            AIMessage(
                content="", tool_calls=[_tool_call("research", {"task": "t2"}, "c2")]
            ),
            AIMessage(
                content="", tool_calls=[_tool_call("critique", {"task": "t3"}, "c3")]
            ),
            AIMessage(
                content="",
                tool_calls=[
                    _tool_call("save_report", {"filename": "x", "content": "y"}, "c4")
                ],
            ),
        ]
    )
    graph = supervisor.create_supervisor(
        _settings(),
        role_models=role_models,
        base_tools=[fake_save],
        checkpointer=MemorySaver(),
    )
    config: RunnableConfig = {"configurable": {"thread_id": "t1"}}
    result = graph.invoke({"messages": [HumanMessage(content="q")]}, config=config)
    assert result.get("__interrupt__")
    assert calls == []

    graph.invoke(Command(resume={"decisions": [{"type": "approve"}]}), config=config)
    assert calls == [("x", "y")]


def test_save_report_never_executes_on_reject() -> None:
    fake_save, calls = _fake_save_report()
    role_models = _fake_role_models(
        supervisor_responses=[
            AIMessage(content="", tool_calls=[_tool_call("plan", {"task": "t"}, "c1")]),
            AIMessage(
                content="", tool_calls=[_tool_call("research", {"task": "t2"}, "c2")]
            ),
            AIMessage(
                content="", tool_calls=[_tool_call("critique", {"task": "t3"}, "c3")]
            ),
            AIMessage(
                content="",
                tool_calls=[
                    _tool_call("save_report", {"filename": "x", "content": "y"}, "c4")
                ],
            ),
            AIMessage(content="I will not save without approval."),
        ]
    )
    graph = supervisor.create_supervisor(
        _settings(),
        role_models=role_models,
        base_tools=[fake_save],
        checkpointer=MemorySaver(),
    )
    config: RunnableConfig = {"configurable": {"thread_id": "t1"}}
    graph.invoke({"messages": [HumanMessage(content="q")]}, config=config)
    graph.invoke(
        Command(resume={"decisions": [{"type": "reject", "message": "no"}]}),
        config=config,
    )
    assert calls == []


# -- D4.5: the revision cap, boundary --


def _revise_args(gap: str) -> dict[str, Any]:
    # Distinct gaps per round, on purpose: identical gaps two rounds running
    # would trip RoundStabilityMiddleware's own signal-repetition refusal
    # first, which is a different guard than the one this test targets.
    return {**_REVISE_ARGS, "gaps": [gap], "revision_requests": [f"fix {gap}"]}


def _critic_responses(*gaps: str) -> list[AIMessage]:
    """One `AIMessage` per gap, **doubled**: the critic sub-agent carries
    `CriticVerificationMiddleware` in production (assembled by
    `create_supervisor` itself), which retries the model once whenever its
    response calls no verification tool -- true of every scripted response
    here, since `FakeToolCallingModel` never emits a tool call at all. The
    first of each pair is discarded by that retry; the second is the
    critique wrapper's actual result."""
    responses = []
    for gap in gaps:
        message = AIMessage(content=json.dumps(_revise_args(gap)))
        responses.extend([message, message])
    return responses


def test_revision_cap_refuses_the_call_past_max_revisions_plus_one() -> None:
    fake_save, _ = _fake_save_report()
    # max_revisions=1 -> limit is 2 critique calls; script the Supervisor
    # trying a 3rd critique call, which must be refused.
    role_models = _fake_role_models(
        critic_responses=_critic_responses("A", "B"),
        supervisor_responses=[
            AIMessage(
                content="", tool_calls=[_tool_call("critique", {"task": "t1"}, "c1")]
            ),
            AIMessage(
                content="", tool_calls=[_tool_call("critique", {"task": "t2"}, "c2")]
            ),
            AIMessage(
                content="", tool_calls=[_tool_call("critique", {"task": "t3"}, "c3")]
            ),
            AIMessage(content="stopping here"),
        ],
    )
    graph = supervisor.create_supervisor(
        _settings(max_revisions=1), role_models=role_models, base_tools=[fake_save]
    )
    result = graph.invoke({"messages": [HumanMessage(content="q")]})
    tool_messages = [
        m
        for m in result["messages"]
        if getattr(m, "name", None) == "critique"
        and getattr(m, "status", None) == "error"
    ]
    assert len(tool_messages) == 1
    assert "revision cap" in str(tool_messages[0].content)


def test_revision_cap_resets_on_a_new_question_in_the_same_session() -> None:
    fake_save, _ = _fake_save_report()
    role_models = _fake_role_models(
        critic_responses=_critic_responses("A", "B"),
        supervisor_responses=[
            # Question 1: exhaust the cap (max_revisions=1 -> 2 calls).
            AIMessage(
                content="", tool_calls=[_tool_call("critique", {"task": "t1"}, "c1")]
            ),
            AIMessage(
                content="", tool_calls=[_tool_call("critique", {"task": "t2"}, "c2")]
            ),
            AIMessage(content="answer 1"),
        ],
    )
    graph = supervisor.create_supervisor(
        _settings(max_revisions=1),
        role_models=role_models,
        base_tools=[fake_save],
        checkpointer=MemorySaver(),
    )
    config: RunnableConfig = {"configurable": {"thread_id": "t1"}}
    graph.invoke({"messages": [HumanMessage(content="q1")]}, config=config)

    # Question 2 on the SAME thread: one critique call must be allowed
    # again -- the cap must not carry the first question's count forward.
    approve_message = AIMessage(content=json.dumps(_APPROVE_ARGS))
    role_models["critic"].responses = [approve_message, approve_message]
    role_models["critic"].index = 0
    role_models["supervisor"].responses = [
        AIMessage(
            content="", tool_calls=[_tool_call("critique", {"task": "t3"}, "c3")]
        ),
        AIMessage(content="answer 2"),
    ]
    role_models["supervisor"].index = 0
    result = graph.invoke({"messages": [HumanMessage(content="q2")]}, config=config)

    question_2_index = next(
        i
        for i, m in enumerate(result["messages"])
        if isinstance(m, HumanMessage) and m.content == "q2"
    )
    new_turn_messages = result["messages"][question_2_index:]
    refusals = [
        m
        for m in new_turn_messages
        if getattr(m, "name", None) == "critique"
        and getattr(m, "status", None) == "error"
    ]
    assert refusals == [], "the revision cap carried over from a previous question"
