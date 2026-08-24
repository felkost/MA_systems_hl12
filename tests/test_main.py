"""`main.py`: the REPL driving loop over both coordination paths
(`docs/specs/stage-4.md`).

Every test injects a fake `save_report` -- `main.run_session` otherwise
defaults to the real tool, which writes to `output/`.
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from pydantic import SecretStr

import main
import observability
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
_DRAFT_ARGS = {"filename": "report", "content": "body"}


def _settings(**overrides: Any) -> Settings:
    return Settings(openrouter_api_key=SecretStr("test-key"), **overrides)


def _fake_save_report(monkeypatch: Any, module: Any) -> list[tuple[str, str]]:
    """Patch `module.tools.SUPERVISOR_TOOLS` so both coordination paths'
    default `base_tools` pick up a fake `save_report` that never touches
    disk."""
    calls: list[tuple[str, str]] = []

    @tool("save_report")
    def fake_save_report(filename: str, content: str) -> str:
        """Save."""
        calls.append((filename, content))
        return f"Report saved to: {filename}"

    monkeypatch.setattr(module.tools, "SUPERVISOR_TOOLS", [fake_save_report])
    return calls


def _supervisor_role_models(
    *, network_probe: list[str]
) -> dict[str, FakeToolCallingModel]:
    class NoNetworkModel(FakeToolCallingModel):
        def _generate(  # type: ignore[override]
            self, messages: list[Any], **kwargs: Any
        ) -> Any:
            network_probe.append("model_call")  # scripted, no real network
            return super()._generate(messages, **kwargs)

    return {
        "planner": NoNetworkModel(
            responses=[AIMessage(content=json.dumps(_PLAN_ARGS))]
        ),
        "researcher": NoNetworkModel(responses=[AIMessage(content="findings")]),
        "critic": NoNetworkModel(
            responses=[
                AIMessage(content=json.dumps(_APPROVE_ARGS)),
                AIMessage(content=json.dumps(_APPROVE_ARGS)),
            ]
        ),
        "supervisor": NoNetworkModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[{"name": "plan", "args": {"task": "t"}, "id": "c1"}],
                ),
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "research", "args": {"task": "t2"}, "id": "c2"}
                    ],
                ),
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "critique", "args": {"task": "t3"}, "id": "c3"}
                    ],
                ),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "save_report",
                            "args": {"filename": "x", "content": "y"},
                            "id": "c4",
                        }
                    ],
                ),
                AIMessage(content="Done."),
            ]
        ),
    }


def _orchestrator_role_models() -> dict[str, FakeToolCallingModel]:
    return {
        "planner": FakeToolCallingModel(
            responses=[AIMessage(content=json.dumps(_PLAN_ARGS))]
        ),
        "researcher": FakeToolCallingModel(responses=[AIMessage(content="findings")]),
        "critic": FakeToolCallingModel(
            responses=[
                AIMessage(content=json.dumps(_APPROVE_ARGS)),
                AIMessage(content=json.dumps(_APPROVE_ARGS)),
            ]
        ),
        "supervisor": FakeToolCallingModel(
            responses=[AIMessage(content=json.dumps(_DRAFT_ARGS))]
        ),
    }


def _scripted_io(*answers: str) -> tuple[Any, list[str]]:
    outputs: list[str] = []
    reads = iter(answers)

    def read(_prompt: str) -> str:
        return next(reads)

    return read, outputs


def test_supervisor_path_reaches_interrupt_with_zero_network_and_resumes(
    monkeypatch: Any,
) -> None:
    import supervisor as supervisor_module

    calls = _fake_save_report(monkeypatch, supervisor_module)
    network_probe: list[str] = []
    role_models = _supervisor_role_models(network_probe=network_probe)

    seen_requests: list[Any] = []

    def resolve(request: Any) -> Any:
        seen_requests.append(request)
        return {"decisions": [{"type": "approve"}]}

    questions = iter(["q1", None])
    outputs: list[str] = []
    tid = main.run_session(
        _settings(max_revisions=1),
        orchestration="supervisor",
        role_models=role_models,
        checkpointer=MemorySaver(),
        read_question=lambda: next(questions),
        write=outputs.append,
        resolve_decisions=resolve,
    )
    assert tid
    assert len(seen_requests) == 1
    assert calls == [("x", "y")]
    assert network_probe  # the fake model was called at all -- sanity


def test_graph_path_reaches_interrupt_with_zero_network_and_resumes(
    monkeypatch: Any,
) -> None:
    import orchestrator as orchestrator_module

    calls = _fake_save_report(monkeypatch, orchestrator_module)
    role_models = _orchestrator_role_models()

    seen_requests: list[Any] = []

    def resolve(request: Any) -> Any:
        seen_requests.append(request)
        return {"decisions": [{"type": "approve"}]}

    questions = iter(["q1", None])
    outputs: list[str] = []
    tid = main.run_session(
        _settings(max_revisions=1),
        orchestration="graph",
        role_models=role_models,
        checkpointer=MemorySaver(),
        read_question=lambda: next(questions),
        write=outputs.append,
        resolve_decisions=resolve,
    )
    assert tid
    assert len(seen_requests) == 1
    assert calls == [("report", "body")]


def test_reject_never_executes_save_report(monkeypatch: Any) -> None:
    import supervisor as supervisor_module

    calls = _fake_save_report(monkeypatch, supervisor_module)
    network_probe: list[str] = []
    role_models = _supervisor_role_models(network_probe=network_probe)
    # After a reject the fake supervisor model keeps replaying its last
    # scripted response (save_report) forever, so cap the turn at one
    # resolved interrupt by having the resolver switch to reject once.
    resolved = {"count": 0}

    def resolve(request: Any) -> Any:
        resolved["count"] += 1
        return {"decisions": [{"type": "reject", "message": "not ready"}]}

    questions = iter(["q1", None])
    outputs: list[str] = []
    main.run_session(
        _settings(max_revisions=1),
        orchestration="supervisor",
        role_models=role_models,
        checkpointer=MemorySaver(),
        read_question=lambda: next(questions),
        write=outputs.append,
        resolve_decisions=resolve,
    )
    assert calls == []
    assert resolved["count"] >= 1


def test_one_thread_id_reused_across_turns(monkeypatch: Any) -> None:
    import orchestrator as orchestrator_module

    _fake_save_report(monkeypatch, orchestrator_module)
    role_models = _orchestrator_role_models()
    role_models["critic"].responses = [
        AIMessage(content=json.dumps(_APPROVE_ARGS)),
        AIMessage(content=json.dumps(_APPROVE_ARGS)),
        AIMessage(content=json.dumps(_APPROVE_ARGS)),
        AIMessage(content=json.dumps(_APPROVE_ARGS)),
    ]

    seen_thread_ids: set[str] = set()

    def resolve(request: Any) -> Any:
        return {"decisions": [{"type": "approve"}]}

    questions = iter(["q1", "q2", None])
    saver = MemorySaver()
    tid = main.run_session(
        _settings(max_revisions=1),
        orchestration="graph",
        role_models=role_models,
        checkpointer=saver,
        read_question=lambda: next(questions),
        write=lambda _: None,
        resolve_decisions=resolve,
    )
    for checkpoint in saver.list(None):
        seen_thread_ids.add(checkpoint.config["configurable"]["thread_id"])
    assert seen_thread_ids == {tid}


def test_headless_auto_approve_answers_through_the_real_contract() -> None:
    import hitl as hitl_module

    request = hitl_module.build_interrupt_request(
        [{"name": "save_report", "args": {"filename": "x", "content": "y"}}]
    )
    response = main.auto_approve_decisions(request)
    assert response == {"decisions": [{"type": "approve"}]}


def test_interactive_decisions_drives_hitl_resolve_interrupt() -> None:
    import hitl as hitl_module

    request = hitl_module.build_interrupt_request(
        [{"name": "save_report", "args": {"filename": "x", "content": "y"}}]
    )
    read, outputs = _scripted_io("approve")
    resolver = main.interactive_decisions(read=read, write=outputs.append)
    response = resolver(request)
    assert response == {"decisions": [{"type": "approve"}]}


def test_build_graph_selects_supervisor_or_orchestrator(monkeypatch: Any) -> None:
    import orchestrator as orchestrator_module
    import supervisor as supervisor_module

    _fake_save_report(monkeypatch, supervisor_module)
    _fake_save_report(monkeypatch, orchestrator_module)
    role_models = _orchestrator_role_models()

    supervisor_graph = main.build_graph(
        _settings(), "supervisor", role_models=role_models, checkpointer=None
    )
    graph_graph = main.build_graph(
        _settings(), "graph", role_models=role_models, checkpointer=None
    )
    assert supervisor_graph is not graph_graph


# -- Stage 5, D5.7: each turn opens its own "repl.question" span with a
# fresh run_id, via OTel Baggage + RunIdStampingProcessor -- not a
# configure_observability() parameter (that draft was unbuildable, see
# docs/specs/stage-5.md's "corrected" note on D5.7).


def test_run_session_opens_one_repl_question_span_per_turn_with_a_fresh_run_id(
    monkeypatch: Any,
) -> None:
    import supervisor as supervisor_module
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    from tests.test_observability import _reset_otel_global_state

    _fake_save_report(monkeypatch, supervisor_module)
    network_probe: list[str] = []
    role_models = _supervisor_role_models(network_probe=network_probe)

    _reset_otel_global_state()
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(observability.RunIdStampingProcessor())
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    from opentelemetry import trace as otel_trace

    otel_trace.set_tracer_provider(provider)
    try:
        questions = iter(["q1", "q2", None])
        main.run_session(
            _settings(max_revisions=1),
            orchestration="supervisor",
            role_models=role_models,
            checkpointer=MemorySaver(),
            read_question=lambda: next(questions),
            write=lambda _: None,
            resolve_decisions=lambda request: {"decisions": [{"type": "approve"}]},
        )
    finally:
        _reset_otel_global_state()

    question_spans = [
        s for s in exporter.get_finished_spans() if s.name == "repl.question"
    ]
    assert len(question_spans) == 2
    run_ids = []
    for span in question_spans:
        assert span.attributes is not None
        run_ids.append(span.attributes["run_id"])
    assert len(set(run_ids)) == 2
