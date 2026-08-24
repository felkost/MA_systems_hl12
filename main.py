"""The REPL: streams responses, prints tool calls, drives the interrupt/
resume loop over either coordination path (`docs/specs/stage-4.md`).

**D4.6 -- one stable `thread_id` per session**, created once at REPL start
and reused across every turn, so a multi-turn conversation shares
checkpointed state and the revision-cap middleware's per-question scoping
(`middleware.py`, `_run_tool_call_ids`) resets correctly on each new
`HumanMessage`.

**D4.6 -- the resume payload is a list, not a bare decision.** An interrupt
can gate more than one tool call in one turn
(`HITLRequest["action_requests"]`), so `Command(resume=...)` always carries
`{"decisions": [...]}`, one entry per action request, via
`hitl.resolve_interrupt`.

**D4.11 -- headless decisions.** `--headless` supplies `approve` for every
gated call through the *same* `HumanInTheLoopMiddleware`/manual-`interrupt`
contract, never by disabling the gate. This exists because stage 8/9's
evaluation runs 15 cases, each stopping at the same interrupt, and
`evals/runner.py` is obs-layer and may not import `supervisor`
(`tests/test_layering.py`) -- nothing else could drive an unattended run.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal, cast
from uuid import uuid4

from langchain.agents.middleware.human_in_the_loop import HITLRequest, HITLResponse
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command
from opentelemetry import baggage, context, trace

import hitl
import models
import observability
import orchestrator
import supervisor
from config import Settings, load_settings

Orchestration = Literal["supervisor", "graph"]

DecisionResolver = Callable[[HITLRequest], HITLResponse]


def build_role_models(settings: Settings) -> dict[str, BaseChatModel]:
    """One chat model per `Settings.ROLES` entry, via `models.build_chat_model`."""
    return {role: models.build_chat_model(settings, role) for role in Settings.ROLES}


def build_graph(
    settings: Settings,
    orchestration: Orchestration,
    *,
    role_models: Mapping[str, BaseChatModel],
    checkpointer: BaseCheckpointSaver[Any] | None,
) -> CompiledStateGraph[Any, Any, Any, Any]:
    """Build the compiled graph `--orchestration` selects.

    Both `create_supervisor` and `create_orchestrator_graph` return a
    `CompiledStateGraph` exposing the same `stream`/`invoke` contract, so
    the REPL loop below is written once against that shared interface --
    only the resume-payload *unwrapping* differs between the two, and that
    lives in `hitl.py`/`orchestrator.py`'s manual gate, not here.
    """
    if orchestration == "supervisor":
        return supervisor.create_supervisor(
            settings, role_models=role_models, checkpointer=checkpointer
        )
    return orchestrator.create_orchestrator_graph(
        settings, role_models=role_models, checkpointer=checkpointer
    )


def auto_approve_decisions(request: HITLRequest) -> HITLResponse:
    """D4.11's headless resolver: `approve` for every gated action request,
    through the real decision contract."""
    return HITLResponse(
        decisions=[{"type": "approve"} for _ in request["action_requests"]]
    )


def interactive_decisions(
    read: Callable[[str], str], write: Callable[[str], None]
) -> DecisionResolver:
    """The real REPL's resolver, built from injected input/output streams
    so it can be swapped for a scripted pair in a test."""

    def resolve(request: HITLRequest) -> HITLResponse:
        return hitl.resolve_interrupt(request, read=read, write=write)

    return resolve


def _print_update(chunk: dict[str, Any], write: Callable[[str], None]) -> None:
    """Print each streamed node's new messages -- tool calls as they are
    emitted, findings and verdicts as they land."""
    for node, update in chunk.items():
        if node == "__interrupt__" or not isinstance(update, dict):
            continue
        for message in update.get("messages", []):
            content = getattr(message, "content", None)
            if not content:
                continue
            if isinstance(message, AIMessage) and message.tool_calls:
                for call in message.tool_calls:
                    write(f"[{node}] calling {call['name']}({call['args']})")
            if isinstance(message, ToolMessage):
                write(f"[{node}:{message.name}] {content}")
            elif isinstance(message, AIMessage) and content:
                write(f"[{node}] {content}")


def _drive_turn(
    graph: CompiledStateGraph[Any, Any, Any, Any],
    payload: Any,
    config: RunnableConfig,
    *,
    write: Callable[[str], None],
    resolve_decisions: DecisionResolver,
) -> None:
    """Stream one turn to completion, resolving every interrupt it raises
    in order -- a turn can raise more than one in sequence (e.g. `reject`
    sends the model back to try again)."""
    while True:
        request: HITLRequest | None = None
        for chunk in graph.stream(payload, config=config, stream_mode="updates"):
            if "__interrupt__" in chunk:
                request = chunk["__interrupt__"][0].value
                break
            _print_update(chunk, write)
        if request is None:
            return
        response = resolve_decisions(request)
        payload = Command(resume=response)


def run_session(
    settings: Settings,
    *,
    orchestration: Orchestration,
    role_models: Mapping[str, BaseChatModel],
    checkpointer: BaseCheckpointSaver[Any] | None,
    read_question: Callable[[], str | None],
    write: Callable[[str], None],
    resolve_decisions: DecisionResolver,
    thread_id: str | None = None,
) -> str:
    """Drive one REPL session: one `thread_id`, any number of turns.

    Parameters
    ----------
    settings : Settings
    orchestration : {"supervisor", "graph"}
    role_models : Mapping of str to BaseChatModel
        Built by `build_role_models`; injectable so a test can pass fakes.
    checkpointer : BaseCheckpointSaver, optional
        Typically a fresh `MemorySaver()` per session.
    read_question : Callable[[], str | None]
        Returns the next question, or `None` to end the session -- `input`
        wrapped to return `None` on EOF in the real REPL, a scripted
        iterator in a test.
    write : Callable[[str], None]
        `print` in the real REPL.
    resolve_decisions : Callable[[HITLRequest], HITLResponse]
        `interactive_decisions(...)` or `auto_approve_decisions` (D4.11).
    thread_id : str, optional
        Defaults to a fresh `uuid4()` -- the one stable id reused across
        every turn in this session (D4.6).

    Returns
    -------
    str
        The `thread_id` used, so a caller (or a test) can inspect the
        resulting checkpoint.
    """
    tid = thread_id or str(uuid4())
    graph = build_graph(
        settings, orchestration, role_models=role_models, checkpointer=checkpointer
    )
    config: RunnableConfig = {
        "configurable": {"thread_id": tid},
        "recursion_limit": settings.recursion_limit,
    }
    while True:
        question = read_question()
        if question is None:
            return tid
        payload: Any = {"messages": [HumanMessage(content=question)]}
        # D5.7: run_id is fresh per turn, not the session's thread_id --
        # "one question = one trace" (docs/specs/stage-5.md). Rides OTel's
        # own Baggage so RunIdStampingProcessor can stamp it onto every
        # span this turn creates, without threading it through
        # supervisor.py/orchestrator.py/middleware.py by hand.
        run_id = str(uuid4())
        token = context.attach(baggage.set_baggage("run_id", run_id))
        try:
            with trace.get_tracer(__name__).start_as_current_span(
                "repl.question", attributes={"run_id": run_id}
            ):
                _drive_turn(
                    graph,
                    payload,
                    config,
                    write=write,
                    resolve_decisions=resolve_decisions,
                )
        finally:
            context.detach(token)


def _stdin_reader(prompt: str = "> ") -> Callable[[], str | None]:
    def read() -> str | None:
        try:
            line = input(prompt)
        except EOFError:
            return None
        stripped = line.strip()
        return None if stripped.lower() in {"exit", "quit"} else line

    return read


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Multi-agent research assistant REPL")
    parser.add_argument(
        "--orchestration",
        choices=("supervisor", "graph"),
        default="supervisor",
        help="Which coordination path to run.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help=(
            "Auto-approve every HITL gate through the real decision "
            "contract (D4.11) -- for unattended evaluation runs, never "
            "the default."
        ),
    )
    return parser


def main(argv: Sequence[str] = ()) -> None:
    args = build_arg_parser().parse_args(argv or None)
    settings = load_settings()
    role_models = build_role_models(settings)
    orchestration = cast(Orchestration, args.orchestration)

    resolve_decisions = (
        auto_approve_decisions
        if args.headless
        else interactive_decisions(read=input, write=print)
    )

    # Built once per process (D5.3/D5.11/D5.13) -- only main.py (interface)
    # imports observability.py; every other layer reaches OpenTelemetry's
    # ambient global tracer directly.
    handle = observability.configure_observability(settings)
    try:
        print(f"Session thread_id: {(tid := str(uuid4()))}")
        run_session(
            settings,
            orchestration=orchestration,
            role_models=role_models,
            checkpointer=MemorySaver(),
            read_question=_stdin_reader(),
            write=print,
            resolve_decisions=resolve_decisions,
            thread_id=tid,
        )
    finally:
        handle.shutdown()


if __name__ == "__main__":
    main(sys.argv[1:])
