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
from langfuse import Langfuse
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command
from opentelemetry import trace

import hitl
import models
import observability
import orchestrator
import paths
import supervisor
from config import Settings, load_settings
from prompt_store import LangfusePromptStore, PromptStore

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
    prompt_store: PromptStore,
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
            settings,
            role_models=role_models,
            prompt_store=prompt_store,
            checkpointer=checkpointer,
        )
    return orchestrator.create_orchestrator_graph(
        settings,
        role_models=role_models,
        prompt_store=prompt_store,
        checkpointer=checkpointer,
    )


def build_prompt_store(
    settings: Settings, *, client: Langfuse | None = None
) -> PromptStore:
    """The real `PromptStore` `main.py` uses: Langfuse Cloud, with the local
    snapshot as `fallback=` (hl12 stage 1).

    Parameters
    ----------
    settings : Settings
    client : Langfuse, optional
        Reused as-is when given -- `main()` passes
        `handle.langfuse_client`, the one `Langfuse` client
        `configure_observability` already built (D2.1, `docs/specs/stage-2.md`,
        section 1). A second `Langfuse(public_key=...)` call with the same
        key returns the SDK's own cached singleton and silently discards
        whatever `tracer_provider=`/`should_export_span=` the first call
        set, so prompt fetching must never construct its own client when
        one already exists. When `None` (no tracing client to reuse, e.g.
        `tracing_enabled=False`), a client is built here exactly as before
        -- prompt fetching must work whether or not tracing is enabled,
        since every agent's system prompt now depends on it (requirement
        3), while tracing is a genuinely optional concern.
    """
    if settings.langfuse_public_key is None or settings.langfuse_secret_key is None:
        raise RuntimeError(
            "LANGFUSE_PUBLIC_KEY/LANGFUSE_SECRET_KEY are not set -- prompt "
            "fetching requires them regardless of TRACING_ENABLED, since "
            "every agent's system prompt is fetched from Langfuse"
        )
    if client is None:
        client = Langfuse(
            public_key=settings.langfuse_public_key.get_secret_value(),
            secret_key=settings.langfuse_secret_key.get_secret_value(),
            host=settings.langfuse_base_url,
            # Only reached when `configure_observability` built no client of
            # its own (`tracing_enabled=False`) -- without this keyword the
            # SDK's own default (`tracing_enabled=True`) would silently
            # start sending spans to Langfuse Cloud even though this
            # project's own `Settings.tracing_enabled` says not to.
            tracing_enabled=settings.tracing_enabled,
        )
    snapshot_path = (
        paths.prompt_snapshot_path() if settings.prompt_snapshot_enabled else None
    )
    return LangfusePromptStore(
        client,
        snapshot_path=snapshot_path,
        cache_ttl_seconds=settings.prompt_cache_ttl_seconds,
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


def _print_update(chunk: dict[str, Any], write: Callable[[str], None]) -> str | None:
    """Print each streamed node's new messages -- tool calls as they are
    emitted, findings and verdicts as they land.

    Returns
    -------
    str or None
        The `content` of this chunk's `AIMessage`, if it carried one (the
        same text just printed) -- `None` otherwise. `_drive_turn` uses this
        to track the turn's final answer without duplicating the message
        filtering above (D2.4, `docs/specs/stage-2.md`, section 3).
    """
    answer: str | None = None
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
                answer = content
    return answer


def _drive_turn(
    graph: CompiledStateGraph[Any, Any, Any, Any],
    payload: Any,
    config: RunnableConfig,
    *,
    write: Callable[[str], None],
    resolve_decisions: DecisionResolver,
) -> str:
    """Stream one turn to completion, resolving every interrupt it raises
    in order -- a turn can raise more than one in sequence (e.g. `reject`
    sends the model back to try again).

    Returns
    -------
    str
        The last `AIMessage` content emitted before the stream ended without
        an interrupt -- the model's final answer to the user's question, not
        a concatenation of every intermediate agent's chatter or tool-call
        announcement, since each node's terminal `AIMessage` overwrites it in
        turn-order. `set_trace_io` has no other source for this text (D2.4).
    """
    answer = ""
    while True:
        request: HITLRequest | None = None
        for chunk in graph.stream(payload, config=config, stream_mode="updates"):
            if "__interrupt__" in chunk:
                request = chunk["__interrupt__"][0].value
                break
            if (latest := _print_update(chunk, write)) is not None:
                answer = latest
        if request is None:
            return answer
        response = resolve_decisions(request)
        payload = Command(resume=response)


def run_session(
    settings: Settings,
    *,
    orchestration: Orchestration,
    role_models: Mapping[str, BaseChatModel],
    prompt_store: PromptStore,
    checkpointer: BaseCheckpointSaver[Any] | None,
    read_question: Callable[[], str | None],
    write: Callable[[str], None],
    resolve_decisions: DecisionResolver,
    thread_id: str | None = None,
    session_id: str | None = None,
    user_id: str | None = None,
) -> str:
    """Drive one REPL session: one `thread_id`, any number of turns.

    Parameters
    ----------
    settings : Settings
    orchestration : {"supervisor", "graph"}
    role_models : Mapping of str to BaseChatModel
        Built by `build_role_models`; injectable so a test can pass fakes.
    prompt_store : PromptStore
        Built by `build_prompt_store`; injectable so a test can pass
        `prompt_store.SnapshotPromptStore` instead (hl12 stage 1).
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
    session_id : str, optional
        Defaults to a fresh `uuid4()` -- bound once here, outside the
        per-turn loop, and reused by every turn's `run_context` call. This
        is what makes 3-5 turns in one session group into one Langfuse
        session (requirement R2), not 3-5 separate one-trace sessions
        (D2.5, `docs/specs/stage-2.md`, section 3).
    user_id : str, optional
        Defaults to `settings.default_user_id`.

    Returns
    -------
    str
        The `thread_id` used, so a caller (or a test) can inspect the
        resulting checkpoint.
    """
    tid = thread_id or str(uuid4())
    sid = session_id or str(uuid4())
    uid = user_id or settings.default_user_id
    graph = build_graph(
        settings,
        orchestration,
        role_models=role_models,
        prompt_store=prompt_store,
        checkpointer=checkpointer,
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
        # "one question = one trace" (docs/specs/stage-5.md).
        run_id = str(uuid4())
        with observability.run_context(session_id=sid, user_id=uid, run_id=run_id):
            with trace.get_tracer(__name__).start_as_current_span(
                "repl.question", attributes={"run_id": run_id}
            ):
                answer = _drive_turn(
                    graph,
                    payload,
                    config,
                    write=write,
                    resolve_decisions=resolve_decisions,
                )
                observability.set_trace_io(input=question, output=answer)


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
    parser.add_argument(
        "--session-id",
        default=None,
        help=(
            "Langfuse session id every turn in this run is grouped under "
            "(requirement R2). Defaults to a fresh uuid4() per process."
        ),
    )
    parser.add_argument(
        "--user-id",
        default=None,
        help=(
            "Langfuse user id every trace in this run carries (requirement "
            "R2). Defaults to Settings.default_user_id."
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

    # Built first (D2.1, docs/specs/stage-2.md, section 1): main.py (interface)
    # is the only module that imports observability.py, and build_prompt_store
    # below reuses the one Langfuse client this call builds instead of
    # constructing a second one -- a second Langfuse(public_key=...) call with
    # the same key returns the SDK's own cached singleton and silently
    # discards configure_observability's tracer_provider=/should_export_span=.
    handle = observability.configure_observability(settings)
    try:
        prompt_store = build_prompt_store(settings, client=handle.langfuse_client)
        print(f"Session thread_id: {(tid := str(uuid4()))}")
        run_session(
            settings,
            orchestration=orchestration,
            role_models=role_models,
            prompt_store=prompt_store,
            checkpointer=MemorySaver(),
            read_question=_stdin_reader(),
            write=print,
            resolve_decisions=resolve_decisions,
            thread_id=tid,
            session_id=args.session_id,
            user_id=args.user_id,
        )
    finally:
        handle.shutdown()


if __name__ == "__main__":
    main(sys.argv[1:])
