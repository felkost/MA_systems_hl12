"""Runs one real full-pipeline Supervisor turn, live, for the R3c
tool-correctness case (stage 8, `docs/specs/stage-8.md` D8.9).

Not a test module (no `test_*` name, so pytest never collects it) -- the
Supervisor-level counterpart to `tests/live_agents.py`, which runs a single
sub-agent.

Two things this helper deliberately does **not** do differently from
production:

- the HITL gate is answered through `main.auto_approve_decisions`, the same
  headless decision contract the real `--headless` flag uses. The gate is
  never disabled; a run that reached disk without passing it would be a
  defect whatever mode asked for it.
- the whole turn goes through `main.run_session`, not a re-implementation of
  its loop, so what this measures is the coordination path users actually
  run.

The caller is responsible for two things the helper cannot do for itself:
configuring observability once per process (the session-scoped
`live_settings` fixture does it -- a second `configure_observability` call
raises), and redirecting `save_report`'s output directory, which requires
monkeypatching `tools.load_settings` from a test body since `save_report`
resolves its own settings at call time rather than reading the `Settings`
this helper is given.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.messages import BaseMessage
from langgraph.checkpoint.memory import MemorySaver

import main
from config import Settings
from evals.runner import RunSpans, load_run


@dataclass(frozen=True)
class SupervisorLiveRun:
    """One live Supervisor turn: its dump, and the messages it produced.

    `messages` exists because the verdict a run reached is **not** in the
    span dump: the `critique` tool returns a `Command`, and the tracing
    middleware records a `tool.result` attribute only for a `ToolMessage`,
    so every `tool.critique` span in this project carries no result at all.
    The verdict is read from the run's own message history instead.
    """

    run_id: str
    output: str
    spans: RunSpans
    messages: list[BaseMessage]


def _run_dirs(runs_dir: Path) -> set[str]:
    if not runs_dir.is_dir():
        return set()
    return {entry.name for entry in runs_dir.iterdir() if entry.is_dir()}


def run_supervisor_live(agent_input: str, *, settings: Settings) -> SupervisorLiveRun:
    """Drive one full Supervisor turn and return everything it produced.

    Parameters
    ----------
    agent_input : str
        The user's question, sent verbatim as the turn's one `HumanMessage`
        -- which is what the REPL does, so no template is applied here.
    settings : Settings
        Must already carry a `span_dump_dir` pointed off the project's real
        `runs/` (`tests.live_agents.eval_settings`), with observability
        configured for this process.

    Returns
    -------
    SupervisorLiveRun

    Raises
    ------
    RuntimeError
        The turn produced no span dump, or more than one -- either means
        the run_id this helper recovers would be a guess. `run_session`
        mints its own `run_id` per turn and does not return it, so the
        dump directory that appeared during the call is what identifies
        the run; one question must produce exactly one.
    """
    runs_dir = Path(settings.span_dump_dir or "runs")
    before = _run_dirs(runs_dir)

    checkpointer = MemorySaver()
    questions = iter([agent_input, None])
    thread_id = main.run_session(
        settings,
        orchestration="supervisor",
        role_models=main.build_role_models(settings),
        checkpointer=checkpointer,
        read_question=lambda: next(questions),
        write=lambda _: None,
        resolve_decisions=main.auto_approve_decisions,
    )

    new_dirs = _run_dirs(runs_dir) - before
    if len(new_dirs) != 1:
        raise RuntimeError(
            f"expected exactly one new run dump under {runs_dir}, found "
            f"{sorted(new_dirs)} -- cannot identify this turn's run_id"
        )
    run_id = new_dirs.pop()

    messages = _final_messages(checkpointer, thread_id)
    return SupervisorLiveRun(
        run_id=run_id,
        output=_final_text(messages),
        spans=load_run(run_id, runs_dir=runs_dir),
        messages=messages,
    )


def _final_messages(checkpointer: MemorySaver, thread_id: str) -> list[BaseMessage]:
    snapshot = checkpointer.get_tuple({"configurable": {"thread_id": thread_id}})
    if snapshot is None:
        raise RuntimeError(f"no checkpoint written for thread_id {thread_id!r}")
    values: dict[str, Any] = snapshot.checkpoint["channel_values"]
    return list(values.get("messages", []))


def _final_text(messages: list[BaseMessage]) -> str:
    """The Supervisor's own closing text -- the last message carrying any.

    Not the `save_report` result: what a metric judges as `actual_output` is
    what the system said, and `hitl.render_save_status` already exists for
    the separate question of what actually reached disk.
    """
    for message in reversed(messages):
        content = str(message.content).strip()
        if content:
            return content
    return ""
