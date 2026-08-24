"""The REPL's three-choice UX over the installed `HumanInTheLoopMiddleware`
decision contract, shared by both coordination paths (`docs/specs/stage-4.md`).

**D4.1 -- `edit` is not a real edit.** The assignment implies `edit` carries
an edited action; the installed `EditDecision` requires
`edited_action={"name": ..., "args": ...}`, which a human's free-text
feedback string is not, and sending it as-is raises `KeyError: 'name'`
inside `HumanInTheLoopMiddleware._process_decision` -- measured against the
installed LangChain 1.3.16, carrying forward `MA_systems_hl10`'s own D6
rather than re-deriving it. `edit` is mapped instead onto a real `reject`
decision carrying the feedback plus a revision instruction: the tool call
does not execute, the model receives an error `ToolMessage` with that
content and revises. `respond` is excluded on its own merit, not just by
omission from this REPL's menu: `save_report`'s `InterruptOnConfig`
(`supervisor.py`) sets `allowed_decisions=["approve", "reject"]` explicitly,
so a `RespondDecision` sent through any channel is refused by the SDK
itself (`_process_decision` raises `ValueError` for a type outside the
list) -- it would otherwise inject a synthetic *successful* `ToolMessage`
without writing anything, telling the Supervisor a save happened when it
did not.

**D4.16 -- unlike hl10, `build_interrupt_request` returns.** hl10 dropped it
because it had exactly one coordination path
(`HumanInTheLoopMiddleware` on the Supervisor); this project restores the
two-path condition that made it necessary in an earlier iteration.
`orchestrator.py` calls `interrupt()` directly rather than through the
middleware, and its payload must be indistinguishable from the one
`HumanInTheLoopMiddleware._create_action_and_config` builds, or the two
paths' shared `resolve_interrupt`/REPL loop would silently see different
shapes.

**Layering:** application -- imported by `main.py` and by
`supervisor.py`/`orchestrator.py`, importing neither (`tests/test_layering.py`).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Literal, cast

from langchain.agents.middleware.human_in_the_loop import (
    ActionRequest,
    Decision,
    DecisionType,
    HITLRequest,
    HITLResponse,
    RejectDecision,
    ReviewConfig,
)
from langchain_core.messages import BaseMessage, ToolMessage

ReplChoice = Literal["approve", "edit", "reject"]

# D4.1: the SDK-level guarantee that `respond` (and, since this REPL never
# emits one, `edit` as a real `EditDecision`) is refused outright.
ALLOWED_DECISIONS: list[DecisionType] = ["approve", "reject"]

# `HumanInTheLoopMiddleware.__init__`'s own default -- matched here so a
# hand-built request (`build_interrupt_request`) renders identically to one
# the middleware produces for the same tool call.
_DEFAULT_DESCRIPTION_PREFIX = "Tool execution requires approval"

_ARG_PREVIEW_LENGTH = 300
_EDIT_INSTRUCTION = "Revise the report and call save_report again."


def build_decision(choice: ReplChoice, text: str = "") -> Decision:
    """Turn one REPL choice into a real `HumanInTheLoopMiddleware` decision.

    Parameters
    ----------
    choice : {"approve", "edit", "reject"}
    text : str, default ""
        The feedback (`edit`) or reason (`reject`) the human typed, if any.

    Returns
    -------
    Decision

    Raises
    ------
    ValueError
        `choice` is none of the three real choices.

    Notes
    -----
    Singular, not a list: an interrupt can gate more than one tool call in
    one turn (`HITLRequest["action_requests"]`), so the caller prompts once
    per action request and assembles the list itself (`resolve_interrupt`).
    """
    if choice == "approve":
        return {"type": "approve"}
    if choice == "edit":
        feedback = text.strip()
        message = f"{feedback} {_EDIT_INSTRUCTION}" if feedback else _EDIT_INSTRUCTION
        return {"type": "reject", "message": message}
    if choice == "reject":
        reason = text.strip()
        decision: RejectDecision = {"type": "reject"}
        if reason:
            decision["message"] = reason
        return decision
    raise ValueError(f"Unknown HITL choice: {choice!r} -- expected approve/edit/reject")


def build_interrupt_request(
    action_requests: Sequence[ActionRequest],
    *,
    description_prefix: str = _DEFAULT_DESCRIPTION_PREFIX,
) -> HITLRequest:
    """Build a `HITLRequest` for `orchestrator.py`'s manual `interrupt()`
    gate, matching `HumanInTheLoopMiddleware`'s own construction exactly.

    Parameters
    ----------
    action_requests : Sequence of ActionRequest
        Each already carries `name`/`args`; a `description` is filled in
        here if the caller left it out, using the same
        `f"{prefix}\\n\\nTool: {name}\\nArgs: {args}"` format the
        installed middleware's `_create_action_and_config` uses.
    description_prefix : str, default the middleware's own default
        Passed only when a caller intentionally diverges; both coordination
        paths use the default.

    Returns
    -------
    HITLRequest
        `review_configs` all carry `ALLOWED_DECISIONS` -- the same
        `["approve", "reject"]` list `save_report`'s `InterruptOnConfig`
        uses on the supervisor path (D4.1), so the two paths gate the same
        set of decisions even though one raises through middleware and the
        other through a raw `interrupt()` call.
    """
    requests: list[ActionRequest] = []
    configs: list[ReviewConfig] = []
    for action in action_requests:
        name = action["name"]
        args = action["args"]
        description = action.get("description") or (
            f"{description_prefix}\n\nTool: {name}\nArgs: {args}"
        )
        requests.append(ActionRequest(name=name, args=args, description=description))
        configs.append(
            ReviewConfig(action_name=name, allowed_decisions=list(ALLOWED_DECISIONS))
        )
    return HITLRequest(action_requests=requests, review_configs=configs)


def render_interrupt(request: HITLRequest) -> str:
    """Render every gated action request's real arguments for the REPL.

    Shows `action_requests[i]["args"]` verbatim, truncated for readability
    only -- never summarised, never taken from the model's own narration of
    the call.
    """
    return "\n\n".join(_render_action(action) for action in request["action_requests"])


def _render_action(action: ActionRequest) -> str:
    lines = [f"Tool:  {action['name']}"]
    for key, value in action["args"].items():
        lines.append(f"  {key}: {_preview(value)}")
    return "\n".join(lines)


def _preview(value: object) -> str:
    text = str(value)
    if len(text) <= _ARG_PREVIEW_LENGTH:
        return text
    return text[:_ARG_PREVIEW_LENGTH] + "..."


def resolve_interrupt(
    request: HITLRequest,
    *,
    read: Callable[[str], str],
    write: Callable[[str], None],
) -> HITLResponse:
    """Prompt for one decision per gated action request and assemble the
    resume payload.

    Parameters
    ----------
    request : HITLRequest
    read : Callable[[str], str]
        Takes a prompt string, returns the human's raw input -- `input` in
        the real REPL, a scripted stand-in in a test.
    write : Callable[[str], None]
        Takes a line to display -- `print` in the real REPL.

    Returns
    -------
    HITLResponse
        `{"decisions": [...]}`, one entry per `request["action_requests"]`,
        in order -- the exact shape `Command(resume=...)` must carry
        (`human_in_the_loop.py`'s own length check on resume).
    """
    write("=" * 60)
    write("ACTION REQUIRES APPROVAL")
    write(render_interrupt(request))
    decisions = [_prompt_one_decision(read) for _ in request["action_requests"]]
    return HITLResponse(decisions=decisions)


def _prompt_one_decision(read: Callable[[str], str]) -> Decision:
    while True:
        raw = read("approve / edit / reject: ").strip().lower()
        if raw in ("approve", "edit", "reject"):
            choice = cast(ReplChoice, raw)
            break
    text = ""
    if choice == "edit":
        text = read("Your feedback: ")
    elif choice == "reject":
        text = read("Reason: ")
    return build_decision(choice, text)


def render_save_status(outcomes: Sequence[BaseMessage]) -> str:
    """State what actually reached disk this turn, from the tool result --
    never from an `AIMessage`'s prose.

    Parameters
    ----------
    outcomes : Sequence of BaseMessage
        Every message this turn produced.

    Returns
    -------
    str

    Notes
    -----
    hl10 measured a live session where the model's closing text claimed a
    save had happened after two human rejections, with nothing on disk
    (`MA_systems_hl10/hitl.py`, D8). The truth source is the `save_report`
    `ToolMessage` itself: `status="error"` covers both a policy-class
    refusal (D4.18's guard, `RoundStabilityMiddleware`,
    `HumanInTheLoopMiddleware`'s own rejection `ToolMessage`) and a genuine
    tool-level failure whose content also carries the `"ERROR:"` prefix
    every tool in this project uses.
    """
    for message in reversed(outcomes):
        if isinstance(message, ToolMessage) and message.name == "save_report":
            if message.status == "error":
                return "[system] No report was saved this turn."
            content = message.text
            if content.startswith("ERROR:"):
                return "[system] No report was saved this turn."
            # tools.save_report already returns "Report saved to: <path>" --
            # this is the message verbatim, not a second wrapping of it.
            return content
    return "[system] No report was saved this turn."


def action_requests_from_tool_calls(
    tool_calls: Sequence[dict[str, Any]],
) -> list[ActionRequest]:
    """Build the `ActionRequest`s `orchestrator.py`'s manual gate needs from
    the composer's emitted `save_report` tool call(s), before handing them
    to `build_interrupt_request`."""
    return [ActionRequest(name=call["name"], args=call["args"]) for call in tool_calls]
