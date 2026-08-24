"""`hitl.py`'s REPL decision UX, shared by both coordination paths.

D4.1: `edit` maps onto a real `reject` decision (D6), never `EditDecision`
(whose `edited_action` a human's free text cannot satisfy) and never
`RespondDecision` (which claims a write happened without performing one).
D4.16: the hand-built `HITLRequest` payload must render identically to what
`HumanInTheLoopMiddleware` itself would build for the same tool call.
"""

from __future__ import annotations

from langchain_core.messages import ToolMessage

import hitl


def test_approve_maps_to_approve_decision() -> None:
    assert hitl.build_decision("approve") == {"type": "approve"}


def test_edit_maps_to_a_reject_decision_carrying_feedback() -> None:
    decision = hitl.build_decision("edit", "make it shorter")
    assert decision["type"] == "reject"
    assert "make it shorter" in decision["message"]
    assert "edited_action" not in decision


def test_edit_with_no_feedback_still_carries_the_revision_instruction() -> None:
    decision = hitl.build_decision("edit", "")
    assert decision["type"] == "reject"
    assert decision["message"]


def test_reject_maps_to_reject_decision_with_reason() -> None:
    decision = hitl.build_decision("reject", "not ready")
    assert decision == {"type": "reject", "message": "not ready"}


def test_reject_with_no_reason_carries_no_message() -> None:
    decision = hitl.build_decision("reject", "")
    assert decision == {"type": "reject"}


def test_build_decision_rejects_an_unknown_choice() -> None:
    import pytest

    with pytest.raises(ValueError, match="Unknown HITL choice"):
        hitl.build_decision("respond")  # type: ignore[arg-type]


def test_no_code_path_ever_constructs_a_respond_or_edit_decision() -> None:
    """Static check on build_decision's own source: neither
    RespondDecision's `type: "respond"` nor EditDecision's `edited_action`
    key is ever written -- the only two ways this module could emit a
    decision `allowed_decisions=["approve", "reject"]` would refuse."""
    import inspect

    source = inspect.getsource(hitl.build_decision)
    assert '"respond"' not in source
    assert "edited_action" not in source


def test_allowed_decisions_excludes_edit_and_respond() -> None:
    assert hitl.ALLOWED_DECISIONS == ["approve", "reject"]


def test_build_interrupt_request_matches_the_middleware_default_description() -> None:
    """The exact format `HumanInTheLoopMiddleware._create_action_and_config`
    builds when no explicit `description` is supplied -- D4.16's
    indistinguishability requirement."""
    request = hitl.build_interrupt_request(
        [{"name": "save_report", "args": {"filename": "x", "content": "y"}}]
    )
    description = request["action_requests"][0]["description"]
    assert description == (
        "Tool execution requires approval\n\n"
        "Tool: save_report\n"
        "Args: {'filename': 'x', 'content': 'y'}"
    )
    assert request["review_configs"][0]["allowed_decisions"] == ["approve", "reject"]
    assert request["review_configs"][0]["action_name"] == "save_report"


def test_build_interrupt_request_keeps_an_explicit_description() -> None:
    request = hitl.build_interrupt_request(
        [{"name": "save_report", "args": {}, "description": "custom"}]
    )
    assert request["action_requests"][0]["description"] == "custom"


def test_render_interrupt_shows_real_arguments() -> None:
    request = hitl.build_interrupt_request(
        [{"name": "save_report", "args": {"filename": "report.md"}}]
    )
    rendered = hitl.render_interrupt(request)
    assert "save_report" in rendered
    assert "report.md" in rendered


def test_resolve_interrupt_approve_needs_no_further_input() -> None:
    request = hitl.build_interrupt_request(
        [{"name": "save_report", "args": {"filename": "x"}}]
    )
    reads = iter(["approve"])
    response = hitl.resolve_interrupt(
        request, read=lambda _: next(reads), write=lambda _: None
    )
    assert response == {"decisions": [{"type": "approve"}]}


def test_resolve_interrupt_reject_prompts_for_a_reason() -> None:
    request = hitl.build_interrupt_request(
        [{"name": "save_report", "args": {"filename": "x"}}]
    )
    reads = iter(["reject", "not good enough"])
    response = hitl.resolve_interrupt(
        request, read=lambda _: next(reads), write=lambda _: None
    )
    assert response == {"decisions": [{"type": "reject", "message": "not good enough"}]}


def test_resolve_interrupt_reprompts_on_an_invalid_choice() -> None:
    request = hitl.build_interrupt_request(
        [{"name": "save_report", "args": {"filename": "x"}}]
    )
    reads = iter(["banana", "approve"])
    response = hitl.resolve_interrupt(
        request, read=lambda _: next(reads), write=lambda _: None
    )
    assert response == {"decisions": [{"type": "approve"}]}


def test_resolve_interrupt_collects_one_decision_per_action_request() -> None:
    request = hitl.build_interrupt_request(
        [
            {"name": "save_report", "args": {"filename": "a"}},
            {"name": "save_report", "args": {"filename": "b"}},
        ]
    )
    reads = iter(["approve", "reject", "no"])
    response = hitl.resolve_interrupt(
        request, read=lambda _: next(reads), write=lambda _: None
    )
    assert response["decisions"] == [
        {"type": "approve"},
        {"type": "reject", "message": "no"},
    ]


def test_render_save_status_reports_the_saved_path_on_success() -> None:
    outcomes = [
        ToolMessage(
            content="Report saved to: output/x.md",
            name="save_report",
            tool_call_id="c1",
        )
    ]
    assert hitl.render_save_status(outcomes) == "Report saved to: output/x.md"


def test_render_save_status_reports_nothing_saved_on_a_rejected_call() -> None:
    outcomes = [
        ToolMessage(
            content="not good enough",
            name="save_report",
            status="error",
            tool_call_id="c1",
        )
    ]
    assert (
        hitl.render_save_status(outcomes) == "[system] No report was saved this turn."
    )


def test_render_save_status_reports_nothing_saved_on_a_tool_level_failure() -> None:
    outcomes = [
        ToolMessage(
            content="ERROR: disk full",
            name="save_report",
            status="success",
            tool_call_id="c1",
        )
    ]
    assert (
        hitl.render_save_status(outcomes) == "[system] No report was saved this turn."
    )


def test_render_save_status_ignores_ai_message_prose() -> None:
    """D8's own regression: an AIMessage claiming success must never be
    read as the truth source."""
    from langchain_core.messages import AIMessage

    outcomes = [AIMessage(content="I have saved the report successfully.")]
    assert (
        hitl.render_save_status(outcomes) == "[system] No report was saved this turn."
    )


def test_render_save_status_reads_the_last_save_report_message() -> None:
    outcomes = [
        ToolMessage(
            content="not good enough",
            name="save_report",
            status="error",
            tool_call_id="c1",
        ),
        ToolMessage(
            content="Report saved to: output/x.md",
            name="save_report",
            tool_call_id="c1",
        ),
    ]
    assert hitl.render_save_status(outcomes) == "Report saved to: output/x.md"
