"""`truncate_for_span` and `was_truncated_for_span` must agree (stage 8).

Non-regression for a defect the stage-8 live run found: `tools_called_for_agent`
raised on a real Supervisor turn because `tool.args` was **truncated** and
therefore not parsable JSON. Truncation is normal -- a delegation argument
routinely exceeds the payload cap -- so the reader has to tell a cut payload
apart from a corrupt one, which means the writer's marker and the reader's
predicate must not drift.

These tests exist so that changing the marker's wording without changing the
predicate is a red gate test rather than a live-run surprise.
"""

from __future__ import annotations

import json

from middleware import truncate_for_span, was_truncated_for_span


def test_untruncated_text_is_not_marked() -> None:
    text = "short enough"
    assert truncate_for_span(text, 100) == text
    assert was_truncated_for_span(text) is False


def test_truncated_text_is_detected_by_the_predicate() -> None:
    truncated = truncate_for_span("x" * 500, 100)
    assert truncated != "x" * 500
    assert was_truncated_for_span(truncated) is True


def test_a_truncated_json_payload_stops_being_parsable() -> None:
    # The exact shape that broke the first stage-8 live run: valid JSON in,
    # unparsable text out, with nothing wrong anywhere.
    payload = json.dumps({"task": "y" * 500})
    truncated = truncate_for_span(payload, 100)

    assert was_truncated_for_span(truncated) is True
    try:
        json.loads(truncated)
    except json.JSONDecodeError:
        pass
    else:  # pragma: no cover - the raise below is the real failure path
        raise AssertionError("a truncated JSON payload must not still parse")


def test_text_merely_mentioning_truncation_is_not_treated_as_truncated() -> None:
    # The marker is anchored to the end of the string, so a payload whose
    # own content discusses truncation is not mistaken for a cut one.
    assert was_truncated_for_span("...[truncated, 5 chars total] and more") is False
