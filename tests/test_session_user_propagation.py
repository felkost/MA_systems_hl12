"""`observability.run_context`/`set_trace_io` -- session/user tracking
(requirement R2), stage 2, `docs/specs/stage-2.md`.

Proves every span of one turn carries `session.id`/`user.id` under the
SDK's own constant names (never a hand-typed string literal), and that the
root span carries `TRACE_INPUT`/`TRACE_OUTPUT` after `set_trace_io` -- what
stage 4's judges will read `{{input}}`/`{{output}}` from.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langfuse._client.attributes import LangfuseOtelSpanAttributes
from opentelemetry import trace
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from pydantic import SecretStr

import observability
from config import Settings
from tests.test_observability import _reset_otel_global_state


def _settings(**overrides: Any) -> Settings:
    # See tests/test_observability.py's identical helper: without this
    # default, a developer `.env` with TRACING_ENABLED=true and real
    # Langfuse keys makes configure_observability(_settings()) build a
    # REAL client here too.
    overrides.setdefault("tracing_enabled", False)
    return Settings(openrouter_api_key=SecretStr("test-key"), **overrides)


def test_every_span_of_one_turn_carries_session_and_user_under_the_sdk_constant_names(
    tmp_path: Path,
) -> None:
    _reset_otel_global_state()
    handle = observability.configure_observability(
        _settings(span_dump_dir=str(tmp_path), log_dir=str(tmp_path))
    )
    exporter = InMemorySpanExporter()
    handle.tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = trace.get_tracer(__name__)

    try:
        with observability.run_context(
            session_id="sess-1", user_id="user-1", run_id="run-1"
        ):
            with tracer.start_as_current_span("repl.question"):
                with tracer.start_as_current_span("agent.researcher"):
                    pass
        handle.shutdown()
    finally:
        _reset_otel_global_state()

    spans = exporter.get_finished_spans()
    assert {s.name for s in spans} == {"repl.question", "agent.researcher"}
    for span in spans:
        assert span.attributes is not None
        assert span.attributes[LangfuseOtelSpanAttributes.TRACE_SESSION_ID] == "sess-1"
        assert span.attributes[LangfuseOtelSpanAttributes.TRACE_USER_ID] == "user-1"


def test_root_span_carries_trace_input_and_output_after_set_trace_io(
    tmp_path: Path,
) -> None:
    _reset_otel_global_state()
    handle = observability.configure_observability(
        _settings(span_dump_dir=str(tmp_path), log_dir=str(tmp_path))
    )
    exporter = InMemorySpanExporter()
    handle.tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = trace.get_tracer(__name__)

    try:
        with observability.run_context(
            session_id="sess-2", user_id="user-2", run_id="run-2"
        ):
            with tracer.start_as_current_span("repl.question"):
                observability.set_trace_io(
                    input="What is naive RAG?", output="It is..."
                )
        handle.shutdown()
    finally:
        _reset_otel_global_state()

    root = next(s for s in exporter.get_finished_spans() if s.name == "repl.question")
    assert root.attributes is not None
    assert (
        root.attributes[LangfuseOtelSpanAttributes.TRACE_INPUT] == "What is naive RAG?"
    )
    assert root.attributes[LangfuseOtelSpanAttributes.TRACE_OUTPUT] == "It is..."
