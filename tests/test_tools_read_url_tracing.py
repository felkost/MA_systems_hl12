"""`read_url` records the fetched page's text onto the current span,
symmetrically with `knowledge_search`."""

from __future__ import annotations

from typing import Any

import pytest
from opentelemetry import trace as otel_trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.util._once import Once
from pydantic import SecretStr

import tools
from config import Settings


@pytest.fixture(autouse=True)
def _isolated_provider() -> Any:
    otel_trace._TRACER_PROVIDER = None
    otel_trace._TRACER_PROVIDER_SET_ONCE = Once()
    yield
    otel_trace._TRACER_PROVIDER = None
    otel_trace._TRACER_PROVIDER_SET_ONCE = Once()


def _settings() -> Settings:
    return Settings(openrouter_api_key=SecretStr("test-key"))


def test_read_url_records_retrieval_chunks_on_current_span(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    otel_trace.set_tracer_provider(provider)

    monkeypatch.setattr(
        tools,
        "read_url_with_client",
        lambda *a, **k: tools.wrap_untrusted_content("Real page content."),
    )
    monkeypatch.setattr(tools, "load_settings", _settings)

    tracer = otel_trace.get_tracer(__name__)
    with tracer.start_as_current_span("tool.read_url"):
        result = tools.read_url.invoke({"url": "https://example.com/article"})

    assert "Real page content." in result

    finished = exporter.get_finished_spans()
    recorded_span = next(s for s in finished if s.name == "tool.read_url")
    assert recorded_span.attributes is not None
    chunks = recorded_span.attributes["retrieval.chunks"]
    # The recorded chunk is the unwrapped page text, not the
    # untrusted-content markers the model itself sees.
    assert chunks == ("Real page content.",) or chunks == ["Real page content."]


def test_read_url_error_result_writes_no_retrieval_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    otel_trace.set_tracer_provider(provider)

    monkeypatch.setattr(
        tools,
        "read_url_with_client",
        lambda *a, **k: "ERROR: The page is unavailable.",
    )
    monkeypatch.setattr(tools, "load_settings", _settings)

    tracer = otel_trace.get_tracer(__name__)
    with tracer.start_as_current_span("tool.read_url"):
        result = tools.read_url.invoke({"url": "https://example.com/gone"})

    assert result.startswith("ERROR:")

    finished = exporter.get_finished_spans()
    recorded_span = next(s for s in finished if s.name == "tool.read_url")
    assert recorded_span.attributes is not None
    assert "retrieval.chunks" not in recorded_span.attributes
