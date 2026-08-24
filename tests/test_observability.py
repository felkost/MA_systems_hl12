"""`observability.py` -- TracerProvider ownership, Langfuse attachment, the
offline span dump, cost computation, rotating file logs (stage 5,
`docs/specs/stage-5.md`).

D5.14 -- OpenTelemetry's global `TracerProvider` registration is a
process-wide singleton with no public reset API. `_reset_otel_global_state`
resets both OTel's own private globals and this module's own `_CONFIGURED`
sentinel (D5.13); an autouse fixture applies it before and after every test
in this file. `tests/test_main.py` imports and reuses this same helper
around its one test that also needs a registered provider (D5.14, round 2).
"""

from __future__ import annotations

import json
import types
from pathlib import Path
from typing import Any, cast

import pytest
from langfuse import Langfuse
from opentelemetry import baggage, context, trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.util._once import Once
from pydantic import SecretStr

import observability
import paths
from config import Settings


def _settings(**overrides: Any) -> Settings:
    # `tracing_enabled` defaults False here, never inherited from a real
    # `.env` -- pydantic-settings falls through to the dotenv/environment
    # source for any field this helper doesn't pass explicitly, so a
    # developer machine with TRACING_ENABLED=true and real Langfuse keys
    # would otherwise make every `configure_observability(_settings())`
    # call in this file build a REAL client and export real spans to the
    # real project (confirmed live, stage 2: two of this file's tests did
    # exactly that before this default was added -- CLAUDE.md's "a gate
    # test that reaches the network is a broken test, not a slow one").
    # A test that deliberately wants `tracing_enabled=True` overrides it.
    overrides.setdefault("tracing_enabled", False)
    return Settings(openrouter_api_key=SecretStr("test-key"), **overrides)


def _reset_otel_global_state() -> None:
    trace._TRACER_PROVIDER = None
    trace._TRACER_PROVIDER_SET_ONCE = Once()
    observability._CONFIGURED = False


@pytest.fixture(autouse=True)
def _isolated_otel_state() -> Any:
    _reset_otel_global_state()
    yield
    _reset_otel_global_state()


def test_configure_observability_registers_one_tracer_provider(
    tmp_path: Path,
) -> None:
    handle = observability.configure_observability(_settings(log_dir=str(tmp_path)))
    assert trace.get_tracer_provider() is handle.tracer_provider
    assert isinstance(handle.tracer_provider, TracerProvider)


def test_second_configure_call_raises(tmp_path: Path) -> None:
    observability.configure_observability(_settings(log_dir=str(tmp_path)))
    with pytest.raises(RuntimeError):
        observability.configure_observability(_settings(log_dir=str(tmp_path)))


def test_no_langfuse_processor_when_tracing_disabled(tmp_path: Path) -> None:
    handle = observability.configure_observability(
        _settings(tracing_enabled=False, log_dir=str(tmp_path))
    )
    processor_types = [
        type(p).__name__
        for p in handle.tracer_provider._active_span_processor._span_processors
    ]
    assert "LangfuseSpanProcessor" not in processor_types


def test_langfuse_attaches_to_the_supplied_provider_when_tracing_enabled(
    tmp_path: Path,
) -> None:
    # A unique public_key per test -- Langfuse() is a thread-safe singleton
    # keyed by public_key within a process (its own docstring: "a
    # thread-safe singleton pattern for each unique public API key"). Two
    # gate tests sharing one fake key would have the second construction
    # silently return the first test's cached client, still bound to the
    # PREVIOUS (already-reset-by-fixture) provider -- found live this
    # session via a genuinely flaky test under pytest-randomly.
    settings = _settings(
        tracing_enabled=True,
        langfuse_public_key=SecretStr("pk-test-attaches"),
        langfuse_secret_key=SecretStr("sk-test-attaches"),
        log_dir=str(tmp_path),
    )
    handle = observability.configure_observability(settings)
    processor_types = [
        type(p).__name__
        for p in handle.tracer_provider._active_span_processor._span_processors
    ]
    assert "LangfuseSpanProcessor" in processor_types


def test_langfuse_exports_every_span_not_only_gen_ai_ones(tmp_path: Path) -> None:
    """Confirmed live this session, not merely from source: Langfuse's
    default `should_export_span` (`is_default_export_span`) drops any span
    with no `gen_ai.*`-prefixed attribute -- silently filtering out every
    `agent.*`/`tool.*`/`repl.question` span before it ever reached the
    network, reproducing hl10's "collection-correct, publication-incomplete"
    defect for a new reason. `configure_observability` must override this
    with `should_export_span=lambda span: True`."""
    settings = _settings(
        tracing_enabled=True,
        langfuse_public_key=SecretStr("pk-test-export-filter"),
        langfuse_secret_key=SecretStr("sk-test-export-filter"),
        log_dir=str(tmp_path),
    )
    handle = observability.configure_observability(settings)
    processors = handle.tracer_provider._active_span_processor._span_processors
    langfuse_processor = next(
        p for p in processors if type(p).__name__ == "LangfuseSpanProcessor"
    )
    non_gen_ai_span = types.SimpleNamespace(attributes={"run_id": "x"})
    # cast: `type(p).__name__` string comparison above doesn't narrow the
    # generator's static type past the base SpanProcessor, which declares
    # no _should_export_span attribute -- the real LangfuseSpanProcessor
    # does, confirmed against the installed langfuse SDK source.
    assert cast(Any, langfuse_processor)._should_export_span(non_gen_ai_span) is True


def test_offline_span_dump_uses_a_simple_not_batch_processor(tmp_path: Path) -> None:
    handle = observability.configure_observability(_settings(log_dir=str(tmp_path)))
    processors = handle.tracer_provider._active_span_processor._span_processors
    simple = [p for p in processors if isinstance(p, SimpleSpanProcessor)]
    assert any(
        isinstance(p.span_exporter, observability.SpanJsonExporter) for p in simple
    )


# -- Stage 2, D2.1: the provider-hijack non-regression tests. Reproduces the
# diagnosed defect (`docs/specs/stage-2.md`, section 1): `main.py` used to
# build a prompt-store `Langfuse` client before `configure_observability`,
# and `LangfuseResourceManager`'s per-public_key singleton silently
# discarded the second construction's `tracer_provider=`/
# `should_export_span=` -- the project's own provider (and its
# `should_export_span` override) never actually took effect.


def test_configure_observability_then_reused_client_still_writes_the_offline_dump(
    tmp_path: Path,
) -> None:
    import main as main_module

    settings = _settings(
        tracing_enabled=True,
        langfuse_public_key=SecretStr("pk-test-d21"),
        langfuse_secret_key=SecretStr("sk-test-d21"),
        span_dump_dir=str(tmp_path),
        log_dir=str(tmp_path),
    )
    handle = observability.configure_observability(settings)
    # D2.1: main.py's real startup order -- configure_observability first,
    # then build_prompt_store reusing the one client it built. Before this
    # stage, `build_prompt_store` took no `client=` keyword at all, so this
    # call would have raised `TypeError` -- the RED state this test pins.
    main_module.build_prompt_store(settings, client=handle.langfuse_client)

    run_id = "run-d21"
    token = context.attach(baggage.set_baggage("run_id", run_id))
    try:
        with trace.get_tracer(__name__).start_as_current_span("repl.question"):
            pass
    finally:
        context.detach(token)
    handle.shutdown()

    dump = json.loads(
        paths.span_dump_path(run_id, tmp_path).read_text(encoding="utf-8")
    )
    assert dump[0]["name"] == "repl.question"


def test_a_second_langfuse_client_with_the_same_public_key_ignores_new_kwargs(
    tmp_path: Path,
) -> None:
    """Pins the load-bearing SDK finding behind D2.1 (`docs/specs/stage-2.md`,
    section 1): `LangfuseResourceManager` is a singleton keyed by
    `public_key`, so a second `Langfuse(public_key=<same key>, ...)`
    construction returns the first instance and silently ignores every
    other keyword this second call passed -- including a
    `should_export_span` that would otherwise have re-enabled the SDK's
    default gen_ai-only filter."""
    settings = _settings(
        tracing_enabled=True,
        langfuse_public_key=SecretStr("pk-test-singleton"),
        langfuse_secret_key=SecretStr("sk-test-singleton"),
        span_dump_dir=str(tmp_path),
        log_dir=str(tmp_path),
    )
    handle = observability.configure_observability(settings)
    assert handle.langfuse_client is not None

    second = Langfuse(
        public_key="pk-test-singleton",
        secret_key="sk-test-singleton",
        host=settings.langfuse_base_url,
        tracer_provider=TracerProvider(),
        should_export_span=lambda span: False,
    )
    # `Langfuse(...)` itself is not a singleton -- every call returns a new
    # wrapper object (measured live: `second is handle.langfuse_client` is
    # False). What is cached is its internal `_resources`
    # (`LangfuseResourceManager`, keyed by `public_key`): both wrappers
    # share the exact same one, so the second call's `tracer_provider=`/
    # `should_export_span=` never took effect on anything either object
    # actually uses.
    assert second is not handle.langfuse_client
    resources = cast(Any, second)._resources
    assert resources is cast(Any, handle.langfuse_client)._resources

    processors = handle.tracer_provider._active_span_processor._span_processors
    langfuse_processor = next(
        p for p in processors if type(p).__name__ == "LangfuseSpanProcessor"
    )
    non_gen_ai_span = types.SimpleNamespace(attributes={"run_id": "x"})
    assert cast(Any, langfuse_processor)._should_export_span(non_gen_ai_span) is True


def test_configure_observability_writes_a_rotating_log_under_the_configured_dir(
    tmp_path: Path,
) -> None:
    observability.configure_observability(_settings(log_dir=str(tmp_path)))
    assert paths.log_path("main", tmp_path).exists()


def test_compute_cost_returns_none_for_an_unpriced_model() -> None:
    assert observability.compute_cost("vendor/unpriced-model", 100, 50) is None


def test_compute_cost_uses_the_price_table() -> None:
    cost = observability.compute_cost("openai/gpt-4.1-mini", 1000, 500)
    prompt_price, completion_price = observability.PRICE_TABLE["openai/gpt-4.1-mini"]
    assert cost == pytest.approx(1000 * prompt_price + 500 * completion_price)


def test_price_table_covers_every_resolved_agent_and_judge_model() -> None:
    settings = _settings(judge_model_name="google/gemini-2.5-pro")
    for role in Settings.ROLES:
        assert settings.resolved_model(role) in observability.PRICE_TABLE
    assert settings.judge_model_name in observability.PRICE_TABLE


def test_compute_cost_or_raise_returns_the_same_value_as_compute_cost() -> None:
    assert observability.compute_cost_or_raise(
        "openai/gpt-4.1-mini", 1000, 500
    ) == observability.compute_cost("openai/gpt-4.1-mini", 1000, 500)


def test_compute_cost_or_raise_names_the_model_for_an_unpriced_one() -> None:
    """D9e.17: `tests/test_e2e.py` used to apply `or 0.0` to `compute_cost`,
    which silently recorded an unpriced judge model's entire cost as $0.00.
    An unknown model must fail loudly instead, naming the model, not return
    a silent zero."""
    with pytest.raises(ValueError, match="vendor/unpriced-model"):
        observability.compute_cost_or_raise("vendor/unpriced-model", 100, 50)


def test_rotating_file_logger_writes_trace_id_or_dash(tmp_path: Path) -> None:
    logger = observability.configure_file_logging(
        "test-service", _settings(), log_dir=tmp_path
    )
    logger.info("no span active")
    provider = TracerProvider()
    tracer = provider.get_tracer(__name__)
    with tracer.start_as_current_span("some.span"):
        logger.info("span active")
    for handler in logger.handlers:
        handler.flush()
    lines = [
        line
        for line in paths.log_path("test-service", tmp_path)
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]
    assert len(lines) == 2
    assert "trace_id=- " in lines[0]
    assert "trace_id=- " not in lines[1]


def test_run_id_stamping_tags_every_span_in_a_turn_not_the_previous_turns(
    tmp_path: Path,
) -> None:
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    # span_dump_dir redirected to tmp_path even though this test only
    # inspects the in-memory exporter below -- configure_observability
    # always attaches a real SpanJsonExporter too (D5.4), which would
    # otherwise write run-A/run-B directories into the real project's
    # runs/ on every gate run.
    handle = observability.configure_observability(
        _settings(span_dump_dir=str(tmp_path), log_dir=str(tmp_path))
    )
    exporter = InMemorySpanExporter()
    handle.tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = trace.get_tracer(__name__)

    for run_id in ("run-A", "run-B"):
        token = context.attach(baggage.set_baggage("run_id", run_id))
        try:
            with tracer.start_as_current_span("repl.question"):
                with tracer.start_as_current_span("agent.planner"):
                    pass
        finally:
            context.detach(token)

    by_run: dict[Any, list[str]] = {}
    for span in exporter.get_finished_spans():
        assert span.attributes is not None
        by_run.setdefault(span.attributes.get("run_id"), []).append(span.name)
    assert set(by_run["run-A"]) == {"repl.question", "agent.planner"}
    assert set(by_run["run-B"]) == {"repl.question", "agent.planner"}


def test_span_json_exporter_routes_by_each_spans_own_run_id_attribute(
    tmp_path: Path,
) -> None:
    handle = observability.configure_observability(
        _settings(span_dump_dir=str(tmp_path), log_dir=str(tmp_path))
    )
    tracer = trace.get_tracer(__name__)
    for run_id in ("run-A", "run-B"):
        token = context.attach(baggage.set_baggage("run_id", run_id))
        try:
            with tracer.start_as_current_span("repl.question"):
                pass
        finally:
            context.detach(token)
    handle.shutdown()

    for run_id in ("run-A", "run-B"):
        dump = json.loads(
            paths.span_dump_path(run_id, tmp_path).read_text(encoding="utf-8")
        )
        assert len(dump) == 1
        assert dump[0]["name"] == "repl.question"
        assert dump[0]["attributes"]["run_id"] == run_id
