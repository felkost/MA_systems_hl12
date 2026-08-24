"""Owns the process's one `TracerProvider`, the offline span dump, rotating
file logs, and cost computed from token counts (stage 5,
`docs/specs/stage-5.md`).

**One provider, two processors.** `configure_observability` builds exactly
one `TracerProvider` and attaches `LangfuseSpanProcessor` (via
`Langfuse(tracer_provider=...)`, only when `settings.tracing_enabled`) and a
project-owned `SpanJsonExporter` on a `SimpleSpanProcessor` (D5.4). The
offline dump is deliberately **not** a `BatchSpanProcessor`: it writes each
span synchronously as it ends, so a hard crash never loses its tail --
Langfuse export staying batched (and therefore losing an unflushed tail on a
crash) is accepted, since it is best-effort UI-quality tracing, never the
evaluation pipeline's data source.

**`run_id` is not a constructor argument (D5.7, corrected mid-spec-review).**
The first draft took `run_id` at `configure_observability()` call time, which
cannot work: the provider is built once per process, while `run_id` is fresh
per REPL turn. Instead, `run_id` rides OpenTelemetry's own `Baggage` --
`main.py`'s per-turn loop attaches it to the ambient context before opening
`repl.question`, and `RunContextStampingProcessor` (registered first, so every
other processor sees the attribute already set) copies it onto every span's
own attributes at `on_start`. `SpanJsonExporter` then routes each span it
receives by that attribute, keyed per run, never by a fixed value from
construction time. Verified live: two sequential turns with different
`run_id`s produced spans (root and nested) each carrying only their own
turn's value, no cross-contamination.

**Only `main.py` (interface) imports this module.** Every other layer
reaches OpenTelemetry's own ambient global tracer directly
(`opentelemetry.trace.get_tracer(__name__)`), a third-party import invisible
to `tests/test_layering.py`'s AST walk over project-local module names
(D5.3) -- this module never crosses the `application`/`infra` boundary that
would otherwise forbid importing an `obs`-layer module from there.
"""

from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from dataclasses import dataclass
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Iterator, Sequence

from langfuse import Langfuse, propagate_attributes
from langfuse._client.attributes import LangfuseOtelSpanAttributes
from opentelemetry import baggage, context, trace
from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor, TracerProvider
from opentelemetry.sdk.trace.export import (
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)
from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased

import paths
from config import Settings
from models import (  # noqa: F401 -- re-exported
    PRICE_TABLE,
    compute_cost,
    compute_cost_or_raise,
)

_MAX_QUEUE_LENGTH = 10_000  # generous headroom for one run's own spans

_CONFIGURED = False


@dataclass(frozen=True)
class ObservabilityHandle:
    """What `configure_observability` returns."""

    tracer_provider: TracerProvider
    langfuse_client: Langfuse | None

    def shutdown(self) -> None:
        """Flush every registered `SpanProcessor`, including the batched
        Langfuse one, before the process exits (D5.11)."""
        self.tracer_provider.shutdown()


def configure_observability(settings: Settings) -> ObservabilityHandle:
    """Build and globally register the process's one `TracerProvider`.

    Parameters
    ----------
    settings : Settings

    Returns
    -------
    ObservabilityHandle
        Exposes the one `Langfuse` client this call builds
        (`langfuse_client`, `None` when `settings.tracing_enabled` is
        `False`) so a caller does not construct a second one -- a second
        `Langfuse(public_key=...)` call with the same key returns the SDK's
        own cached singleton and silently discards `tracer_provider=`/
        `should_export_span=` (D2.1, `docs/specs/stage-2.md`, section 1).

    Raises
    ------
    RuntimeError
        On a second call in the same process. OpenTelemetry's own
        `set_tracer_provider` silently no-ops on a second call rather than
        raising (measured live, D5.13) -- this project enforces what the
        SDK does not, so a caller bug (e.g. `main()` re-entered) is loud
        rather than silently inert.
    """
    global _CONFIGURED
    if _CONFIGURED:
        raise RuntimeError(
            "configure_observability() already called once in this process"
        )
    _CONFIGURED = True

    sampler = ParentBased(TraceIdRatioBased(settings.trace_sample_rate))
    provider = TracerProvider(sampler=sampler)
    trace.set_tracer_provider(provider)

    # First, so every later processor sees run_id/session.id/user.id already
    # stamped (D5.7, extended by D2.5 to session/user).
    provider.add_span_processor(RunContextStampingProcessor())

    runs_dir = settings.span_dump_dir or "runs"
    provider.add_span_processor(
        SimpleSpanProcessor(SpanJsonExporter(runs_dir=runs_dir))
    )

    langfuse_client: Langfuse | None = None
    if settings.tracing_enabled:
        assert settings.langfuse_public_key is not None
        assert settings.langfuse_secret_key is not None
        langfuse_client = Langfuse(
            public_key=settings.langfuse_public_key.get_secret_value(),
            secret_key=settings.langfuse_secret_key.get_secret_value(),
            host=settings.langfuse_base_url,
            tracer_provider=provider,
            # Confirmed live this session, not in the SDK-measurements table
            # above: LangfuseSpanProcessor's DEFAULT should_export_span
            # (langfuse._client.span_filter.is_default_export_span) drops
            # any span with no gen_ai.*-prefixed attribute, unless it also
            # matches a "known LLM instrumentor" scope. Every model.* span
            # (gen_ai.usage.* attributes) passed; every agent.*/tool.*/
            # repl.question span (no gen_ai attributes) was silently
            # filtered before ever reaching the network -- reproducing
            # hl10's "collection-correct, publication-incomplete" defect
            # for a different underlying reason (a default filter, not a
            # LangChain-integration-only publisher). This override only
            # actually takes effect because `main.py` never builds a second
            # `Langfuse(public_key=...)` client with the same key (D2.1) --
            # `LangfuseResourceManager` is a singleton keyed by public_key,
            # and a second construction silently discards this keyword,
            # the exact defect stage 2 diagnosed and fixed
            # (`docs/specs/stage-2.md`, section 1). Nothing else is attached
            # to this provider (D5.3), so accepting every span it ever sees
            # is safe.
            should_export_span=lambda span: True,
        )

    configure_file_logging("main", settings, log_dir=settings.log_dir or "logs")

    return ObservabilityHandle(
        tracer_provider=provider, langfuse_client=langfuse_client
    )


@contextmanager
def run_context(*, session_id: str, user_id: str, run_id: str) -> Iterator[None]:
    """Scope one REPL turn's `session.id`/`user.id`/`run_id` (D2.5).

    Attaches `session_id`/`user_id`/`run_id` as OTel `Baggage`, which
    `RunContextStampingProcessor` copies onto every span opened afterwards
    -- deterministic and provider-side, so the offline JSON dump carries
    `session.id`/`user.id` on every span the same way it already carries
    `run_id`, whether or not a Langfuse client is attached
    (`settings.tracing_enabled`). This is *not* `propagate_attributes`'s own
    re-application path: that path lives entirely inside
    `LangfuseSpanProcessor.on_start` (`langfuse/_client/span_processor.py:147-169`,
    measured, not assumed) and only runs when `tracing_enabled=True` builds
    one -- relying on it alone would leave every nested `agent.*`/`tool.*`
    span in the offline dump without `session.id`/`user.id` whenever tracing
    is off, the project's own default (`docs/specs/stage-2.md`, section 1,
    the finding behind D2.5).

    `propagate_attributes` is still entered here too, for the SDK's own
    recommended usage and whatever internal Langfuse-side features key off
    its Context-stored values when a real client is attached -- harmless
    and redundant with the baggage stamping above when there is no such
    client.

    Parameters
    ----------
    session_id : str
        The same value across every turn in one REPL session -- what makes
        Langfuse group 3-5 traces into one session (requirement R2), not
        3-5 separate one-trace sessions.
    user_id : str
    run_id : str
        Fresh per turn (D5.7).
    """
    ctx = baggage.set_baggage("run_id", run_id)
    ctx = baggage.set_baggage("session_id", session_id, context=ctx)
    ctx = baggage.set_baggage("user_id", user_id, context=ctx)
    token = context.attach(ctx)
    try:
        with propagate_attributes(session_id=session_id, user_id=user_id):
            yield None
    finally:
        context.detach(token)


def set_trace_io(*, input: str, output: str) -> None:
    """Stamp trace-level input/output on the current span (D2.4).

    Sets `LangfuseOtelSpanAttributes.TRACE_INPUT`/`TRACE_OUTPUT` directly on
    `trace.get_current_span()` -- the same low-level mechanism
    `LangfuseObservationWrapper.set_trace_io` uses internally, without
    depending on that method: it is `@deprecated` and only defined on a
    Langfuse *wrapper* span, which this project never constructs (it only
    ever opens raw OTel spans). Must be called while the root `repl.question`
    span is still current -- once its `with` block exits,
    `trace.get_current_span()` returns the non-recording default span and
    this call silently no-ops, exactly as the deprecated method would too.
    """
    trace.get_current_span().set_attribute(
        LangfuseOtelSpanAttributes.TRACE_INPUT, input
    )
    trace.get_current_span().set_attribute(
        LangfuseOtelSpanAttributes.TRACE_OUTPUT, output
    )


class RunContextStampingProcessor(SpanProcessor):
    """Copies the ambient `Baggage` `run_id`/`session_id`/`user_id` values
    onto every span's own attributes at creation time -- registered first in
    the provider's processor list so both the Langfuse and offline-dump
    processors see them already set (D5.7, extended by D2.5). A future edit
    that reorders `add_span_processor` calls ahead of this one silently
    loses `run_id` on every span; pinned by a declared test constructing the
    provider with processors added out of order.

    Session/user attribution is stamped here, deterministically, rather than
    relying only on `langfuse.propagate_attributes`'s own re-application
    path: that path lives entirely inside `LangfuseSpanProcessor.on_start`
    and only runs when a real Langfuse client is attached
    (`settings.tracing_enabled`) -- this class instead makes `session.id`/
    `user.id` present on every span in the offline dump too, the project's
    own default posture (`docs/specs/stage-2.md`, section 1)."""

    def on_start(self, span: Any, parent_context: Any = None) -> None:
        run_id = baggage.get_baggage("run_id", parent_context)
        if run_id is not None:
            span.set_attribute("run_id", run_id)
        session_id = baggage.get_baggage("session_id", parent_context)
        if session_id is not None:
            span.set_attribute(LangfuseOtelSpanAttributes.TRACE_SESSION_ID, session_id)
        user_id = baggage.get_baggage("user_id", parent_context)
        if user_id is not None:
            span.set_attribute(LangfuseOtelSpanAttributes.TRACE_USER_ID, user_id)

    def on_end(self, span: ReadableSpan) -> None:
        return None

    def shutdown(self) -> None:
        return None

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        return True


class SpanJsonExporter(SpanExporter):
    """Writes each ended span into `runs/<run_id>/spans.json`, keyed by the
    span's own `run_id` attribute (stamped by `RunContextStampingProcessor`) --
    one long-lived exporter for the whole process, never bound to a single
    fixed run at construction (D5.7, corrected).

    `SimpleSpanProcessor.on_end` calls `export()` once per single span
    (measured: `opentelemetry-sdk`'s `SimpleSpanProcessor` is synchronous,
    one-span batches) -- `export()` is therefore not a byte-append, which
    could never produce valid JSON for a list-of-objects file. It keeps an
    in-memory `dict[run_id, list[span_record]]` accumulator and, on every
    call, appends the new span's record and rewrites
    `paths.span_dump_path(run_id)` whole -- valid JSON after every single
    `export()` call, not only after the last one. O(n^2) in spans per run
    across a run's lifetime, accepted at this project's real scale (a few
    dozen spans/run), not optimised pre-emptively.
    """

    def __init__(self, runs_dir: str | Path = "runs") -> None:
        self._runs_dir = runs_dir
        self._by_run: dict[str, list[dict[str, Any]]] = {}

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        for span in spans:
            attributes = dict(span.attributes or {})
            run_id = attributes.get("run_id")
            if run_id is None:
                logging.getLogger(__name__).warning(
                    "span %r has no run_id attribute -- dropped from the "
                    "offline dump",
                    span.name,
                )
                continue
            record = _span_record(span, attributes)
            self._by_run.setdefault(str(run_id), []).append(record)
            path = paths.span_dump_path(str(run_id), self._runs_dir)
            path.write_text(
                json.dumps(self._by_run[str(run_id)], indent=2), encoding="utf-8"
            )
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        return None


def _span_record(span: ReadableSpan, attributes: dict[str, Any]) -> dict[str, Any]:
    context = span.get_span_context()
    assert context is not None
    parent = span.parent
    return {
        "trace_id": format(context.trace_id, "032x"),
        "span_id": format(context.span_id, "016x"),
        "parent_span_id": format(parent.span_id, "016x") if parent else None,
        "name": span.name,
        "start_time": span.start_time,
        "end_time": span.end_time,
        "status": "ERROR" if span.status.is_ok is False else "OK",
        "attributes": attributes,
    }


class _TraceIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        span_context = trace.get_current_span().get_span_context()
        record.trace_id = (
            format(span_context.trace_id, "032x") if span_context.is_valid else "-"
        )
        return True


def configure_file_logging(
    service: str, settings: Settings, *, log_dir: str | Path = "logs"
) -> logging.Logger:
    """A rotating file logger for one long-lived process.

    Parameters
    ----------
    service : str
        Process name, e.g. `"main"` -- the file is `<service>.log`
        (`paths.log_path`).
    settings : Settings
    log_dir : str or Path, default "logs"
        Explicit so a test can redirect it under `tmp_path`.

    Returns
    -------
    logging.Logger
        Every record carries `trace_id` (the current OTel span's, hex) or
        `"-"` when no span is active -- the post-mortem artefact for a hard
        crash, since `BatchSpanProcessor` keeps spans in memory and loses
        the tail.
    """
    logger = logging.getLogger(f"hl11.{service}")
    logger.setLevel(logging.INFO)
    # `logger` is the same named object across every call in this process
    # (Python's logging module keys loggers by name globally) -- since
    # `configure_observability` now calls this on every invocation (D2.3),
    # a fresh call must not pile a new handler onto whatever a previous
    # invocation (a previous test, in the gate) already attached.
    logger.handlers.clear()
    logger.filters.clear()
    path = paths.log_path(service, log_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(
        path, maxBytes=10_000_000, backupCount=5, encoding="utf-8"
    )
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s trace_id=%(trace_id)s - %(message)s"
        )
    )
    logger.addHandler(handler)
    logger.addFilter(_TraceIdFilter())
    return logger
