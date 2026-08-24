"""One system run -> a validated span record list (stage 5 slice, D5.1;
extended stage 7, D7.1).

`load_run` reads and schema-validates `runs/<run_id>/spans.json` -- the
stage-5/stage-8 contract this project's `docs/specs/stage-5.md` splits:
this module ships the schema and the reader at stage 5. `retrieval_context`
extraction shipped early, at stage 7 (`retrieval_context_for_agent`,
`docs/specs/stage-7.md` D7.1), because R2c (Groundedness) needs it a stage
before `build_llm_test_case` was originally due -- the plan's own audit A4
names this as a hard prerequisite, not a nice-to-have. `tools_called`
extraction and the full `build_llm_test_case` (turning a `RunSpans` into a
complete DeepEval `LLMTestCase`, including `tools_called` from `tool.<name>`
spans' `tool.args`) still ship at stage 8, against this schema -- not
designed here, per the rolling-wave rule that only the stage actually
needing a piece of design commits to its exact shape.

A malformed dump raises immediately rather than returning a partial
result: an evaluation stage reading spans that don't match this schema is a
defect to surface loudly, not silently work around.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from deepeval.test_case import LLMTestCase, ToolCall
from deepeval.test_case.llm_test_case import RetrievedContextData

import paths
from middleware import was_truncated_for_span

_TOOL_SPAN_PREFIX = "tool."
# Stage 9e, D9e.7's third lever: `read_url` now writes `retrieval.chunks`
# symmetrically with `knowledge_search` (`tools.py`), so both span names
# count as retrieval here. `web_search` stays excluded on purpose -- a
# search snippet is not a retrieved document.
_RETRIEVAL_SPAN_NAMES = frozenset({"tool.knowledge_search", "tool.read_url"})

_REQUIRED_SPAN_FIELDS = (
    "trace_id",
    "span_id",
    "parent_span_id",
    "name",
    "start_time",
    "end_time",
    "status",
    "attributes",
)


@dataclass(frozen=True)
class RunSpans:
    """One run's validated span records, as written by
    `observability.SpanJsonExporter`."""

    run_id: str
    spans: list[dict[str, Any]]


def load_run(run_id: str, runs_dir: str | Path = "runs") -> RunSpans:
    """Read and schema-validate `runs/<run_id>/spans.json`.

    Parameters
    ----------
    run_id : str
    runs_dir : str or Path, default "runs"

    Returns
    -------
    RunSpans

    Raises
    ------
    FileNotFoundError
        No dump exists for `run_id`.
    ValueError
        A span record is missing a required field.
    """
    dump_path = paths.span_dump_path(run_id, runs_dir)
    if not dump_path.exists():
        raise FileNotFoundError(f"no span dump for run_id {run_id!r} at {dump_path}")

    spans = json.loads(dump_path.read_text(encoding="utf-8"))
    for index, span in enumerate(spans):
        missing = [field for field in _REQUIRED_SPAN_FIELDS if field not in span]
        if missing:
            raise ValueError(
                f"span {index} in {dump_path} is missing required field(s): "
                f"{missing}"
            )
    return RunSpans(run_id=run_id, spans=spans)


def _has_ancestor(
    span: dict[str, Any],
    *,
    by_id: dict[str, dict[str, Any]],
    agent_span_name: str,
) -> bool:
    """Does `agent_span_name` appear in `span`'s own `parent_span_id` chain?

    A chain that reaches a `parent_span_id` absent from the dump returns
    `False` -- an incomplete ancestor chain must not silently widen what
    counts as one agent's own work.
    """
    current: dict[str, Any] | None = span
    while current is not None:
        if current["name"] == agent_span_name:
            return True
        parent_id = current.get("parent_span_id")
        if parent_id is None:
            return False
        current = by_id.get(parent_id)
    return False


def _spans_in_scope(run: RunSpans, agent_span_name: str | None) -> list[dict[str, Any]]:
    """The spans `agent_span_name` selects.

    `None` means whole-run scope, and walks nothing: the Supervisor's own
    tool calls have no `agent.supervisor` span to be scoped by, since the
    only `agent.*` spans this system opens are the three delegation sites.
    """
    if agent_span_name is None:
        return list(run.spans)
    by_id = {span["span_id"]: span for span in run.spans}
    return [
        span
        for span in run.spans
        if _has_ancestor(span, by_id=by_id, agent_span_name=agent_span_name)
    ]


def retrieval_context_for_agent(
    run: RunSpans, agent_span_name: str | None
) -> list[str]:
    """Ancestor-scoped `retrieval.chunks` extraction for one agent.

    `knowledge_search` is on the Planner's, Researcher's and Critic's tool
    allowlist alike, so a flat filter on `tool.knowledge_search` mixes
    chunks none of them actually saw as their own retrieval context --
    measured on two real runs (`docs/specs/stage-7.md`, D7.2): one where a
    `knowledge_search` call belongs to `agent.planner` alongside the
    Researcher's own, another where a third belongs to `agent.critic`.

    Walks each `tool.knowledge_search`/`tool.read_url` span's
    `parent_span_id` chain and keeps it only if `agent_span_name` appears
    somewhere in that chain. A span whose walk hits a `parent_span_id`
    absent from `run.spans` is **excluded**, never assumed in scope -- an
    incomplete dump must not silently widen what counts as this agent's
    own retrieval.

    Parameters
    ----------
    run : RunSpans
    agent_span_name : str or None
        E.g. `"agent.researcher"`. `None` is whole-run scope -- every
        `knowledge_search`/`read_url` call in the run, whichever agent made
        it. That is the right scope for a system-wide metric judging the
        run's output as a whole, and the wrong one for judging a single
        agent.

    Returns
    -------
    list of str
        Every chunk string found, in span start order, with exact
        duplicates collapsed while preserving first-seen order.
    """
    seen: set[str] = set()
    chunks: list[str] = []
    for span in _spans_in_scope(run, agent_span_name):
        if span["name"] not in _RETRIEVAL_SPAN_NAMES:
            continue
        for chunk in span.get("attributes", {}).get("retrieval.chunks", []):
            if chunk not in seen:
                seen.add(chunk)
                chunks.append(chunk)
    return chunks


def tools_called_for_agent(
    run: RunSpans, agent_span_name: str | None
) -> list[ToolCall]:
    """Every tool call in scope, in the order the run made them.

    Repeated calls are **not** collapsed, unlike the chunk de-duplication
    above: calling a tool twice is a fact the dump records, and a caller
    comparing against an expected sequence needs to see it. (DeepEval's own
    `ToolCorrectnessMetric` cannot credit a repeated expectation -- it
    matches into a set of value-compared `ToolCall`s -- but that is the
    metric's limitation to absorb in its `expected_tools`, not a reason for
    this extraction to misreport what happened.)

    Parameters
    ----------
    run : RunSpans
    agent_span_name : str or None
        `None` is whole-run scope, which includes each sub-agent's own tool
        calls as well as the coordinator's -- "the Supervisor's tools" and
        "every tool call in the run" are different sets, and this is the
        second.

    Returns
    -------
    list of ToolCall
        `name` is the span name minus its `tool.` prefix;
        `input_parameters` comes from the span's `tool.args` attribute when
        present, and is left unset when it is not (a tool may take no
        arguments).

    Raises
    ------
    ValueError
        A `tool.args` attribute is neither truncated nor parsable JSON.
        The tracing middleware always writes it as a JSON string and caps
        its length, so a value that is short enough to be intact and still
        will not parse is a real defect; a truncated one is ordinary, and
        simply yields a call with no `input_parameters`.
    """
    calls: list[ToolCall] = []
    for span in _spans_in_scope(run, agent_span_name):
        name = str(span["name"])
        if not name.startswith(_TOOL_SPAN_PREFIX):
            continue
        raw_args = span.get("attributes", {}).get("tool.args")
        parsed: dict[str, Any] | None = None
        # A truncated payload is normal, not corrupt: the tracing
        # middleware caps `tool.args`, and a Supervisor delegation argument
        # routinely runs past that cap, so valid JSON arrives with its tail
        # removed. The arguments are simply unrecoverable then -- which is
        # different from a value that was never JSON at all, and only the
        # second is a defect worth raising on. Measured the hard way: the
        # first stage-8 live run raised here on a real, healthy run.
        if raw_args is not None and not was_truncated_for_span(str(raw_args)):
            try:
                parsed = json.loads(raw_args)
            except (TypeError, json.JSONDecodeError) as error:
                raise ValueError(
                    f"span {span['span_id']!r} in run {run.run_id!r} carries a "
                    f"tool.args attribute that is neither truncated nor valid "
                    f"JSON: {raw_args!r}"
                ) from error
        # `input_parameters` carries a default, but pydantic's mypy plugin
        # treats an aliased field as a required argument, so it is passed
        # explicitly rather than silenced.
        calls.append(
            ToolCall(name=name[len(_TOOL_SPAN_PREFIX) :], input_parameters=parsed)
        )
    return calls


def total_agent_cost(run: RunSpans) -> float:
    """Sum `gen_ai.usage.cost_usd` across every span in one run.

    Stage 9a's own measured-cost obligation (`docs/specs/stage-9a.md` D9a.8):
    the same attribute stage 7/8 already read by hand when reporting cost,
    made reusable. A span with no such attribute (a tool span, not a model
    call) contributes `0.0`, never an error -- only a subset of spans ever
    carry it.

    Parameters
    ----------
    run : RunSpans

    Returns
    -------
    float
    """
    return sum(
        span.get("attributes", {}).get("gen_ai.usage.cost_usd", 0.0) or 0.0
        for span in run.spans
    )


def build_llm_test_case(
    run: RunSpans,
    *,
    input: str,
    actual_output: str,
    agent_span_name: str | None = None,
    expected_output: str | None = None,
    expected_tools: list[ToolCall] | None = None,
    name: str | None = None,
) -> LLMTestCase:
    """Compose one run's span dump into a DeepEval `LLMTestCase`.

    Parameters
    ----------
    run : RunSpans
    input : str
        What the agent or system under test was actually sent.
    actual_output : str
        Its rendered response.
    agent_span_name : str or None, default None
        Scopes both `retrieval_context` and `tools_called`. `None` is
        whole-run scope -- what a system-wide metric needs, and what the
        Supervisor's own tool calls require, since no `agent.supervisor`
        span exists.
    expected_output : str, optional
    expected_tools : list of ToolCall, optional
        Both left `None` when not supplied, never defaulted to an empty
        value: a metric reading `expected_tools` would treat `[]` as a real
        expectation rather than as "this caller is not measuring that".
    name : str, optional
        Carried into DeepEval's own persisted run file, which is otherwise
        the only record of which test produced a case and has no source-file
        field of its own (`evals/summarize_e2e.py`, D9d.8).

    Returns
    -------
    LLMTestCase
    """
    # `retrieval_context`'s declared element type is a union and `list` is
    # invariant, so the narrower `list[str]` is rebound through an
    # explicitly-typed local rather than passed directly.
    retrieval_context: list[str | RetrievedContextData] = list(
        retrieval_context_for_agent(run, agent_span_name)
    )
    return LLMTestCase(
        name=name,
        input=input,
        actual_output=actual_output,
        expected_output=expected_output,
        retrieval_context=retrieval_context,
        tools_called=tools_called_for_agent(run, agent_span_name),
        expected_tools=expected_tools,
    )
