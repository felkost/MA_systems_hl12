"""Fabrication-candidate extraction and a run-scoped grounding nudge for the
Researcher (stage 9e, D9e.7a/D9e.7b, `docs/specs/stage-9e.md`).

**D9e.7a -- the extractor requires a digit or an all-caps token.** A v1
heuristic (any title-case token run) was measured against every real report
in the 9d run: 15-74 candidates per case, no separation between passing and
failing cases, dominated by model-authored markdown headings. Requiring a
digit or an all-caps token cuts that to a class fabricated version strings
and model names actually belong to (`AG2`, `v0.4`, `MemGPT-2` in; `OpenAI`
alone out) -- ordinary Title Case prose no longer qualifies on its own.

**D9e.7b -- a run-scoped nudge counter, not a per-model-call one.** A nudge
issued from `wrap_model_call` cannot be persisted in `request.messages` (the
overridden request is never stored, only the response `AIMessage` is), so
"no marker at all" would nudge on every model call up to
`ModelCallLimitMiddleware`'s own ceiling. `GroundingState.grounding_nudge_count`
uses the same `UntrackedValue`/`PrivateStateAttr` idiom
`ModelCallLimitState.run_model_call_count` already uses in the installed
langchain -- run-scoped, invisible to the model, capped at
`_MAX_NUDGES_PER_RUN`.

This module is INFRA (`MAY_IMPORT[INFRA] = {KERNEL, DOMAIN, INFRA}`,
`tests/test_layering.py`), so it imports the private helpers it needs
(`_run_tool_call_ids`, `_tool_results`, `_VERIFICATION_TOOLS`) straight from
`middleware.py` rather than duplicating them.
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from typing import Annotated, cast

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ContextT,
    ExtendedModelResponse,
    ModelRequest,
    ModelResponse,
    PrivateStateAttr,
    ResponseT,
)
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.channels.untracked_value import UntrackedValue
from langgraph.types import Command
from typing_extensions import NotRequired

from middleware import (
    _VERIFICATION_TOOLS,
    _run_tool_call_ids,
    _tool_results,
    record_superseded_model_call,
)

_FENCED_CODE_BLOCK = re.compile(r"```.*?```", re.DOTALL)
_HEADING_LINE = re.compile(r"^#{1,6}\s.*$", re.MULTILINE)
_INLINE_CODE_SPAN = re.compile(r"`([^`\n]+)`")

_VERSION_TOKEN = re.compile(r"\bv\d+(?:\.\d+)*\b", re.IGNORECASE)
_YEAR_TOKEN = re.compile(r"\b(?:19|20)\d{2}\b")
_CAPITALISED_RUN = re.compile(
    r"[A-Z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)*(?:\s[A-Z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)*){0,3}"
)

# Common function/verb words capitalise at a sentence's start for reasons
# that have nothing to do with being part of an entity's name -- "The AG2"
# would otherwise report "The" as part of the candidate, and the resulting
# string then fails to match "AG2" appearing on its own in a tool result,
# turning a *supported* claim into a false unsupported one. Trimmed from the
# ends of a run only, never from the interior (an entity's own name can
# legitimately contain one of these words, e.g. "Bank of America").
_RUN_STOPWORDS = frozenset(
    {
        "the",
        "this",
        "that",
        "these",
        "those",
        "a",
        "an",
        "and",
        "or",
        "but",
        "for",
        "with",
        "uses",
        "using",
        "used",
        "built",
        "exposed",
        "released",
        "published",
        "named",
        "called",
        "announced",
        "found",
        "shows",
        "show",
        "based",
        "compared",
        "comparing",
        "supports",
        "including",
        "such",
        "over",
        "throughout",
        "via",
        "through",
    }
)


def _trim_run_stopwords(run: str) -> str:
    """Strip leading/trailing tokens that are common function words, so a
    sentence-initial capital (e.g. "The" before "AG2") is never reported as
    part of the candidate."""
    tokens = run.split(" ")
    start = 0
    end = len(tokens)
    while start < end and tokens[start].lower() in _RUN_STOPWORDS:
        start += 1
    while end > start and tokens[end - 1].lower() in _RUN_STOPWORDS:
        end -= 1
    return " ".join(tokens[start:end])


def _token_has_digit(token: str) -> bool:
    return any(char.isdigit() for char in token)


def _token_is_allcaps(token: str) -> bool:
    """Every alphabetic character in `token` is uppercase, and there are at
    least two of them -- a single capital letter (e.g. the article "A") is
    too common to be a fabrication signal on its own."""
    letters = [char for char in token if char.isalpha()]
    return len(letters) >= 2 and all(char.isupper() for char in letters)


def _run_qualifies(run: str) -> bool:
    """A capitalised run counts only if one of its own tokens carries a
    digit or is itself all-caps -- ordinary Title Case prose has neither."""
    return any(
        _token_has_digit(token) or _token_is_allcaps(token) for token in run.split(" ")
    )


def extract_candidates(text: str) -> list[str]:
    """Candidate fabricated entities in `text` (D9e.7a).

    Fenced code blocks and markdown headings are skipped entirely; inline
    code spans are unwrapped rather than skipped, since a model puts
    fabricated version strings there too. Three candidate shapes, in order,
    de-duplicated while preserving first-seen order: an explicit `vN.N`
    version token, a 4-digit year, and a capitalised run of 1-4 tokens where
    at least one token carries a digit or is itself all-caps.

    Parameters
    ----------
    text : str
        A model's draft answer.

    Returns
    -------
    list of str
        Candidate strings, in the order they were found.
    """
    cleaned = _FENCED_CODE_BLOCK.sub(" ", text)
    cleaned = _HEADING_LINE.sub("", cleaned)
    cleaned = _INLINE_CODE_SPAN.sub(r"\1", cleaned)

    seen: set[str] = set()
    candidates: list[str] = []

    def add(candidate: str) -> None:
        if candidate not in seen:
            seen.add(candidate)
            candidates.append(candidate)

    for match in _VERSION_TOKEN.finditer(cleaned):
        add(match.group(0))
    for match in _YEAR_TOKEN.finditer(cleaned):
        add(match.group(0))
    for match in _CAPITALISED_RUN.finditer(cleaned):
        run = _trim_run_stopwords(match.group(0))
        if run and _run_qualifies(run):
            add(run)

    return candidates


class GroundingState(AgentState[ResponseT]):
    """State schema for `UnsupportedClaimMiddleware`."""

    grounding_nudge_count: NotRequired[Annotated[int, UntrackedValue, PrivateStateAttr]]


_MAX_NUDGES_PER_RUN = 2

_GROUNDING_NUDGE_TEMPLATE = (
    "This draft names at least one specific entity, version or year that "
    "does not appear in any web_search, read_url or knowledge_search result "
    "this turn: {candidates}. Verify it with one of those tools or remove "
    "it from the answer, then redraft."
)


def _verification_tool_context(messages: list[BaseMessage]) -> str:
    """Every result a research tool returned this run, concatenated -- what
    a candidate must appear in to count as supported."""
    parts: list[str] = []
    for tool_name in _VERIFICATION_TOOLS:
        call_ids = _run_tool_call_ids(messages, tool_name)
        parts.extend(_tool_results(messages, call_ids))
    return "\n".join(parts)


def _draft_text(response: ModelResponse[ResponseT]) -> str | None:
    """The response's own answer text, or `None` when this call is a tool
    call rather than a draft -- fabrication is a property of prose, not of
    a tool invocation."""
    for message in response.result:
        if isinstance(message, AIMessage) and not message.tool_calls:
            content = message.content
            if isinstance(content, str) and content:
                return content
    return None


class UnsupportedClaimMiddleware(
    AgentMiddleware[GroundingState[ResponseT], ContextT, ResponseT]
):
    """Nudges the Researcher to verify or drop a claim its own draft names
    but no tool result this run actually supports (D9e.7b).

    Fires only when the response carries no tool calls (a draft, not a
    search) and `extract_candidates` finds at least one candidate absent
    from every `web_search`/`read_url`/`knowledge_search` result seen this
    run. One retry per qualifying response, capped at `_MAX_NUDGES_PER_RUN`
    total for the run -- `tool_choice="any"` is deliberately not set, since
    the nudge asks for a redraft, not a tool call.

    Defines **both** `wrap_model_call` and `awrap_model_call` (D3.1b).

    `role`, default `"researcher"` -- the only role this middleware is ever
    attached to (`orchestrator.py`/`supervisor.py`'s `research_graph`),
    passed to `middleware.record_superseded_model_call` so a grounding
    nudge's discarded first response still gets its own `model.<role>` span
    (stage 9e, phase 3 R.2 finding, found while verifying this middleware's
    own retry: `TracingMiddleware`'s own span otherwise sees only the
    redrafted response).
    """

    state_schema = GroundingState  # type: ignore[assignment]

    def __init__(self, *, role: str = "researcher") -> None:
        super().__init__()
        self.role = role

    def wrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], ModelResponse[ResponseT]],
    ) -> ModelResponse[ResponseT] | ExtendedModelResponse[ResponseT]:
        response = handler(request)
        nudge = self._nudge_for(request, response)
        if nudge is None:
            return response
        record_superseded_model_call(self.role, request, response)
        retried = handler(self._retry_request(request, nudge))
        return self._extend(request, retried)

    async def awrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[
            [ModelRequest[ContextT]], Awaitable[ModelResponse[ResponseT]]
        ],
    ) -> ModelResponse[ResponseT] | ExtendedModelResponse[ResponseT]:
        response = await handler(request)
        nudge = self._nudge_for(request, response)
        if nudge is None:
            return response
        record_superseded_model_call(self.role, request, response)
        retried = await handler(self._retry_request(request, nudge))
        return self._extend(request, retried)

    def _nudge_for(
        self, request: ModelRequest[ContextT], response: ModelResponse[ResponseT]
    ) -> str | None:
        count = cast(int, request.state.get("grounding_nudge_count", 0))
        if count >= _MAX_NUDGES_PER_RUN:
            return None

        text = _draft_text(response)
        if text is None:
            return None
        candidates = extract_candidates(text)
        if not candidates:
            return None

        messages = cast("list[BaseMessage]", request.state["messages"])
        context = _verification_tool_context(messages).lower()
        unsupported = [c for c in candidates if c.lower() not in context]
        if not unsupported:
            return None
        return _GROUNDING_NUDGE_TEMPLATE.format(candidates=", ".join(unsupported))

    def _extend(
        self, request: ModelRequest[ContextT], retried: ModelResponse[ResponseT]
    ) -> ExtendedModelResponse[ResponseT]:
        count = cast(int, request.state.get("grounding_nudge_count", 0)) + 1
        return ExtendedModelResponse(
            model_response=retried,
            command=Command(update={"grounding_nudge_count": count}),
        )

    @staticmethod
    def _retry_request(
        request: ModelRequest[ContextT], nudge: str
    ) -> ModelRequest[ContextT]:
        retry_messages = [*request.messages, HumanMessage(content=nudge)]
        return request.override(messages=retry_messages)
