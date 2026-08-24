"""Shared offline test doubles for stage 3's agent and middleware tests.

`FakeToolCallingModel` is the same shape both donor projects use to test
agent factories without a network call: it accepts `bind_tools` (called
once by `create_agent` to attach the real tools plus, for a
`ProviderStrategy` response format, a synthetic structured-output tool) and
returns itself unchanged, so a test scripts the model's *output*, never its
binding.

Not named `conftest.py`: mypy resolves the same file as both `conftest` (its
own root-relative discovery) and `tests.conftest` (an explicit import from a
test module) with no `tests/__init__.py`, which it refuses as "found twice
under different module names". A plain module has no such ambiguity.
"""

from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

import prompt_store


class FakeToolCallingModel(BaseChatModel):
    """Cycles through scripted `AIMessage` responses, ignoring bound tools."""

    responses: list[BaseMessage]
    index: int = 0

    def bind_tools(
        self,
        tools: Any,
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> "FakeToolCallingModel":
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        response = self.responses[self.index]
        self.index = min(self.index + 1, len(self.responses) - 1)
        return ChatResult(generations=[ChatGeneration(message=response)])

    @property
    def _llm_type(self) -> str:
        return "fake-tool-calling-model"


class _FakeTextPromptClient:
    """Mirrors the real `langfuse.model.TextPromptClient` surface
    `prompt_store.py` actually calls: `.prompt` (raw text), `.compile(**kw)`
    (naive `{{var}}` substitution, permissive like the real
    `TemplateParser.compile_template` -- an unmatched variable is left
    literal), and `.is_fallback` (stage-1 SDK finding, `docs/specs/stage-1.md`
    section 2: the real SDK sets this on a fallback-served client instead of
    raising, whenever a `fallback=` value was given)."""

    def __init__(self, prompt: str, *, is_fallback: bool = False) -> None:
        self.prompt = prompt
        self.is_fallback = is_fallback

    def compile(self, **kwargs: str) -> str:
        text = self.prompt
        for key, value in kwargs.items():
            text = text.replace("{{" + key + "}}", value)
        return text


class FakeLangfuse:
    """A `Langfuse` double holding prompts in memory, matching
    `Langfuse.get_prompt`'s measured signature and fallback contract
    (`docs/specs/stage-1.md`, section 2)."""

    def __init__(self, prompts: dict[str, str]) -> None:
        self._prompts = dict(prompts)
        self.get_prompt_calls: list[tuple[str, str | None]] = []

    def get_prompt(
        self,
        name: str,
        *,
        label: str | None = None,
        cache_ttl_seconds: int | None = None,
        fallback: str | None = None,
        **_: Any,
    ) -> _FakeTextPromptClient:
        self.get_prompt_calls.append((name, label))
        if name in self._prompts:
            return _FakeTextPromptClient(self._prompts[name])
        if fallback:
            return _FakeTextPromptClient(fallback, is_fallback=True)
        raise RuntimeError(f"no prompt registered for {name!r} and no fallback")


class RaisingLangfuse:
    """A `Langfuse` double whose fetch always fails -- simulating the API
    being unreachable, honouring the same `fallback=` contract as the real
    SDK (return an `is_fallback=True` client if `fallback` is truthy, else
    raise)."""

    def get_prompt(
        self,
        name: str,
        *,
        label: str | None = None,
        cache_ttl_seconds: int | None = None,
        fallback: str | None = None,
        **_: Any,
    ) -> _FakeTextPromptClient:
        if fallback:
            return _FakeTextPromptClient(fallback, is_fallback=True)
        raise RuntimeError(f"Langfuse unreachable while fetching {name!r}")


def fake_prompt_store() -> prompt_store.SnapshotPromptStore:
    """A `SnapshotPromptStore` seeded with short placeholder text for all six
    hl12 prompt names -- the offline gate's stand-in for
    `prompt_store.LangfusePromptStore` wherever a test builds a real
    Supervisor/orchestrator graph and needs *some* `system_prompt` text, not
    a specific one. Tests that assert on prompt *content* build their own
    scripted `FakeToolCallingModel` responses instead; this only has to be
    non-empty and importable without a network call.
    """
    return prompt_store.SnapshotPromptStore(
        {
            "hl12-planner": "You are the Planner.",
            "hl12-researcher": "You are the Researcher.",
            "hl12-critic": "You are the Critic. Today is {{today}}.",
            "hl12-supervisor": "You are the Supervisor.",
            "hl12-composer": "You are the report composer.",
            "hl12-critic-verification": "Verify at least one claim.",
        }
    )
