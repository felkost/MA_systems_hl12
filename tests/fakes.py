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
