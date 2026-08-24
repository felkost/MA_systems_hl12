"""Shared offline test doubles for the agent and middleware test suite.

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
from langfuse.api import NotFoundError

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
    literal), and `.is_fallback` (a measured SDK behaviour: the real SDK
    sets this on a fallback-served client instead of raising, whenever a
    `fallback=` value was given)."""

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
    `Langfuse.get_prompt`'s measured signature and fallback contract."""

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


class FakeLangfuseDatasets:
    """A `Langfuse` double covering only the Datasets surface
    `evals.langfuse_dataset.sync_golden_dataset` calls -- `get_dataset`/
    `create_dataset`/`create_dataset_item` -- matching the real SDK's
    measured signatures and its `NotFoundError`-on-missing-dataset contract.
    `NotFoundError` is the real
    `langfuse.api.NotFoundError`, not a stand-in, so a test proves the
    production code's `except NotFoundError` clause against the exact type
    it will see live."""

    def __init__(self) -> None:
        self._datasets: dict[str, dict[str, dict[str, Any]]] = {}
        self.create_dataset_calls: list[str] = []

    def get_dataset(self, name: str) -> dict[str, Any]:
        if name not in self._datasets:
            raise NotFoundError(body=f"dataset {name!r} not found")
        return {"name": name, "items": dict(self._datasets[name])}

    def create_dataset(self, *, name: str, **_: Any) -> dict[str, Any]:
        self.create_dataset_calls.append(name)
        self._datasets.setdefault(name, {})
        return {"name": name}

    def create_dataset_item(
        self,
        *,
        dataset_name: str,
        input: Any = None,
        expected_output: Any = None,
        metadata: Any = None,
        id: str | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        item_id = id or f"generated-{len(self._datasets.get(dataset_name, {}))}"
        item = {
            "id": item_id,
            "input": input,
            "expected_output": expected_output,
            "metadata": metadata,
        }
        self._datasets.setdefault(dataset_name, {})[item_id] = item
        return item


class RaisingLangfuseDatasets:
    """A `Langfuse` double whose `get_dataset` fails with something other
    than "not found" -- simulating a bad/expired API key (401/403) so a
    test can prove `sync_golden_dataset` does not misread an auth failure
    as "dataset missing, create it"."""

    def get_dataset(self, name: str) -> dict[str, Any]:
        raise RuntimeError(f"401 Unauthorized while fetching dataset {name!r}")

    def create_dataset(self, *, name: str, **_: Any) -> dict[str, Any]:
        raise AssertionError(
            "create_dataset must never be called after a non-NotFoundError "
            "failure from get_dataset"
        )

    def create_dataset_item(self, **_: Any) -> dict[str, Any]:
        raise AssertionError(
            "create_dataset_item must never be called after get_dataset "
            "fails with something other than NotFoundError"
        )


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
