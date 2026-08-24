"""A `DeepEvalBaseLLM` over OpenRouter's chat-completions endpoint.

This is the single place DeepEval learns about a model: every `GEval`,
`ToolCorrectnessMetric` and `AnswerRelevancyMetric` call in this project takes
an instance of `OpenRouterModel` as its `model=` argument.

Deliberately has no environment or `Settings` coupling of its own -- see
`CLAUDE.md`'s layer table, where `config.py` (kernel) does not exist until
stage 2 while this module is a stage-1 deliverable. The constructor takes
`api_key` and `model_name` explicitly; the caller (a test, or `evals/runner.py`
from stage 5 onward) is what reads `Settings` or the environment.

DeepEval never accrues cost for a custom (non-native) model --
`initialize_model` in the installed package marks any `DeepEvalBaseLLM`
subclass `using_native_model=False`, and DeepEval's own cost tracking only
fires on the native branch. This wrapper therefore records usage on every
call itself, in `self.last_usage`, rather than assuming DeepEval will. An
optional `usage_log` (stage 9a, `docs/specs/stage-9a.md` D9a.7) accumulates
every call's usage instead of overwriting it, for a caller that needs a
metric's real total cost when one `.measure()` makes more than one model
call internally.
"""

from __future__ import annotations

import json
from typing import Any, Type, TypeVar, cast

import httpx
from deepeval.models.base_model import DeepEvalBaseLLM
from pydantic import BaseModel

SchemaT = TypeVar("SchemaT", bound=BaseModel)

_DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterModel(DeepEvalBaseLLM):
    """One model, addressed by its OpenRouter id (e.g. `openai/gpt-4.1-mini`).

    `last_usage` is set after every `generate`/`a_generate` call to the
    provider's own `usage` object (`prompt_tokens`, `completion_tokens`,
    `total_tokens`) -- the only way this project's cost accounting sees a
    GEval-driven call, since DeepEval will not report it.
    """

    def __init__(
        self,
        model_name: str,
        api_key: str,
        base_url: str = _DEFAULT_BASE_URL,
        # 180s, raised from 120 at stage 9e phase 1b, itself raised from 60
        # at stage 9d (D9d.4). Each raise was made against a measured
        # timeout, never pre-emptively: 9a/9c lost metrics at 60s, and the
        # 9e phase-1b run lost `adversarial-direct-jailbreak`'s Answer
        # Relevancy to `httpx.ReadTimeout` at 120s -- with DeepEval's own
        # per-task budget (600s, `Settings.deepeval_per_task_timeout_seconds`)
        # nowhere near exhausted, which is what identified the client
        # timeout rather than the task budget as the binding limit.
        #
        # The ceiling on this value is the invariant D9e.1 names:
        # `PER_TASK >= n_sequential_judge_calls * httpx_timeout + slack`.
        # `AnswerRelevancyMetric` makes 3 sequential calls, so 3 * 180 =
        # 540s against a 600s budget leaves only 60s of slack -- which is
        # why the per-task default moves to 900s in the same change.
        timeout: float = 180.0,
        # httpx.MockTransport implements both BaseTransport and
        # AsyncBaseTransport (httpx/_transports/mock.py); the union below
        # is what lets one test double satisfy both httpx.Client and
        # httpx.AsyncClient below.
        transport: httpx.BaseTransport | httpx.AsyncBaseTransport | None = None,
        usage_log: list[dict[str, int]] | None = None,
        reasoning_effort: str | None = None,
    ) -> None:
        # `transport` is a constructor seam for tests -- passing a
        # `httpx.MockTransport` here is how the offline suite exercises this
        # class with no network call, instead of monkeypatching `httpx.Client`
        # globally (which mypy cannot type-check and which leaks across
        # tests if a teardown step is ever missed).
        self._model_name = model_name
        self._api_key = api_key
        self._base_url = base_url
        self._timeout = timeout
        self._transport: httpx.BaseTransport | httpx.AsyncBaseTransport | None = (
            transport
        )
        self.last_usage: dict[str, int] | None = None
        # `usage_log`, unlike `last_usage`, is never overwritten -- every
        # call appends. `last_usage` alone cannot report a metric's real
        # cost when one `.measure()` makes more than one model call
        # (`AnswerRelevancyMetric` makes three, stage 9a's own SDK
        # measurement, `docs/specs/stage-9a.md` N2): reading `last_usage`
        # after the fact only ever sees the last of them. A caller that
        # passes a shared list here (stage 9a's `e2e_judge_model` fixture)
        # gets every call's usage, in order, regardless of how many calls
        # one metric measurement made internally.
        self.usage_log: list[dict[str, int]] | None = usage_log
        # D9e.16: `None` (the default) keeps the payload byte-for-byte what
        # it always was -- no `reasoning` key at all. Measured this stage:
        # 89-94% of every judge call's cost is `gemini-2.5-pro`'s own
        # thinking, billed as completion tokens at $10/M; capping it is the
        # cheapest lever available, but a judge that thinks less may also
        # score differently, so this ships disabled until a probe compares
        # capped and uncapped scores on the same cases.
        self._reasoning_effort = reasoning_effort
        super().__init__(model=model_name)

    def load_model(self) -> "OpenRouterModel":
        return self

    def get_model_name(self) -> str:
        return self._model_name

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    def _payload(self, prompt: str, schema: Type[BaseModel] | None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self._model_name,
            "messages": [{"role": "user", "content": prompt}],
        }
        if schema is not None:
            # OpenRouter's structured-output contract is OpenAI's
            # `response_format: json_schema`. `strict: True` is what
            # `models.assert_structured_output_supported` (stage 2) will
            # check the provider actually honours, per model, at startup.
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema.__name__,
                    "schema": schema.model_json_schema(),
                    "strict": True,
                },
            }
        if self._reasoning_effort is not None:
            payload["reasoning"] = {"effort": self._reasoning_effort}
        return payload

    def _record_usage(self, body: dict[str, Any]) -> None:
        usage = body.get("usage") or {}
        self.last_usage = {
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        }
        if self.usage_log is not None:
            self.usage_log.append(dict(self.last_usage))

    def _extract_content(self, body: dict[str, Any]) -> str:
        return body["choices"][0]["message"]["content"]

    def generate(self, prompt: str, schema: Type[BaseModel] | None = None) -> str:
        # MockTransport implements both transport interfaces at once
        # (httpx/_transports/mock.py); the cast reflects what is actually
        # true at runtime for the one test double this project constructs,
        # not a blind assertion.
        sync_transport = cast("httpx.BaseTransport | None", self._transport)
        with httpx.Client(
            base_url=self._base_url, timeout=self._timeout, transport=sync_transport
        ) as client:
            response = client.post(
                "/chat/completions",
                headers=self._headers(),
                json=self._payload(prompt, schema),
            )
            response.raise_for_status()
            body = response.json()
        self._record_usage(body)
        return self._extract_content(body)

    async def a_generate(
        self, prompt: str, schema: Type[BaseModel] | None = None
    ) -> str:
        async_transport = cast("httpx.AsyncBaseTransport | None", self._transport)
        async with httpx.AsyncClient(
            base_url=self._base_url, timeout=self._timeout, transport=async_transport
        ) as client:
            response = await client.post(
                "/chat/completions",
                headers=self._headers(),
                json=self._payload(prompt, schema),
            )
            response.raise_for_status()
            body = response.json()
        self._record_usage(body)
        return self._extract_content(body)

    # The base class declares `generate_with_schema(self, *args, schema=None,
    # **kwargs)` -- open on purpose, since a schema-optional capability is
    # not part of every provider's contract (see insights.md, 2026-08-22).
    # This override narrows `prompt` to `str` and makes `schema` required,
    # which mypy reads as a Liskov violation even though every call site in
    # this project goes through DeepEval's own dispatch, never through the
    # base class's open signature directly.
    def generate_with_schema(  # type: ignore[override]
        self, prompt: str, schema: Type[SchemaT]
    ) -> SchemaT:
        """Returns a validated `schema` instance directly.

        DeepEval accepts either this or a raw string it JSON-parses itself
        (`generate_with_schema_and_extract` in the installed package); a
        pre-validated instance skips that second parse and fails loudly here
        instead of ambiguously inside DeepEval's own extraction path.
        """
        raw = self.generate(prompt, schema=schema)
        return schema.model_validate(json.loads(raw))

    async def a_generate_with_schema(  # type: ignore[override]
        self, prompt: str, schema: Type[SchemaT]
    ) -> SchemaT:
        raw = await self.a_generate(prompt, schema=schema)
        return schema.model_validate(json.loads(raw))
