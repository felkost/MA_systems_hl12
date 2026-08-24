"""Turn a resolved role into a real LangChain client.

One provider for chat models -- OpenRouter -- and `Settings.resolved_model`
already does the only real logic (which override wins). This module is the
one place that turns a resolved model id into a client, and the one place
that knows OpenRouter's base URL.

`langchain_huggingface` is imported inside `build_embeddings`, not at module
scope: importing it eagerly pulls in `sentence_transformers`, the same cost
`retriever.py` defers for the cross-encoder. `EMBEDDING_PROVIDER` stays
`"openrouter"` by default (the corrected stage-2 decision -- see
`docs/specs/stage-2.md`, "The embedding decision"); `"local"` is the
documented offline escape hatch.
"""

from __future__ import annotations

from typing import Any

import httpx
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

import paths
from config import Settings

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# model id -> (prompt $/token, completion $/token). Mirrors
# docs/model-prices-2026-08-22.md exactly (docs/specs/stage-5.md, D5.2) --
# that file is gitignored, so the literal is reproduced here rather than
# merely cited. Lives in models.py, not observability.py: middleware.py
# (infra) needs compute_cost for TracingMiddleware's per-model-call span
# attributes, and infra may not import an obs-layer module
# (tests/test_layering.py's MAY_IMPORT[INFRA] has no OBS entry) -- a gap
# the stage-5 spec's own detailed design missed until implementation.
# observability.py re-exports both names so `observability.PRICE_TABLE`/
# `observability.compute_cost` still work for its own callers.
PRICE_TABLE: dict[str, tuple[float, float]] = {
    "openai/gpt-4.1-mini": (0.0000004, 0.0000016),
    "google/gemini-2.5-pro": (0.00000125, 0.00001),
    "openai/text-embedding-3-small": (0.00000002, 0.0),
}


def compute_cost(model: str, input_tokens: int, output_tokens: int) -> float | None:
    """Cost in USD from token counts against `PRICE_TABLE`.

    Returns
    -------
    float or None
        `None` when `model` has no `PRICE_TABLE` entry -- never a partial
        sum. A partial sum distorts more than an explicit "we don't know"
        (the plan's own rule for this exact situation).
    """
    prices = PRICE_TABLE.get(model)
    if prices is None:
        return None
    prompt_price, completion_price = prices
    return input_tokens * prompt_price + output_tokens * completion_price


def compute_cost_or_raise(model: str, input_tokens: int, output_tokens: int) -> float:
    """`compute_cost`, but an unpriced model raises instead of returning
    `None` (stage 9e, D9e.17).

    `or 0.0` on `compute_cost`'s own `None` return silently records an
    unpriced model's entire cost as $0.00 -- exactly the "unmeasured
    component presented as a total" this project forbids. This is the
    version a caller reporting a real total (never a caller merely checking
    whether a price exists) should call.

    Raises
    ------
    ValueError
        `model` has no `PRICE_TABLE` entry -- names the model and this file.
    """
    cost = compute_cost(model, input_tokens, output_tokens)
    if cost is None:
        raise ValueError(
            f"model {model!r} has no PRICE_TABLE entry in models.py -- add "
            "its prompt/completion price before using it as a judge or "
            "agent model, rather than letting its cost read as $0.00"
        )
    return cost


# The roles whose caller builds a strict structured-output request:
# "planner"/"critic" via ProviderStrategy(..., strict=True) (agents/planner.py,
# agents/critic.py, stage 3) -- hardcoded rather than introspected from
# agents/*.py, since this module sits below agents/ in the layer table and
# must not import upward. "judge" joins the same way (stage 7+): a judge
# model that cannot honour strict structured output must fail at startup,
# not mid-dataset-run.
_STRUCTURED_OUTPUT_ROLES = ("planner", "critic", "judge")


def build_chat_model(settings: Settings, role: str) -> BaseChatModel:
    """Build the chat model `role` resolves to.

    Parameters
    ----------
    settings : Settings
    role : str
        One of `Settings.ROLES`.

    Returns
    -------
    BaseChatModel
        `ChatOpenAI` pointed at OpenRouter -- OpenRouter is OpenAI-API-
        compatible and differs only in `base_url` and the key, so no second
        chat-model class exists. `temperature` is passed only when set: some
        models routed through OpenRouter do not accept it.
    """
    model = settings.resolved_model(role)
    kwargs: dict[str, Any] = {
        "model": model,
        "api_key": settings.openrouter_api_key.get_secret_value(),
        "base_url": OPENROUTER_BASE_URL,
    }
    if settings.temperature is not None:
        kwargs["temperature"] = settings.temperature
    return ChatOpenAI(**kwargs)


def build_embeddings(settings: Settings) -> Embeddings:
    """Build the embedder `settings.embedding_provider` selects.

    Returns
    -------
    Embeddings
        `OpenAIEmbeddings` pointed at OpenRouter for `"openrouter"`
        (`POST /api/v1/embeddings`, verified live -- see
        `docs/specs/stage-2.md`); otherwise `HuggingFaceEmbeddings`
        (`langchain_huggingface`), cached under `settings.model_cache_dir`.
    """
    if settings.embedding_provider == "local":
        from langchain_huggingface import HuggingFaceEmbeddings

        cache = paths.resolve(settings.model_cache_dir)
        return HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            cache_folder=str(cache / "huggingface"),
        )

    kwargs: dict[str, Any] = {
        "model": settings.embedding_model,
        "api_key": settings.openrouter_api_key.get_secret_value(),
        "base_url": OPENROUTER_BASE_URL,
    }
    if settings.embedding_dimensions is not None:
        kwargs["dimensions"] = settings.embedding_dimensions
    return OpenAIEmbeddings(**kwargs)


def embedding_fingerprint(settings: Settings) -> dict[str, Any]:
    """{"provider", "model", "dimensions"} -- written into `manifest.json` by
    `ingest.py` and recomputed by `retriever.py` to compare."""
    return {
        "provider": settings.embedding_provider,
        "model": settings.embedding_model,
        "dimensions": settings.embedding_dimensions,
    }


class UnsupportedStructuredOutputError(Exception):
    """`role` needs strict structured output but the resolved OpenRouter
    model does not report support for it -- named explicitly, never a silent
    downgrade to best-effort parsing."""


async def assert_structured_output_supported(settings: Settings, role: str) -> None:
    """Refuse if `role` cannot actually get the structured output its agent
    requires from the model it resolves to.

    No-op outside `{"planner", "critic", "judge"}` -- the only roles whose
    caller builds a strict structured-output request. Fetches the live
    OpenRouter model catalog once and refuses, naming the model, if it is
    absent or does not report `"structured_outputs"` support.

    Raises
    ------
    UnsupportedStructuredOutputError
    """
    if role not in _STRUCTURED_OUTPUT_ROLES:
        return
    model = (
        settings.judge_model_name if role == "judge" else settings.resolved_model(role)
    )
    if model is None:
        raise UnsupportedStructuredOutputError(
            f"role {role!r} has no model configured -- set JUDGE_MODEL_NAME"
        )

    catalog = await _fetch_openrouter_models(settings)
    entry = catalog.get(model)
    if entry is None:
        raise UnsupportedStructuredOutputError(
            f"role {role!r} resolves to OpenRouter model {model!r}, which "
            f"does not appear in OpenRouter's model listing -- check "
            f"{role.upper()}_MODEL_NAME"
        )
    if not _supports_structured_output(entry):
        raise UnsupportedStructuredOutputError(
            f"role {role!r} resolves to OpenRouter model {model!r}, which "
            f"does not report support for strict structured output -- pick "
            f"a different {role.upper()}_MODEL_NAME"
        )


async def _fetch_openrouter_models(settings: Settings) -> dict[str, dict[str, Any]]:
    """`{model_id: entry}` for OpenRouter's full catalog -- one request, no
    auth required for the listing itself."""
    headers = {
        "Authorization": f"Bearer {settings.openrouter_api_key.get_secret_value()}"
    }
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(f"{OPENROUTER_BASE_URL}/models", headers=headers)
        response.raise_for_status()
    return {entry["id"]: entry for entry in response.json()["data"]}


def _supports_structured_output(entry: dict[str, Any]) -> bool:
    """`"structured_outputs"` only -- looser than `"response_format"`, which
    some models report without honouring `strict=True` (verified live at
    stage 1 against the full catalog)."""
    return "structured_outputs" in entry.get("supported_parameters", [])
