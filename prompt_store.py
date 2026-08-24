"""Fetches every agent system prompt from Langfuse Prompt Management by name
and label (stage 1, `docs/specs/stage-1.md`) -- the mechanism requirement 3
exists for: no prompt text lives in a `.py` file, so a prompt change is a
Langfuse action, never a code deploy.

**The fallback contract rides the real SDK's own, not a hand-rolled one**
(measured on the installed `langfuse==4.14.4`, `docs/specs/stage-1.md`
section 2): `Langfuse.get_prompt(..., fallback=text)` returns a client with
`is_fallback=True` on a fetch error when `text` is truthy, and raises when it
is not. `LangfusePromptStore.get` always passes the local snapshot's text
(or `None`) as `fallback`, so it never needs its own try/except-then-read-
file logic -- only a check of `is_fallback` afterwards.

**A silent fallback is the failure mode this module exists to prevent.**
`get` logs a `WARNING` naming the prompt whenever `is_fallback` is set, and
raises `PromptUnavailableError` -- a typed policy-class error, never a bare
string -- when even the snapshot has nothing.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Mapping, Protocol

logger = logging.getLogger("hl12.prompt_store")


class PromptUnavailableError(RuntimeError):
    """A prompt could be fetched from neither Langfuse nor the local
    snapshot. Typed so a caller (or a test) can distinguish this from any
    other `RuntimeError` -- CLAUDE.md's tool-error invariant, applied to a
    non-tool sink."""

    def __init__(self, name: str) -> None:
        super().__init__(
            f"prompt {name!r} is unavailable: Langfuse fetch failed and no "
            "local snapshot has a cached copy"
        )
        self.name = name


class PromptStore(Protocol):
    """What `supervisor.py`/`orchestrator.py` need from a prompt source --
    narrow enough that a test can satisfy it with `SnapshotPromptStore`
    alone, never touching the network."""

    def get(
        self,
        name: str,
        *,
        label: str,
        variables: Mapping[str, str] | None = None,
    ) -> str:
        """Return prompt `name`'s text at `label`, with `variables`
        compiled in (`{{var}}` substitution)."""
        ...


class SnapshotPromptStore:
    """A `PromptStore` backed entirely by an in-memory mapping -- the gate
    tier's double, and what `LangfusePromptStore` itself uses to read the
    on-disk snapshot file.

    `label` is accepted and ignored: a snapshot holds one text per name,
    since it exists to survive a single process's local cache, not to model
    Langfuse's own label/version history.
    """

    def __init__(self, snapshot: Mapping[str, str]) -> None:
        self._snapshot = dict(snapshot)

    def get(
        self,
        name: str,
        *,
        label: str,
        variables: Mapping[str, str] | None = None,
    ) -> str:
        try:
            text = self._snapshot[name]
        except KeyError:
            raise PromptUnavailableError(name) from None
        return _compile(text, variables)


class LangfusePromptStore:
    """Fetches from Langfuse Cloud, with the local snapshot as `fallback=`.

    Parameters
    ----------
    client : Langfuse
        Or any object exposing the same `get_prompt(name, *, label,
        cache_ttl_seconds, fallback, ...)` surface -- a test injects a fake
        rather than a real `Langfuse` instance.
    snapshot_path : Path or None
        Where the local fallback lives (`paths.prompt_snapshot_path()`
        normally). `None` disables persistence entirely -- every fetch
        failure with no in-process cache raises immediately, which a
        deliberately network-only test uses to prove the raise path without
        also creating a file.
    cache_ttl_seconds : int, default 300
        Passed straight through to `Langfuse.get_prompt` -- the SDK's own
        client-side cache TTL (its own default is 60s; this project widens
        it slightly since a prompt changing mid-session is rare and every
        turn re-fetching adds latency for no benefit).
    """

    def __init__(
        self,
        client: Any,
        *,
        snapshot_path: Path | None,
        cache_ttl_seconds: int = 300,
    ) -> None:
        self._client = client
        self._snapshot_path = snapshot_path
        self._cache_ttl_seconds = cache_ttl_seconds

    def get(
        self,
        name: str,
        *,
        label: str,
        variables: Mapping[str, str] | None = None,
    ) -> str:
        fallback_text = self._read_snapshot().get(name)
        try:
            result = self._client.get_prompt(
                name,
                label=label,
                cache_ttl_seconds=self._cache_ttl_seconds,
                fallback=fallback_text,
            )
        except Exception as exc:
            raise PromptUnavailableError(name) from exc

        if getattr(result, "is_fallback", False):
            logger.warning(
                "prompt %r served from the local snapshot fallback -- "
                "Langfuse fetch failed and no fresher copy was cached",
                name,
            )
        else:
            self._write_snapshot(name, result.prompt)

        # Delegates to the client's own `.compile()` -- the real SDK's
        # `TemplateParser.compile_template` (measured permissive both ways:
        # docs/specs/stage-1.md section 2) -- rather than reimplementing
        # variable substitution a second time here.
        return result.compile(**variables) if variables else result.prompt

    def _read_snapshot(self) -> dict[str, str]:
        if self._snapshot_path is None or not self._snapshot_path.exists():
            return {}
        return json.loads(self._snapshot_path.read_text(encoding="utf-8"))

    def _write_snapshot(self, name: str, text: str) -> None:
        if self._snapshot_path is None:
            return
        snapshot = self._read_snapshot()
        snapshot[name] = text
        self._snapshot_path.write_text(
            json.dumps(snapshot, indent=2, ensure_ascii=False), encoding="utf-8"
        )


def _compile(text: str, variables: Mapping[str, str] | None) -> str:
    """`SnapshotPromptStore`'s own compile -- there is no `PromptClient` to
    delegate to when the source is a plain in-memory mapping. Same
    substitution shape as the real SDK's `TemplateParser.compile_template`:
    an unmatched `{{var}}` is left literal."""
    if not variables:
        return text
    for key, value in variables.items():
        text = text.replace("{{" + key + "}}", value)
    return text
