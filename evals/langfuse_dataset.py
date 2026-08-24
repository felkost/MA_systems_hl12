"""Idempotently uploads `tests/golden_dataset.json` into Langfuse Datasets --
infrastructure for the LLM-as-a-Judge evaluators to be validated against
known cases, ahead of scoring live traffic.

**Idempotency rides `create_dataset_item`'s own upsert-by-id contract**
(measured on the installed `langfuse==4.14.4`: `Langfuse.create_dataset_item`
resolves `item_id = id if id is not None else str(uuid.uuid4())`, then always
calls `self.api.dataset_items.create(..., id=item_id)` -- passing the same
id twice updates the existing item rather than creating a second one).
`_item_id` derives that id deterministically from a golden case's own `id`
field, so re-running this module against the same `tests/golden_dataset.json`
upserts in place. **This makes a case's `id` load-bearing**: editing it
(rather than the case's content) creates a new dataset item alongside the
old one instead of updating it, since the two ids no longer match.

**Dataset existence is checked explicitly, never assumed idempotent by
name.** `Langfuse.create_dataset`'s own docstring and source carry no
existence check or upsert note, unlike `create_dataset_item`. A missing
dataset's `get_dataset` call raises `langfuse.api.NotFoundError` --
measured against the installed SDK's `raw_client.py`, a *sibling* of the
generic `langfuse.api.Error` under the shared `ApiError` base, not its
subclass. Catching only `NotFoundError` (rather than `Exception` broadly)
is deliberate: a bad or expired API key raises `UnauthorizedError`/
`AccessDeniedError`, and treating those the same as "dataset missing" would
silently mask an auth failure behind a confusing second error from
`create_dataset`.
"""

from __future__ import annotations

import json
from typing import Any

from langfuse import Langfuse
from langfuse.api import NotFoundError

import paths
from config import Settings


class DatasetSyncError(RuntimeError):
    """A golden case failed to upload to Langfuse -- typed so a caller (or a
    test) can distinguish this from any other `RuntimeError`, the same class
    of exception `prompt_store.PromptUnavailableError` already is for a
    non-tool sink."""

    def __init__(self, case_id: str, cause: Exception) -> None:
        super().__init__(f"golden case {case_id!r} failed to sync: {cause}")
        self.case_id = case_id


_METADATA_EXCLUDED_KEYS = frozenset({"input", "expected_output"})


def _item_id(case_id: str) -> str:
    """The dataset item id one golden case's own `id` deterministically
    maps to -- what makes re-running `sync_golden_dataset` idempotent
    (module docstring)."""
    return f"hl12-golden-{case_id}"


def sync_golden_dataset(
    client: Any,  # Langfuse, or a fake exposing the same narrow surface
    *,
    name: str,
    cases: list[dict[str, Any]],
) -> list[str]:
    """Idempotently upload `cases` (`tests/golden_dataset.json`'s own rows)
    into a Langfuse dataset named `name`.

    Parameters
    ----------
    client : Langfuse
        Or any object exposing `get_dataset`/`create_dataset`/
        `create_dataset_item` with the same signatures.
    name : str
        The Langfuse dataset name (`Settings.langfuse_golden_dataset_name`
        normally).
    cases : list of dict
        Rows shaped like `tests/golden_dataset.json`'s own JSON: `id`,
        `input`, `expected_output` are required; every other key
        (`category`, `rationale`, `expects`, `fixture`,
        `needs_poisoned_index`, `freshness_cutoff`) is carried through into
        the dataset item's `metadata` as-is.

    Returns
    -------
    list of str
        The dataset item ids written, in `cases` order.

    Raises
    ------
    DatasetSyncError
        A case failed to upload.
    """
    try:
        client.get_dataset(name)
    except NotFoundError:
        client.create_dataset(name=name)

    item_ids: list[str] = []
    for case in cases:
        item_id = _item_id(case["id"])
        metadata = {
            key: value
            for key, value in case.items()
            if key not in _METADATA_EXCLUDED_KEYS
        }
        try:
            client.create_dataset_item(
                dataset_name=name,
                input=case["input"],
                expected_output=case["expected_output"],
                metadata=metadata,
                id=item_id,
            )
        except Exception as exc:
            raise DatasetSyncError(case["id"], exc) from exc
        item_ids.append(item_id)
    return item_ids


def main() -> list[str]:
    """Build a real `Langfuse` client from `config.load_settings()`, load
    `tests/golden_dataset.json`, and sync it -- the command the author runs
    by hand once Langfuse credentials exist (`ingest.py`'s standalone-script
    shape), not wired into `main.py`'s REPL startup: dataset sync is an
    operator action taken once per dataset change, not a per-turn cost.
    """
    from config import load_settings

    settings: Settings = load_settings()
    if settings.langfuse_public_key is None or settings.langfuse_secret_key is None:
        raise RuntimeError(
            "LANGFUSE_PUBLIC_KEY/LANGFUSE_SECRET_KEY are not set -- dataset "
            "sync requires them regardless of TRACING_ENABLED"
        )
    client = Langfuse(
        public_key=settings.langfuse_public_key.get_secret_value(),
        secret_key=settings.langfuse_secret_key.get_secret_value(),
        host=settings.langfuse_base_url,
        # Fixed false, not settings.tracing_enabled: this script only calls
        # the Datasets REST endpoints and never opens a span, so nothing
        # should ever be exported through it regardless of the local
        # tracing configuration.
        tracing_enabled=False,
    )
    cases = json.loads(
        (paths.PROJECT_ROOT / "tests" / "golden_dataset.json").read_text(
            encoding="utf-8"
        )
    )
    ids = sync_golden_dataset(
        client, name=settings.langfuse_golden_dataset_name, cases=cases
    )
    dataset_name = settings.langfuse_golden_dataset_name
    print(f"synced {len(ids)} items into dataset {dataset_name!r}:")
    for item_id in ids:
        print(f"  {item_id}")
    return ids


if __name__ == "__main__":
    main()
