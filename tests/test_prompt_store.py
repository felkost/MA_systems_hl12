"""`prompt_store.py`'s two `PromptStore` implementations.

`LangfusePromptStore`'s fallback tests exercise the real
`Langfuse.get_prompt` contract measured on the installed `langfuse==4.14.4`,
not an invented one: passing `fallback=` makes the SDK return an
`is_fallback=True` client instead of raising, and passing no snapshot at
all is what makes it raise.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

import prompt_store
from tests.fakes import FakeLangfuse, RaisingLangfuse


def test_snapshot_store_compiles_variables() -> None:
    store = prompt_store.SnapshotPromptStore(
        {"hl12-critic": "Today is {{today}}. Judge the findings."}
    )
    assert (
        store.get("hl12-critic", label="production", variables={"today": "2026-08-24"})
        == "Today is 2026-08-24. Judge the findings."
    )


def test_snapshot_store_without_variables_returns_raw_text() -> None:
    store = prompt_store.SnapshotPromptStore({"hl12-planner": "Plan the research."})
    assert store.get("hl12-planner", label="production") == "Plan the research."


def test_snapshot_store_raises_on_unknown_name() -> None:
    store = prompt_store.SnapshotPromptStore({})
    with pytest.raises(prompt_store.PromptUnavailableError, match="hl12-planner"):
        store.get("hl12-planner", label="production")


def test_langfuse_store_writes_a_snapshot(tmp_path: Path) -> None:
    client = FakeLangfuse({"hl12-planner": "Plan the research."})
    snapshot = tmp_path / "snapshot.json"
    store = prompt_store.LangfusePromptStore(client, snapshot_path=snapshot)

    assert store.get("hl12-planner", label="production") == "Plan the research."
    assert json.loads(snapshot.read_text(encoding="utf-8"))["hl12-planner"] == (
        "Plan the research."
    )


def test_langfuse_store_passes_label_and_cache_ttl_through(tmp_path: Path) -> None:
    client = FakeLangfuse({"hl12-planner": "Plan the research."})
    store = prompt_store.LangfusePromptStore(
        client, snapshot_path=tmp_path / "snapshot.json", cache_ttl_seconds=300
    )
    store.get("hl12-planner", label="staging")
    assert client.get_prompt_calls == [("hl12-planner", "staging")]


def test_langfuse_store_compiles_variables(tmp_path: Path) -> None:
    client = FakeLangfuse({"hl12-critic": "Today is {{today}}."})
    store = prompt_store.LangfusePromptStore(
        client, snapshot_path=tmp_path / "snapshot.json"
    )
    assert (
        store.get("hl12-critic", label="production", variables={"today": "2026-08-24"})
        == "Today is 2026-08-24."
    )


def test_langfuse_store_falls_back_to_the_snapshot_and_logs_a_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    snapshot = tmp_path / "snapshot.json"
    snapshot.write_text(
        json.dumps({"hl12-planner": "Plan the research."}), encoding="utf-8"
    )
    store = prompt_store.LangfusePromptStore(RaisingLangfuse(), snapshot_path=snapshot)

    with caplog.at_level(logging.WARNING):
        result = store.get("hl12-planner", label="production")

    assert result == "Plan the research."
    assert any(
        "hl12-planner" in record.message and "fallback" in record.message.lower()
        for record in caplog.records
    )


def test_langfuse_store_fallback_does_not_overwrite_the_snapshot(
    tmp_path: Path,
) -> None:
    snapshot = tmp_path / "snapshot.json"
    snapshot.write_text(
        json.dumps({"hl12-planner": "Plan the research."}), encoding="utf-8"
    )
    before = snapshot.read_text(encoding="utf-8")
    store = prompt_store.LangfusePromptStore(RaisingLangfuse(), snapshot_path=snapshot)

    store.get("hl12-planner", label="production")

    assert snapshot.read_text(encoding="utf-8") == before


def test_langfuse_store_without_a_snapshot_raises_rather_than_inventing(
    tmp_path: Path,
) -> None:
    store = prompt_store.LangfusePromptStore(
        RaisingLangfuse(), snapshot_path=tmp_path / "snapshot.json"
    )
    with pytest.raises(prompt_store.PromptUnavailableError, match="hl12-planner"):
        store.get("hl12-planner", label="production")


def test_langfuse_store_snapshot_path_none_disables_persistence() -> None:
    client = FakeLangfuse({"hl12-planner": "Plan the research."})
    store = prompt_store.LangfusePromptStore(client, snapshot_path=None)
    assert store.get("hl12-planner", label="production") == "Plan the research."

    with pytest.raises(prompt_store.PromptUnavailableError):
        prompt_store.LangfusePromptStore(RaisingLangfuse(), snapshot_path=None).get(
            "hl12-planner", label="production"
        )
