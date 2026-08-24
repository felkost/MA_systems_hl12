"""`retriever.py` checks `index/manifest.json`, it does not trust it.

An index built with one embedding model returns confident nonsense under a
different one -- nothing about a vector-dimension or provider mismatch raises
on its own. `retriever.verify_manifest` is the one place that compares what
the index says it was built with against what `Settings` currently resolves
to, and refuses instead of degrading silently (`docs/specs/stage-2.md`,
"`ingest.py` -- manifest is checked, not trusted").

Offline: no embedding call, no Chroma collection is opened. Only the JSON
comparison is under test.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import SecretStr

from config import Settings
from retriever import IndexMismatchError, verify_manifest


def _settings(index_dir: Path, data_dir: Path) -> Settings:
    return Settings(
        openrouter_api_key=SecretStr("test-key"),
        index_dir=str(index_dir),
        data_dir=str(data_dir),
    )


def _write_manifest(index_dir: Path, **fields: object) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)
    base: dict[str, object] = {
        "collection_name": "knowledge_base",
        "chunk_size": 500,
        "chunk_overlap": 100,
        "embedding_fingerprint": {
            "provider": "openrouter",
            "model": "openai/text-embedding-3-small",
            "dimensions": None,
        },
        "sources": {},
        "files": 0,
        "chunks": 1,
    }
    base.update(fields)
    (index_dir / "manifest.json").write_text(json.dumps(base), encoding="utf-8")


def test_manifest_mismatch_raises_on_wrong_embedding_model(tmp_path: Path) -> None:
    index_dir = tmp_path / "index"
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_manifest(
        index_dir,
        embedding_fingerprint={
            "provider": "openrouter",
            "model": "openai/text-embedding-ada-002",  # wrong model on purpose
            "dimensions": None,
        },
    )
    settings = _settings(index_dir, data_dir)

    with pytest.raises(IndexMismatchError, match="embedding_fingerprint"):
        verify_manifest(settings)


def test_manifest_matching_the_current_settings_does_not_raise(
    tmp_path: Path,
) -> None:
    index_dir = tmp_path / "index"
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_manifest(index_dir)
    settings = _settings(index_dir, data_dir)

    verify_manifest(settings)  # must not raise


def test_missing_manifest_raises_file_not_found(tmp_path: Path) -> None:
    index_dir = tmp_path / "index"
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    settings = _settings(index_dir, data_dir)

    with pytest.raises(FileNotFoundError):
        verify_manifest(settings)
