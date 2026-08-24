"""`_build_reranker`'s returned compressor shares one cross-encoder
instance across every `knowledge_search` call in a process
(`get_retriever`'s own `lru_cache`, `retriever.py`). A live run crashed with
a Windows access violation inside `torch`/`transformers` when two of
LangGraph's own concurrently-executed tool calls scored on that shared
model at the same time. `retriever.py`'s existing `_retriever_lock` -- built
for `get_retriever`'s one-time construction -- now also wraps the scoring
call itself. This pins that the lock actually serializes concurrent
scoring, without needing a real model or a real crash to reproduce it.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

from langchain_community.cross_encoders import BaseCrossEncoder
from langchain_core.documents import Document

from retriever import _build_reranker


class _RecordingFakeCrossEncoder(BaseCrossEncoder):
    """Records how many `.score()` calls were in flight at once.

    `CrossEncoderReranker` is a pydantic model whose `model` field is typed
    `BaseCrossEncoder` and validated with `isinstance`, not just duck-typed
    -- a plain object with a `.score()` method fails construction.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active = 0
        self.max_concurrent = 0

    def score(self, pairs: list[tuple[str, str]]) -> list[float]:
        with self._lock:
            self._active += 1
            self.max_concurrent = max(self.max_concurrent, self._active)
        time.sleep(0.05)
        with self._lock:
            self._active -= 1
        return [0.5 for _ in pairs]


def test_reranker_serializes_concurrent_scoring_calls() -> None:
    fake_model = _RecordingFakeCrossEncoder()
    reranker = _build_reranker(fake_model, top_n=1)
    documents = [Document(page_content="a"), Document(page_content="b")]

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [
            pool.submit(reranker.compress_documents, documents, "query")
            for _ in range(4)
        ]
        for future in futures:
            future.result()

    assert fake_model.max_concurrent == 1
