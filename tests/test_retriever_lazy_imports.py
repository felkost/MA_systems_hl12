"""Importing `retriever` must not pull in `sentence_transformers`.

The cross-encoder reranker is the one local Hugging Face model left in this
project's design (embeddings moved to OpenRouter). `sentence_transformers`
is a heavy, slow import;
if `retriever.py` imported it at module scope, every gate run -- including
one that never calls `knowledge_search` -- would pay that cost, and (per
hl10's own finding) it eagerly pulls in more through
`langchain_classic.retrievers`. This runs in a **subprocess**: a prior test
in the same pytest session may have already imported
`sentence_transformers` for an unrelated reason, which would make an
in-process `sys.modules` check pass by accident.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

_CHILD_SCRIPT = (
    "import sys\n"
    "import retriever\n"
    "assert 'sentence_transformers' not in sys.modules, sorted(sys.modules)\n"
    "print('ok')\n"
)


def test_importing_retriever_does_not_load_sentence_transformers() -> None:
    result = subprocess.run(
        [sys.executable, "-c", _CHILD_SCRIPT],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == "ok"
