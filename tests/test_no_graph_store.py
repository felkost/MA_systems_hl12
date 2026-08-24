"""No tracked Python module actually reintroduces a graph store.

CLAUDE.md's Forbidden list bans a graph store outright (see "Three
deliberate removals" -- a graph store is a second source of truth this
assignment does not measure).

Checked with `ast`, not a text search: a naive substring search on
`"graph_search"`/`"neo4j"` also matches every *legitimate* mention --
CLAUDE.md and `insights.md` discussing the removal as history, this
module's own docstring saying "minus `graph_search`", and
`test_tools_allowlists.py`'s own `assert "graph_search" not in names`
checks that graph_search stays absent. What actually indicates
reintroduction is narrower: an `import neo4j` / `from neo4j import ...`, or
a function actually named `graph_search`.
"""

from __future__ import annotations

import ast
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
THIS_FILE = Path(__file__).resolve()


def _tracked_python_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "--", "*.py"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return [PROJECT_ROOT / line for line in result.stdout.splitlines() if line]


def _reintroduces_graph_store(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    hits: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            hits.extend(
                f"import {alias.name}"
                for alias in node.names
                if alias.name.split(".")[0] == "neo4j"
            )
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.split(".")[0] == "neo4j":
                hits.append(f"from {node.module} import ...")
        elif isinstance(node, ast.FunctionDef) and node.name == "graph_search":
            hits.append(f"def graph_search (line {node.lineno})")
    return hits


def test_no_tracked_python_file_reintroduces_a_graph_store() -> None:
    offenders = []
    for path in _tracked_python_files():
        if path.resolve() == THIS_FILE:
            continue  # this file names neo4j/graph_search to check for them
        hits = _reintroduces_graph_store(path)
        if hits:
            offenders.append(f"{path.relative_to(PROJECT_ROOT)}: {hits}")

    assert (
        not offenders
    ), "tracked Python files reintroduce a graph store: " + "; ".join(offenders)
