"""No `.py` file in this project may contain system prompt text.

The check is structural, not lexical: any string literal longer than
`PROMPT_LENGTH_FLOOR` characters, anywhere in a project module, is treated as
prompt text until proven otherwise. A bare docstring statement (the first
statement of a module/function/class body) is exempt -- it documents, it is
never sent to a model (`docs/specs/stage-1.md`, requirement 3 -- no system
prompt text may live in a `.py` file, because a fallback constant in code
silently becomes the real prompt the day a Langfuse fetch fails).
"""

from __future__ import annotations

import ast
from pathlib import Path

import paths

PROMPT_LENGTH_FLOOR = 400

_SKIPPED_DIRS = frozenset({".venv", ".cache", ".git", ".claude", "__pycache__", "docs"})

_DocstringHost = ast.Module | ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef

# Long string constants that are not an agent's system prompt and are
# therefore out of scope for requirement 3 -- each entry names exactly the
# assignment target that is exempt, not the whole file, so a *new* long
# literal in the same module still trips the check.
_ALLOWED_LONG_LITERALS: dict[str, frozenset[str]] = {
    "tools.py": frozenset({"UNTRUSTED_PREAMBLE"}),
    # `read_url`'s prompt-injection guard: it wraps fetched web content as
    # data, never passed as `system_prompt=` to `create_agent` -- an
    # infra-owned defensive template, not a system prompt (see CLAUDE.md,
    # "Three deliberate removals" is unrelated; this is OWASP LLM01
    # defense-in-depth, layer 2).
}


def _project_python_files() -> list[Path]:
    root = paths.PROJECT_ROOT
    return sorted(
        path
        for path in root.rglob("*.py")
        if not _SKIPPED_DIRS.intersection(path.relative_to(root).parts)
        and "tests" not in path.relative_to(root).parts
        and path.name != "conftest.py"
    )


class _NonDocstringStringLiteralCollector(ast.NodeVisitor):
    """Every string constant in the module except a body's own leading
    docstring statement.

    A plain `ast.walk` cannot tell a docstring's `Constant` apart from any
    other string literal, since it carries no parent pointer -- this visitor
    instead recurses body-by-body, skipping exactly the first statement of
    each `Module`/`FunctionDef`/`AsyncFunctionDef`/`ClassDef` body when that
    statement is a bare string expression.
    """

    def __init__(self, allowed_targets: frozenset[str] = frozenset()) -> None:
        self.literals: list[str] = []
        self._allowed_targets = allowed_targets

    def visit_Assign(self, node: ast.Assign) -> None:
        target_names = {t.id for t in node.targets if isinstance(t, ast.Name)}
        if target_names & self._allowed_targets:
            return  # exempt assignment target -- see _ALLOWED_LONG_LITERALS
        self.generic_visit(node)

    def _visit_body(self, body: list[ast.stmt]) -> None:
        for index, statement in enumerate(body):
            if index == 0 and _is_docstring_statement(statement):
                continue
            self.visit(statement)

    def visit_Module(self, node: ast.Module) -> None:
        self._visit_body(node.body)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_body(node.body)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_body(node.body)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._visit_body(node.body)

    def visit_Constant(self, node: ast.Constant) -> None:
        if isinstance(node.value, str):
            self.literals.append(node.value)


def _is_docstring_statement(statement: ast.stmt) -> bool:
    return (
        isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Constant)
        and isinstance(statement.value.value, str)
    )


def _non_docstring_string_literals(
    tree: ast.Module, allowed_targets: frozenset[str]
) -> list[str]:
    collector = _NonDocstringStringLiteralCollector(allowed_targets)
    collector.visit(tree)
    return collector.literals


def test_no_module_carries_prompt_text() -> None:
    offenders: list[str] = []
    for path in _project_python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        allowed = _ALLOWED_LONG_LITERALS.get(path.name, frozenset())
        for literal in _non_docstring_string_literals(tree, allowed):
            if len(literal) > PROMPT_LENGTH_FLOOR:
                offenders.append(f"{path.name}: {literal[:60]!r}...")
    assert offenders == [], "prompt text found in code: " + "; ".join(offenders)
