"""Enforces the layer table in CLAUDE.md by walking every module's imports.

The table assigns a layer per file rather than per directory, because the
assignment prescribes a flat module layout and inventing a package tree would
make the deliverable harder to grade. Without this test the layer assignment
would be a paragraph nobody checks.

Two negative rules carry most of the value here, and both are about keeping
the two coordination paths independent. They are meant to enforce the same
revision cap by different mechanisms -- middleware on one side, state
arithmetic on the other -- so that the comparison between them says something.
An import between them would make a change to one silently change the other,
and a sub-agent importing a coordinator would turn the dependency arrow around
entirely.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent

KERNEL = "kernel"
DOMAIN = "domain"
INFRA = "infra"
APPLICATION = "application"
INTERFACE = "interface"
OBS = "obs"

# Mirrors the "Architecture" table in CLAUDE.md. A module absent from this map
# has no declared layer, which `test_every_module_has_a_declared_layer` treats
# as a failure rather than as permission to import anything.
LAYER_OF_MODULE: dict[str, str] = {
    # Package markers (`__init__.py`) declare the same layer as the modules
    # they contain -- an empty marker file still needs a home in the table,
    # or a fresh `agents/__init__.py` silently escapes the import-walk.
    "agents": DOMAIN,
    "evals": OBS,
    "main": INTERFACE,
    "supervisor": APPLICATION,
    "orchestrator": APPLICATION,
    "hitl": APPLICATION,
    "agents.planner": DOMAIN,
    "agents.research": DOMAIN,
    "agents.critic": DOMAIN,
    "agents._allowlist": DOMAIN,
    "schemas": DOMAIN,
    "prompts": DOMAIN,
    "tools": INFRA,
    "prompt_store": INFRA,
    "middleware": INFRA,
    "grounding": INFRA,
    "models": INFRA,
    "retriever": INFRA,
    "ingest": INFRA,
    "config": KERNEL,
    "paths": KERNEL,
    "observability": OBS,
    "evals.deepeval_model": OBS,
    "evals.runner": OBS,
    "evals.summarize_e2e": OBS,
    "evals.aggregate_runs": OBS,
    "evals.repeat_summary": OBS,
}

#  Stage 2 finding: a layer must be allowed to import itself. The table in
# CLAUDE.md ("May import") never says a layer may reach its own -- an
# omission nobody could see at stage 0, when this dict was written and no
# code existed yet. The first real modules (config.py -> paths.py, both
# kernel; retriever.py -> models.py, both infra; tools.py -> retriever.py,
# both infra) all need a same-layer import, so each set below now includes
# its own layer explicitly.
MAY_IMPORT: dict[str, frozenset[str]] = {
    KERNEL: frozenset({KERNEL}),
    DOMAIN: frozenset({KERNEL, DOMAIN}),
    INFRA: frozenset({KERNEL, DOMAIN, INFRA}),
    APPLICATION: frozenset({KERNEL, DOMAIN, INFRA, APPLICATION}),
    INTERFACE: frozenset({KERNEL, DOMAIN, INFRA, APPLICATION, INTERFACE, OBS}),
    OBS: frozenset({KERNEL, DOMAIN, INFRA, OBS}),
}

# Pairs that must never import each other in **either** direction, regardless
# of what the layer rules alone would permit: the two coordination paths are
# meant to be independent implementations of the same contract, so an import
# between them would let a change to one silently change the other.
BIDIRECTIONAL_FORBIDDEN_PAIRS: tuple[tuple[str, str], ...] = (
    ("supervisor", "orchestrator"),
)

# Pairs forbidden in **one** direction only: a sub-agent must never import a
# coordinator (CLAUDE.md, Invariants -- "agents/* must never import
# supervisor or orchestrator"), which would invert the dependency arrow the
# whole agent-as-tool/StateGraph design rests on. The reverse import is not
# just permitted but required -- stage 4 audit finding, D4.22
# (docs/specs/stage-4.md): the first version of this test (stage 0, written
# before either coordinator existed) checked both directions for these pairs
# too, which would have forbidden supervisor.py/orchestrator.py from ever
# calling `create_planner_agent`/`create_research_agent`/`create_critic_agent`
# -- the only functions that build a compiled sub-agent graph at all.
DIRECTIONAL_FORBIDDEN_PAIRS: tuple[tuple[str, str], ...] = (
    ("agents.planner", "supervisor"),
    ("agents.planner", "orchestrator"),
    ("agents.research", "supervisor"),
    ("agents.research", "orchestrator"),
    ("agents.critic", "supervisor"),
    ("agents.critic", "orchestrator"),
)

# Directories that hold no project modules: virtualenvs, caches, local-only
# working areas, and the test suite itself. `.claude` covers Claude Code's
# own worktree isolation (`.claude/worktrees/<name>/`), which can hold a
# full nested checkout of this repo while a background session runs --
# without this, that checkout's own `.py` files get treated as project
# modules absent from `LAYER_OF_MODULE`, failing this test for a reason
# that has nothing to do with this project's own code.
_SKIPPED_DIRS = frozenset(
    {
        ".venv",
        ".cache",
        ".git",
        ".claude",
        "__pycache__",
        "docs",
        "tests",
        "index",
        "runs",
    }
)


def _module_name(path: Path) -> str:
    """Dotted name of `path` relative to the project root, without `.py`."""
    relative = path.relative_to(PROJECT_ROOT).with_suffix("")
    parts = [part for part in relative.parts if part != "__init__"]
    return ".".join(parts)


def _project_modules() -> list[Path]:
    """Every project `.py` file, excluding vendored and local-only trees."""
    return sorted(
        path
        for path in PROJECT_ROOT.rglob("*.py")
        if not _SKIPPED_DIRS.intersection(path.relative_to(PROJECT_ROOT).parts)
    )


def _imported_project_modules(path: Path) -> set[str]:
    """Project-local modules that `path` imports, by dotted name.

    Resolves each import against `LAYER_OF_MODULE` and against its package
    prefix, so `from agents.planner import x` and `import agents.planner` both
    resolve to `agents.planner`, while `import json` resolves to nothing.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module)
            names.update(f"{node.module}.{alias.name}" for alias in node.names)
    return {name for name in names if name in LAYER_OF_MODULE}


def test_every_module_has_a_declared_layer() -> None:
    """A new module must be added to the table before it may import anything.

    Catching this here rather than at review time is what stops the table in
    CLAUDE.md from quietly falling behind the tree it describes.
    """
    undeclared = [
        _module_name(path)
        for path in _project_modules()
        if _module_name(path) not in LAYER_OF_MODULE
    ]
    assert not undeclared, (
        f"modules missing from LAYER_OF_MODULE (and from CLAUDE.md's "
        f"architecture table): {undeclared}"
    )


@pytest.mark.parametrize("path", _project_modules(), ids=_module_name)
def test_module_imports_respect_its_layer(path: Path) -> None:
    """No module imports a layer its own layer may not reach."""
    importer = _module_name(path)
    importer_layer = LAYER_OF_MODULE[importer]
    allowed = MAY_IMPORT[importer_layer]

    violations = [
        f"{importer} ({importer_layer}) imports {imported} "
        f"({LAYER_OF_MODULE[imported]})"
        for imported in sorted(_imported_project_modules(path))
        if imported != importer and LAYER_OF_MODULE[imported] not in allowed
    ]
    assert not violations, "; ".join(violations)


@pytest.mark.parametrize("left,right", BIDIRECTIONAL_FORBIDDEN_PAIRS, ids=lambda v: v)
def test_bidirectional_forbidden_pairs_never_import_each_other(
    left: str, right: str
) -> None:
    """The two coordination paths stay independent implementations of the
    same contract, in both directions.

    Passing vacuously while neither module exists is intended: the rule is
    written before the code so that the first version of that code is already
    constrained by it.
    """
    for importer, forbidden in ((left, right), (right, left)):
        path = PROJECT_ROOT / f"{importer.replace('.', '/')}.py"
        if not path.exists():
            continue
        assert forbidden not in _imported_project_modules(path), (
            f"{importer} imports {forbidden}; they must stay independent "
            "(see CLAUDE.md, Invariants)"
        )


@pytest.mark.parametrize(
    "sub_agent,coordinator", DIRECTIONAL_FORBIDDEN_PAIRS, ids=lambda v: v
)
def test_sub_agents_never_import_a_coordinator(
    sub_agent: str, coordinator: str
) -> None:
    """A sub-agent must never import `supervisor`/`orchestrator` (CLAUDE.md,
    Invariants) -- the reverse import is not checked here because it is
    required: a coordinator's only way to obtain a compiled sub-agent graph
    is `create_planner_agent`/`create_research_agent`/`create_critic_agent`,
    all defined in `agents.*` (D4.22, `docs/specs/stage-4.md` -- stage 0's
    original bidirectional check would have forbidden the architecture it
    was meant to protect).

    Passing vacuously while `sub_agent`'s module exists but reads nothing
    from `coordinator` is the normal case at every stage; it stays
    meaningful once `supervisor.py`/`orchestrator.py` exist because an
    accidental upward import would now be a real, checkable violation.
    """
    path = PROJECT_ROOT / f"{sub_agent.replace('.', '/')}.py"
    if not path.exists():
        return
    assert coordinator not in _imported_project_modules(path), (
        f"{sub_agent} imports {coordinator}; a sub-agent must never import a "
        "coordinator (see CLAUDE.md, Invariants)"
    )
