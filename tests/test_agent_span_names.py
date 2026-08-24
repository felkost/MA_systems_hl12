"""`tests/live_agents.py` must open the same `agent.<role>` span names the
real coordinators use.

`agent.planner`/`agent.researcher`/`agent.critic` are opened only at
`supervisor.py` and `orchestrator.py`'s own call sites -- measured this
session, `agents/*.py` opens none. `retrieval_context_for_agent` walks a
span's ancestor chain looking for one of these literal names; if
`live_agents.py`'s own constant ever drifts from what the real coordinators
use, a live run would silently produce a dump `retrieval_context`
can never match against anything, and the failure would look exactly like
"the model never called knowledge_search" rather than what it actually is.
This is an AST/source scan in the shape `tests/test_layering.py` already
uses, so a rename in one file cannot drift from the other unnoticed.
"""

from __future__ import annotations

import ast
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _agent_span_literals(module_path: Path) -> set[str]:
    """Every literal `"agent.<role>"` string passed to
    `start_as_current_span` in `module_path`."""
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    literals: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "start_as_current_span"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
            and node.args[0].value.startswith("agent.")
        ):
            literals.add(node.args[0].value)
    return literals


def test_live_agents_span_names_match_the_real_coordinators() -> None:
    from tests import live_agents

    coordinator_literals = _agent_span_literals(
        PROJECT_ROOT / "supervisor.py"
    ) | _agent_span_literals(PROJECT_ROOT / "orchestrator.py")

    # orchestrator.py also opens agent.composer, which has no sub-agent
    # factory and is out of scope for a per-role live-agent helper.
    role_literals = coordinator_literals - {"agent.composer"}

    assert role_literals, "no agent.<role> span literals found to compare against"
    assert set(live_agents.AGENT_SPAN_NAME.values()) == role_literals
