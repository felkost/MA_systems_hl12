"""Structural guard for D3.5: a factory's own tools must match its allowlist.

Under D3.2, agent factories receive `tools` from their caller instead of
importing `tools.py` -- which means a caller's mistake is the only thing
standing between an agent and a capability its architecture row denies it.
This check fails at construction, before any model call, so a wrong tool
list is never seen by the model at all.
"""

from __future__ import annotations

from collections.abc import Sequence

from langchain_core.tools import BaseTool


class ToolAllowlistError(Exception):
    """`tools` passed to an agent factory does not match its allowlist."""


def assert_allowlist(
    tools: Sequence[BaseTool], allowlist: tuple[str, ...], agent_name: str
) -> None:
    """Refuse unless `tools` names exactly the tools in `allowlist`.

    Parameters
    ----------
    tools : Sequence of BaseTool
        What the caller handed the factory.
    allowlist : tuple of str
        The agent's architecture-row allowlist, e.g. `PLANNER_ALLOWLIST`.
    agent_name : str
        Named in the error so a multi-agent wiring mistake is diagnosable
        without a debugger.

    Raises
    ------
    ToolAllowlistError
        `tools` contains a name outside `allowlist`, or is missing one.
    """
    names = {tool.name for tool in tools}
    expected = set(allowlist)
    unexpected = names - expected
    missing = expected - names
    if unexpected or missing:
        parts = []
        if unexpected:
            parts.append(f"unexpected: {sorted(unexpected)}")
        if missing:
            parts.append(f"missing: {sorted(missing)}")
        raise ToolAllowlistError(
            f"{agent_name} tools do not match its allowlist ({', '.join(parts)})"
        )
