"""Each agent gets exactly the tools its architecture row prescribes.

The four named lists in `tools.py` are the single source of truth the agent
factories build from (stage 3) -- a name missing here is a tool the model
never learns exists, and a name present here that shouldn't be widens what
that agent can do. Ported from hl8 minus `graph_search`
(`docs/specs/stage-2.md`, deliverables table: "`tools.py` | hl8, minus
`graph_search`"). The Planner in particular gets no `read_url`: it explores
with search tools only, it does not deep-read a single source.
"""

from __future__ import annotations

import tools


def _names(tool_list: list) -> set[str]:
    return {t.name for t in tool_list}


def test_planner_tools_are_search_only_with_no_read_url() -> None:
    names = _names(tools.PLANNER_TOOLS)
    assert names == {"web_search", "knowledge_search"}
    assert "read_url" not in names


def test_researcher_tools_have_no_graph_search() -> None:
    names = _names(tools.RESEARCHER_TOOLS)
    assert names == {"web_search", "read_url", "knowledge_search"}
    assert "graph_search" not in names


def test_critic_tools_have_no_graph_search() -> None:
    names = _names(tools.CRITIC_TOOLS)
    assert names == {"web_search", "read_url", "knowledge_search"}
    assert "graph_search" not in names


def test_supervisor_tools_is_save_report_only() -> None:
    names = _names(tools.SUPERVISOR_TOOLS)
    assert names == {"save_report"}
