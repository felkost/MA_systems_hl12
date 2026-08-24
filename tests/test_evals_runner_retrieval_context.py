"""`retrieval_context_for_agent` -- ancestor-scoped `retrieval.chunks`
extraction (stage 7, D7.1/D7.2).

Adversarial review of `docs/specs/stage-7.md` found this design decision
the hard way: a flat `name == "tool.knowledge_search"` filter over a real
run (`runs/0e9d703a-.../spans.json`) returns chunks belonging to the
Planner alongside the Researcher's own, and a second real run
(`runs/afc7144a-.../spans.json`) has a **third** contributor, the Critic --
all three agents share the `knowledge_search` tool. Scoring Groundedness
against an unscoped mix is not groundedness. These tests pin the scoping
against hand-built span trees, not just the two real dumps that happened to
motivate it.
"""

from __future__ import annotations

from evals.runner import RunSpans, retrieval_context_for_agent


def _span(
    span_id: str,
    parent_span_id: str | None,
    name: str,
    *,
    chunks: list[str] | None = None,
) -> dict[str, object]:
    attributes: dict[str, object] = {}
    if chunks is not None:
        attributes["retrieval.chunks"] = chunks
    return {
        "trace_id": "t1",
        "span_id": span_id,
        "parent_span_id": parent_span_id,
        "name": name,
        "start_time": 0,
        "end_time": 1,
        "status": "OK",
        "attributes": attributes,
    }


def test_selects_only_chunks_under_the_named_agent() -> None:
    # Mirrors the real 0e9d703a trace's shape: one knowledge_search under
    # the planner, one under the researcher.
    spans = [
        _span("root", None, "repl.question"),
        _span("plan-tool", "root", "tool.plan"),
        _span("agent-planner", "plan-tool", "agent.planner"),
        _span("ks-planner", "agent-planner", "tool.knowledge_search", chunks=["P1"]),
        _span("research-tool", "root", "tool.research"),
        _span("agent-researcher", "research-tool", "agent.researcher"),
        _span(
            "ks-researcher",
            "agent-researcher",
            "tool.knowledge_search",
            chunks=["R1", "R2"],
        ),
    ]
    run = RunSpans(run_id="r1", spans=spans)

    assert retrieval_context_for_agent(run, "agent.researcher") == ["R1", "R2"]
    assert retrieval_context_for_agent(run, "agent.planner") == ["P1"]


def test_three_contributors_stay_separated() -> None:
    # Mirrors the real afc7144a trace's shape: planner, researcher, and
    # critic all called knowledge_search in one run.
    spans = [
        _span("root", None, "repl.question"),
        _span("agent-planner", "root", "agent.planner"),
        _span("ks-planner", "agent-planner", "tool.knowledge_search", chunks=["P"]),
        _span("agent-researcher", "root", "agent.researcher"),
        _span(
            "ks-researcher-1",
            "agent-researcher",
            "tool.knowledge_search",
            chunks=["R1"],
        ),
        _span(
            "ks-researcher-2",
            "agent-researcher",
            "tool.knowledge_search",
            chunks=["R2"],
        ),
        _span("agent-critic", "root", "agent.critic"),
        _span("ks-critic", "agent-critic", "tool.knowledge_search", chunks=["C"]),
    ]
    run = RunSpans(run_id="r2", spans=spans)

    assert retrieval_context_for_agent(run, "agent.planner") == ["P"]
    assert retrieval_context_for_agent(run, "agent.researcher") == ["R1", "R2"]
    assert retrieval_context_for_agent(run, "agent.critic") == ["C"]


def test_order_preserved_and_duplicates_collapsed() -> None:
    spans = [
        _span("root", None, "repl.question"),
        _span("agent-researcher", "root", "agent.researcher"),
        _span("ks-1", "agent-researcher", "tool.knowledge_search", chunks=["A", "B"]),
        _span("ks-2", "agent-researcher", "tool.knowledge_search", chunks=["B", "C"]),
    ]
    run = RunSpans(run_id="r3", spans=spans)

    assert retrieval_context_for_agent(run, "agent.researcher") == ["A", "B", "C"]


def test_span_with_missing_parent_is_excluded_not_assumed_in_scope() -> None:
    spans = [
        _span("agent-researcher", "root-not-in-dump", "agent.researcher"),
        _span(
            "ks-orphan-ancestor",
            "some-missing-node",
            "tool.knowledge_search",
            chunks=["X"],
        ),
    ]
    run = RunSpans(run_id="r4", spans=spans)

    # "agent-researcher" itself has a parent absent from the dump, so a
    # span whose own ancestor walk hits that same gap must not silently
    # match -- it is excluded, not assumed to belong to agent.researcher.
    assert retrieval_context_for_agent(run, "agent.researcher") == []


def test_no_matching_agent_returns_empty_list() -> None:
    spans = [
        _span("root", None, "repl.question"),
        _span("agent-planner", "root", "agent.planner"),
        _span("ks-planner", "agent-planner", "tool.knowledge_search", chunks=["P"]),
    ]
    run = RunSpans(run_id="r5", spans=spans)

    assert retrieval_context_for_agent(run, "agent.researcher") == []


def test_knowledge_search_span_with_no_chunks_attribute_contributes_nothing() -> None:
    # E.g. "No matching passages in the knowledge base" -- knowledge_search
    # returns early and never sets retrieval.chunks (tools.py:272-273).
    spans = [
        _span("root", None, "repl.question"),
        _span("agent-researcher", "root", "agent.researcher"),
        _span("ks-empty", "agent-researcher", "tool.knowledge_search"),
    ]
    run = RunSpans(run_id="r6", spans=spans)

    assert retrieval_context_for_agent(run, "agent.researcher") == []


def test_read_url_span_counts_as_retrieval_symmetrically() -> None:
    # Stage 9e, D9e.7's third lever: `read_url` now writes retrieval.chunks
    # (tools.py), so a page a real tool fetched grounds the same way a
    # knowledge_search hit does.
    spans = [
        _span("root", None, "repl.question"),
        _span("agent-researcher", "root", "agent.researcher"),
        _span("ru-1", "agent-researcher", "tool.read_url", chunks=["page text"]),
    ]
    run = RunSpans(run_id="r7", spans=spans)

    assert retrieval_context_for_agent(run, "agent.researcher") == ["page text"]


def test_knowledge_search_and_read_url_chunks_combine_in_span_order() -> None:
    spans = [
        _span("root", None, "repl.question"),
        _span("agent-researcher", "root", "agent.researcher"),
        _span("ks-1", "agent-researcher", "tool.knowledge_search", chunks=["K1"]),
        _span("ru-1", "agent-researcher", "tool.read_url", chunks=["U1"]),
    ]
    run = RunSpans(run_id="r8", spans=spans)

    assert retrieval_context_for_agent(run, "agent.researcher") == ["K1", "U1"]


def test_web_search_span_never_counts_as_retrieval() -> None:
    # A search snippet is not a retrieved document (D9e.7's own scoping
    # decision) -- web_search stays excluded even if it carried the same
    # attribute name by accident.
    spans = [
        _span("root", None, "repl.question"),
        _span("agent-researcher", "root", "agent.researcher"),
        _span("ws-1", "agent-researcher", "tool.web_search", chunks=["snippet"]),
    ]
    run = RunSpans(run_id="r9", spans=spans)

    assert retrieval_context_for_agent(run, "agent.researcher") == []
