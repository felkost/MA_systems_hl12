"""The prompt *name* and *label* registry -- not prompt text. Every agent's
system prompt now lives in Langfuse Prompt Management, fetched by
`prompt_store.PromptStore.get(name, label=...)`; this module only records
which Langfuse prompt name corresponds to which role, so
`supervisor.py`/`orchestrator.py` do not scatter that mapping across two
files.

Ported from `MA_systems_hl11`'s registry mechanism
(`p*`/`r*`/`c*`/`s*`/`w1`, `build_*_prompt(version)`), which this project
deletes outright rather than keeping alongside the Langfuse path: a
fallback prompt constant in code would silently become the real prompt the
day a Langfuse fetch fails, which is exactly the failure mode this
project's own requirement exists to prevent. The prompt text itself is
seeded once, by hand, into Langfuse, each Langfuse prompt started from a
recorded hl11 version (`p2`/`r2`/`c2`/`s2`/`w1`).
"""

from __future__ import annotations

PROMPT_NAMES: dict[str, str] = {
    "planner": "hl12-planner",
    "researcher": "hl12-researcher",
    "critic": "hl12-critic",
    "supervisor": "hl12-supervisor",
    "composer": "hl12-composer",
    "critic_verification": "hl12-critic-verification",
}

# The only compile variable any registered prompt needs -- the Critic's
# `{{today}}` (ISO date). A second prompt gaining its own variable is a
# reason to widen this to a per-prompt mapping, not to add a second
# module-level set next to it.
CRITIC_PROMPT_VARIABLES: frozenset[str] = frozenset({"today"})
