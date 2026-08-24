# MA_systems_hl12 — Langfuse observability for a multi-agent research system

A terminal multi-agent research system — a Supervisor coordinating a Planner,
a Researcher and a Critic in a Plan → Research → Critique loop — instrumented
end to end with **Langfuse**: every run is one trace with a full
sub-agent/tool-call tree, grouped into a session with a user id; every
agent's system prompt is fetched from Langfuse Prompt Management by label,
never hardcoded in this repository; and online LLM-as-a-Judge evaluators
score new traces automatically.

This repository solves homework-lesson-12. The system itself is ported from
[`MA_systems_hl11`](https://github.com/felkost/MA_systems_hl11); the
engineering weight here sits in `prompt_store.py`, `observability.py` and
`evals/`.

> **Status: stage 3 (Langfuse Datasets) complete.** Prompt management
> (stage 1), tracing with session/user grouping (stage 2), and an idempotent
> golden-dataset sync into Langfuse Datasets (stage 3, verified live: dataset
> `hl12-golden` with 15 items) are done. The four online evaluators are still
> in progress — see `CONTRIBUTING.md` for the stage sequence, and the
> repository's own commit history for what has actually landed.

## Setup

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt -r requirements-dev.txt
cp .env.example .env
```

Fill in `.env`:
- `OPENROUTER_API_KEY` — the models the agents run on.
- `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` / `LANGFUSE_BASE_URL` — a
  Langfuse Cloud project created for this assignment. Do not reuse an
  `MA_systems_hl11` project's keys: every trace, prompt and score would land
  in the wrong dashboard.

`deepeval` is in `requirements-dev.txt`, not `requirements.txt`: it is a test
tool, not a runtime dependency. Installing only `requirements.txt` gives you
tests that cannot run.

## Seed the prompts

Every agent's system prompt is fetched from Langfuse Prompt Management by
name and the `production` label — none is hardcoded in this repository. Six
prompts must exist in the Langfuse project referenced above before the system
can run: `hl12-planner`, `hl12-researcher`, `hl12-critic`, `hl12-supervisor`,
`hl12-composer`, `hl12-critic-verification`. See `CLAUDE.md`'s architecture
table for what `prompt_store.py` expects from each.

## Run

```bash
python ingest.py     # build the Chroma index from data/
python main.py       # the REPL
```

Reports are written to `output/` and only after you approve them: the single
write in the system is gated by a human-in-the-loop checkpoint. Every turn
produces one trace in the Langfuse project configured above, grouped by
session (`--session-id`, default: a fresh id per process) and user
(`--user-id`, default: `DEFAULT_USER_ID`/`"anonymous"`).

## Tests

Three tiers, deliberately separate.

```bash
# Offline gate — no network, no API keys, no cost.
.venv/Scripts/python.exe -m black --check . tests/*.py && \
.venv/Scripts/python.exe -m flake8 . && \
.venv/Scripts/python.exe -m mypy . && \
.venv/Scripts/python.exe -m pytest -q -m "not smoke and not eval"

# Smoke — boots real services. On request.
.venv/Scripts/python.exe -m pytest -q -m smoke

# Evaluation — ported from MA_systems_hl11, calls live models and costs money.
deepeval test run tests/
```

The evaluation tier is excluded from the gate. A test that goes red because a
judge model drifted teaches everyone to ignore a red CI, and a CI nobody
trusts stops catching real breakage.

## Layout

Flat module layout: every filename stays where it was placed, and layering is
a property assigned per file in `CLAUDE.md`'s architecture table, enforced by
`tests/test_layering.py`. `data/` holds the source corpus; `output/` holds
approved reports; `screenshots/` holds the four Langfuse UI screenshots this
assignment delivers.

## What leaves this machine

Model calls go to OpenRouter. Traces, prompt fetches and evaluator calls go to
Langfuse Cloud. Nothing else — no self-hosted service, no graph database, no
MCP or A2A network hop.

## License

No license file is included; this is coursework, not a published package.
