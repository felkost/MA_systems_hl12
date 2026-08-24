"""Shared eval-tier fixtures and helpers (stage 7; extended stage 9a,
`docs/specs/stage-9a.md`).

`live_settings` is session-scoped on purpose: `observability.configure_observability`
raises `RuntimeError` on a second call in one process
(`observability.py`'s `_CONFIGURED` sentinel), and every eval-tier test
in `deepeval test run tests/` shares one process. One provider, one
tmp-directory span dump root, reused by every live case in the session.

`eval_run_id`/`e2e_judge_model`/`fixture_base_url`/`record_case_cost` are
stage 9a's own additions, session-scoped for the same reason: one
`OpenRouterModel` instance shared across all 15 golden-dataset cases' 3
metrics is what lets its `usage_log` accumulate a real, measurable judge-cost
total (D9a.7), and one fixture HTTP server (D9a.1) is enough for the one
case that needs it.
"""

from __future__ import annotations

import json
from typing import Any, Iterator
from uuid import uuid4

import pytest
from deepeval.evaluate import assert_test
from deepeval.metrics.base_metric import BaseMetric
from deepeval.test_case import LLMTestCase

import models
import paths
from config import Settings, export_deepeval_timeout_override, load_settings
from evals.deepeval_model import OpenRouterModel
from evals.runner import RunSpans, total_agent_cost
from paths import PROJECT_ROOT
from tests.fixture_server import run_fixture_server
from tests.live_agents import configured_for_eval, eval_settings

_GOLDEN_DATASET_PATH = PROJECT_ROOT / "tests" / "golden_dataset.json"
_INDEX_MANIFEST_PATH = PROJECT_ROOT / "index" / "manifest.json"
_FIXTURES_DIR = PROJECT_ROOT / "evals" / "fixtures"


@pytest.fixture(scope="session")
def live_settings(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Settings]:
    """One real `Settings`, span dump redirected off the project's real
    `runs/` (`insights.md`'s stage-5 pollution mistake, avoided here the
    same way), with observability configured exactly once for the session.
    """
    settings = eval_settings(
        runs_dir=str(tmp_path_factory.mktemp("eval-runs")),
        settings=load_settings(),
    )
    # D9e.1: the binding knob for a judge-metric timeout is DeepEval's own
    # per-task budget, not the httpx read timeout `OpenRouterModel` already
    # carries. Exported once per session, after `load_settings()` has
    # already succeeded (this field has no code default that would let it
    # run at import time the way `_export_cache_env` does).
    export_deepeval_timeout_override(settings)
    with configured_for_eval(settings):
        yield settings


def _golden_cases_by_id() -> dict[str, dict[str, Any]]:
    return {
        case["id"]: case
        for case in json.loads(_GOLDEN_DATASET_PATH.read_text(encoding="utf-8"))
    }


def golden_input(case_id: str) -> str:
    """The `input` field of one `tests/golden_dataset.json` case, by id."""
    return str(_golden_cases_by_id()[case_id]["input"])


def all_case_ids() -> list[str]:
    """Every case id in `tests/golden_dataset.json`, in file order -- the
    parametrize source for `tests/test_e2e.py::test_golden_dataset` (stage
    9a, R4b's "усі 15")."""
    return list(_golden_cases_by_id().keys())


def golden_case(case_id: str) -> dict[str, Any]:
    """The full row of one `tests/golden_dataset.json` case, by id (stage
    9a) -- `golden_input` only ever needed `input`; `resolve_case_input`
    (below) and stage 9a's own `check_expects`/`skip_if_poisoned_chunk_absent`
    (`tests/test_e2e.py`) need `category`, `expected_output`, `fixture`,
    `needs_poisoned_index` and `expects` too.

    Raises
    ------
    KeyError
        No case with this id exists.
    """
    return _golden_cases_by_id()[case_id]


def resolve_case_input(case: dict[str, Any], fixture_base_url: str) -> str:
    """`case["input"]`, with `{fixture_url}` substituted for the one case
    that carries an HTTP-fetchable `fixture` (stage 9a, D9a.3).

    `needs_poisoned_index` is excluded even though that case also carries a
    `fixture` key: its fixture is for the knowledge index, never fetched
    over HTTP, and its own `input` string has no `{fixture_url}` placeholder
    to begin with (verified against `tests/golden_dataset.json` directly).
    The guard exists so a future case that adds both a placeholder and
    `needs_poisoned_index` is not silently mis-resolved.
    """
    fixture_name = case.get("fixture")
    if fixture_name is None or case.get("needs_poisoned_index"):
        return str(case["input"])
    return str(case["input"]).format(fixture_url=f"{fixture_base_url}/{fixture_name}")


def skip_without_index() -> None:
    """Skip an eval-tier test that needs a live index this CI job never
    builds (`index/` is gitignored, the `evals` job runs no `ingest.py`
    step -- the same defect stage 6 found for its own freshness test,
    `docs/specs/stage-7.md` D7.9)."""
    if not _INDEX_MANIFEST_PATH.is_file():
        pytest.skip(
            f"{_INDEX_MANIFEST_PATH} does not exist -- run `python ingest.py` "
            "locally before the eval tier"
        )


def fixture_text(name: str) -> str:
    """Read a fixture file under `evals/fixtures/` by name."""
    path = PROJECT_ROOT / "evals" / "fixtures" / name
    return path.read_text(encoding="utf-8")


@pytest.fixture(scope="session")
def fixture_base_url() -> Iterator[str]:
    """One local HTTP server over `evals/fixtures/`, for the whole session
    (stage 9a, D9a.1). Only `adversarial-indirect-injection` uses it
    (`resolve_case_input`), but it is session-scoped rather than
    function-scoped for the same reason `live_settings` is: cheap to start
    once, and nothing about it is per-case state."""
    with run_fixture_server(_FIXTURES_DIR) as url:
        yield url


@pytest.fixture(scope="session")
def eval_run_id() -> str:
    """One id for this whole e2e evaluation session (stage 9a, D9a.8) --
    a different namespace from each case's own Supervisor `run_id` (D8.9).
    Groups this stage's own artefacts under `runs/<eval_run_id>/`."""
    return str(uuid4())


def _shared_judge_model(live_settings: Settings) -> OpenRouterModel:
    return OpenRouterModel(
        live_settings.judge_model_name or live_settings.model_name,
        api_key=live_settings.openrouter_api_key.get_secret_value(),
        usage_log=[],
        reasoning_effort=live_settings.judge_reasoning_effort,
    )


@pytest.fixture(scope="session")
def e2e_judge_model(live_settings: Settings) -> OpenRouterModel:
    """One judge model instance, shared across all 15 golden-dataset cases'
    3 metrics, with a real `usage_log` (stage 9a, D9a.7) -- sharing one
    instance is what lets the log accumulate a real total; a fresh instance
    per call would reset nothing on `usage_log` itself, but `test_e2e.py`
    needs the *same* list back after 45 measurements, not 45 separate ones.
    """
    return _shared_judge_model(live_settings)


@pytest.fixture(scope="session")
def component_judge_model(live_settings: Settings) -> OpenRouterModel:
    """One judge model instance shared across the four component test
    files' own live cases (stage 9e -- the "shared judge instance" phase-1
    deliverable, extended past `test_e2e.py`'s own D9a.7 fixture).

    Component-file judge cost had never been measured before this stage
    (`insights.md`, stage 9e planning: "the component cases' judge cost has
    never been measured") -- each test body built a fresh, unlogged judge
    model instead. Separate from `e2e_judge_model` because the two never
    run in the same process (D9e.12's two-invocation configuration keeps
    `tests/test_e2e.py` and the four component files in separate `deepeval
    test run` invocations), not because sharing one instance across both
    would be wrong.
    """
    return _shared_judge_model(live_settings)


def run_judged_case(
    eval_run_id: str,
    *,
    case_id: str,
    test_case: LLMTestCase,
    metrics: list[BaseMetric],
    judge: OpenRouterModel,
    spans: RunSpans,
) -> None:
    """Run `assert_test`, then persist this case's span dump and its
    measured agent/judge cost, whatever `assert_test` did (stage 9e).

    Shared by the four component test files. `tests/test_e2e.py` keeps its
    own richer version -- it also combines a deterministic pre-check's
    failure into the raised error (D9e.3), which no component-file case
    has an equivalent of.

    Raises
    ------
    Exception
        Whatever `assert_test` itself raised, re-raised after the `finally`
        block below has already run -- cost and span persistence must not
        depend on the case having passed, and must survive a DeepEval-internal
        crash as well as an ordinary metric failure (stage 9e phase 1b: a
        missing pywin32 made every case die on `AttributeError` from
        DeepEval's own result cache).
    """
    judge_calls_before = len(judge.usage_log or [])
    try:
        assert_test(test_case, metrics)
    finally:
        judge_cost = sum(
            models.compute_cost_or_raise(
                judge.get_model_name(),
                entry["prompt_tokens"],
                entry["completion_tokens"],
            )
            for entry in (judge.usage_log or [])[judge_calls_before:]
        )
        record_case_cost(
            eval_run_id,
            case_id=case_id,
            run_id=spans.run_id,
            agent_cost_usd=total_agent_cost(spans),
            judge_cost_usd=judge_cost,
        )
        persist_case_spans(eval_run_id, case_id=case_id, spans=spans)


def record_case_cost(
    eval_run_id: str,
    *,
    case_id: str,
    run_id: str,
    agent_cost_usd: float,
    judge_cost_usd: float,
) -> None:
    """Append one case's measured cost to `runs/<eval_run_id>/case-costs.jsonl`
    (stage 9a, D9a.8).

    Written incrementally, one line per case, while the data is still in
    memory in the process that produced it -- `evals/summarize_e2e.py` runs
    later, in a separate process, and reads this file rather than trying to
    reconstruct cost from state that will not have survived (the mistake
    this stage's own adversarial review caught in an earlier draft).
    """
    run_dir = paths.run_dir(eval_run_id)
    record = {
        "case_id": case_id,
        "run_id": run_id,
        "agent_cost_usd": agent_cost_usd,
        "judge_cost_usd": judge_cost_usd,
    }
    with (run_dir / "case-costs.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def persist_case_spans(eval_run_id: str, *, case_id: str, spans: RunSpans) -> None:
    """Write one case's own span dump to
    `runs/<eval_run_id>/spans/<case_id>.json` (stage 9e, D9e.2), before the
    `tmp_path_factory` root that produced it is cleaned up.

    Three stages in a row lost a diagnosis to exactly that cleanup (stage
    9b, 9d, and this stage's own planning session, `insights.md`): the
    `tmp_path_factory`-rooted span dump `live_settings` redirects to does
    not survive past the pytest process. Written directly from the
    already-loaded `RunSpans` rather than re-reading the temp file, since
    the caller has it in memory anyway.
    """
    spans_dir = paths.run_dir(eval_run_id) / "spans"
    spans_dir.mkdir(parents=True, exist_ok=True)
    (spans_dir / f"{case_id}.json").write_text(
        json.dumps(spans.spans, indent=2), encoding="utf-8"
    )


def load_case_costs(eval_run_id: str) -> list[dict[str, Any]]:
    """Read back every record `record_case_cost` wrote for one eval run.

    Does not create `runs/<eval_run_id>/` as a side effect (unlike
    `paths.run_dir`) -- a loader for a run that never wrote anything must
    not leave an empty directory behind.
    """
    path = paths.resolve("runs") / eval_run_id / "case-costs.jsonl"
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
