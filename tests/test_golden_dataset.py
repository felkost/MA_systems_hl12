"""Schema/contract tests for `tests/golden_dataset.json` (stage 6).

The dataset is the measuring instrument for stages 7-9; a malformed or
drifted dataset would make every later stage's numbers meaningless without
any of those stages' own tests catching it. These tests pin exactly what
the brief and the plan require: three categories, five cases each, unique
inputs, non-empty expected_output -- plus two project-specific guards the
plan's own audit named:

- the freshness case's cutoff must stay strictly ahead of the corpus's own
  freshest source, or it silently turns into a corpus lookup that passes
  for the wrong reason;
- every case that needs a fixture file actually has one, present and
  readable, so a stage-9 live run does not fail on a missing file instead
  of measuring what it set out to measure.

All of the above except the last (an explicit `@pytest.mark.smoke` test)
run at gate tier, offline, with no dependency on `index/manifest.json` --
`index/` is gitignored and never built in CI, so nothing here reads it
directly.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

DATASET_PATH = Path(__file__).resolve().parent / "golden_dataset.json"
FIXTURES_DIR = Path(__file__).resolve().parent.parent / "evals" / "fixtures"
INDEX_MANIFEST_PATH = Path(__file__).resolve().parent.parent / "index" / "manifest.json"

# The corpus's own freshest source, as arXiv-ID-encoded in its filename
# (`index/manifest.json`'s `sources` map), measured directly against that
# file on 2026-08-22. A committed constant, not a live read, because
# `index/` is gitignored and absent in CI (`.github/workflows/gate.yml`'s
# offline job never runs `python ingest.py`) -- the same shape
# `tests/test_retriever_manifest.py` already uses (a synthetic manifest,
# never the real one, at gate tier).
_MEASURED_FRESHEST_CORPUS_SOURCE = "2606.22151v1.pdf"  # Jun 2026 --
# freshest source in index/manifest.json's `sources` as of 2026-08-22.

_REQUIRED_CATEGORIES = {"happy_path", "edge_case", "failure_case"}
_REQUIRED_FIELDS = ("id", "category", "input", "expected_output", "rationale")

_ARXIV_ID_PATTERN = re.compile(r"^(\d{2})(\d{2})\.\d+")


def _arxiv_year_month(filename: str) -> tuple[int, int] | None:
    """Parse an arXiv-convention `YYMM.NNNNN...` filename into (year, month).

    Returns ``None`` for a filename that does not follow the convention
    (an ACL Anthology ID, a plain topic name) rather than raising -- most
    of `index/manifest.json`'s own `sources` keys are not arXiv IDs at
    all, and a normal `data/` addition must not break this helper.
    """
    match = _ARXIV_ID_PATTERN.match(filename)
    if match is None:
        return None
    return 2000 + int(match.group(1)), int(match.group(2))


def _load_dataset() -> list[dict[str, object]]:
    with DATASET_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


def _freshness_case(dataset: list[dict[str, object]]) -> dict[str, object]:
    candidates = [case for case in dataset if "freshness_cutoff" in case]
    assert len(candidates) == 1, (
        "expected exactly one case carrying freshness_cutoff, "
        f"found {len(candidates)}"
    )
    return candidates[0]


def test_dataset_has_exactly_three_categories_five_cases_each() -> None:
    dataset = _load_dataset()
    by_category: dict[str, int] = {}
    for case in dataset:
        category = str(case["category"])
        by_category[category] = by_category.get(category, 0) + 1

    assert set(by_category) == _REQUIRED_CATEGORIES, sorted(by_category)
    assert all(count == 5 for count in by_category.values()), by_category
    assert len(dataset) == 15


def test_all_inputs_are_unique() -> None:
    dataset = _load_dataset()
    inputs = [case["input"] for case in dataset]
    assert len(inputs) == len(set(inputs)), "duplicate `input` found"


def test_every_case_has_non_empty_expected_output() -> None:
    dataset = _load_dataset()
    for case in dataset:
        expected_output = case.get("expected_output")
        assert isinstance(expected_output, str), case["id"]
        assert expected_output.strip(), f"{case['id']} has empty expected_output"


def test_required_fields_present_and_typed() -> None:
    dataset = _load_dataset()
    ids = []
    for case in dataset:
        for field in _REQUIRED_FIELDS:
            assert field in case, f"{case.get('id', '?')} missing {field!r}"
            assert isinstance(case[field], str), f"{case['id']}.{field} is not str"
        ids.append(case["id"])

    assert len(ids) == len(set(ids)), "duplicate `id` found"


def test_freshness_case_cutoff_is_strictly_after_measured_freshest_corpus_source() -> (
    None
):
    dataset = _load_dataset()
    case = _freshness_case(dataset)

    cutoff_year, cutoff_month = (
        int(part) for part in str(case["freshness_cutoff"]).split("-")
    )
    freshest = _arxiv_year_month(_MEASURED_FRESHEST_CORPUS_SOURCE)
    assert freshest is not None

    assert (cutoff_year, cutoff_month) > freshest, (
        f"{case['id']}'s freshness_cutoff {case['freshness_cutoff']!r} "
        f"is not strictly after the measured freshest source {freshest!r}"
    )


def test_adversarial_fixture_files_exist_and_are_readable() -> None:
    dataset = _load_dataset()
    fixture_cases = [case for case in dataset if "fixture" in case]
    assert fixture_cases, "expected at least one case carrying a `fixture` key"

    for case in fixture_cases:
        fixture_path = FIXTURES_DIR / str(case["fixture"])
        assert fixture_path.is_file(), f"{case['id']}: {fixture_path} missing"
        assert fixture_path.read_text(
            encoding="utf-8"
        ).strip(), f"{case['id']}: {fixture_path} is empty"

    poisoned_cases = [case for case in dataset if case.get("needs_poisoned_index")]
    assert len(poisoned_cases) == 1, poisoned_cases
    assert "fixture" in poisoned_cases[0], (
        "needs_poisoned_index case must also carry a fixture key, so the two "
        "fields cannot drift apart silently"
    )


@pytest.mark.smoke
def test_committed_freshest_source_constant_has_not_gone_stale() -> None:
    """Re-derive the freshest source from the real, locally built index.

    Skipped when `index/manifest.json` does not exist -- true in CI (never
    built there) and in a fresh clone before `python ingest.py` runs. This
    is the check that "loudly breaks" when `data/` gains a source dated
    after the committed constant above, per the plan's own intent; it
    cannot run at gate tier because the artefact it reads is gitignored.
    """
    if not INDEX_MANIFEST_PATH.is_file():
        pytest.skip(
            f"{INDEX_MANIFEST_PATH} does not exist -- run `python ingest.py` "
            "locally to exercise this check"
        )

    with INDEX_MANIFEST_PATH.open(encoding="utf-8") as handle:
        manifest = json.load(handle)

    dated_sources = [
        (name, _arxiv_year_month(name)) for name in manifest.get("sources", {})
    ]
    real_freshest = max(
        (year_month for _, year_month in dated_sources if year_month is not None),
        default=None,
    )
    assert real_freshest is not None, "no arXiv-dated source in the real manifest"

    committed_freshest = _arxiv_year_month(_MEASURED_FRESHEST_CORPUS_SOURCE)
    assert committed_freshest is not None

    assert real_freshest <= committed_freshest, (
        f"the real corpus's freshest source {real_freshest!r} is newer than "
        f"the committed constant {committed_freshest!r} -- update "
        "_MEASURED_FRESHEST_CORPUS_SOURCE and the freshness_cutoff case in "
        "tests/golden_dataset.json"
    )
