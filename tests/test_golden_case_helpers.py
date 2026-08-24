"""`golden_case`/`resolve_case_input` (stage 9a, `docs/specs/stage-9a.md`
D9a.2/D9a.3).

Offline against the real `tests/golden_dataset.json` -- these are plain
functions over a tracked, reviewed file, not a hand-built fixture, matching
`tests/conftest.py`'s own `golden_input` precedent.
"""

from __future__ import annotations

import pytest

from tests.conftest import golden_case, resolve_case_input


def test_golden_case_returns_the_full_row_not_only_input() -> None:
    case = golden_case("core-single-vs-multi-agent")

    assert case["id"] == "core-single-vs-multi-agent"
    assert case["category"] == "happy_path"
    assert "expected_output" in case


def test_resolve_case_input_substitutes_fixture_url_for_the_one_case_that_has_it() -> (
    None
):
    case = golden_case("adversarial-indirect-injection")

    resolved = resolve_case_input(case, "http://127.0.0.1:12345")

    assert "{fixture_url}" not in resolved
    assert "http://127.0.0.1:12345/injection_page.txt" in resolved


def test_resolve_case_input_leaves_every_other_case_unchanged() -> None:
    ordinary = golden_case("core-single-vs-multi-agent")
    assert resolve_case_input(ordinary, "http://127.0.0.1:1") == ordinary["input"]

    poisoned = golden_case("adversarial-poisoned-knowledge-base")
    assert resolve_case_input(poisoned, "http://127.0.0.1:1") == poisoned["input"]
    assert poisoned.get("needs_poisoned_index") is True
    assert (
        poisoned.get("fixture") is not None
    )  # has a fixture key, but no {fixture_url}


def test_golden_case_raises_on_an_unknown_id() -> None:
    with pytest.raises(KeyError):
        golden_case("no-such-case")
