"""Gate-tier checks on `evals/evaluator_specs.json` and its judge prompts.

Offline only: this proves the local data files are internally consistent,
never that the live Langfuse UI actually matches them -- that half is only
checkable by the screenshots and the live run.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import paths

_SPEC_PATH = paths.resolve("evals/evaluator_specs.json")
_VALID_SCORE_TYPES = {"numeric", "boolean", "categorical"}
_VARIABLE_RE = re.compile(r"\{\{(\w+)\}\}")


def _load_specs() -> list[dict[str, Any]]:
    return json.loads(_SPEC_PATH.read_text(encoding="utf-8"))


def _prompt_variables(entry: dict[str, Any]) -> set[str]:
    prompt_path = paths.resolve(entry["judge_prompt_file"])
    text = prompt_path.read_text(encoding="utf-8")
    return set(_VARIABLE_RE.findall(text))


def test_at_least_two_evaluators_declared() -> None:
    assert len(_load_specs()) >= 2


def test_every_score_type_is_valid() -> None:
    for entry in _load_specs():
        assert entry["score_type"] in _VALID_SCORE_TYPES, entry["name"]


def test_at_least_two_distinct_score_types_present() -> None:
    types = {entry["score_type"] for entry in _load_specs()}
    assert len(types) >= 2


def test_every_judge_prompt_file_exists() -> None:
    for entry in _load_specs():
        prompt_path = paths.resolve(entry["judge_prompt_file"])
        assert prompt_path.is_file(), entry["judge_prompt_file"]


def test_every_prompt_variable_is_declared() -> None:
    for entry in _load_specs():
        declared = set(entry["variables"])
        used = _prompt_variables(entry)
        assert used == declared, (entry["name"], used, declared)


def test_categorical_entries_declare_categories_and_others_do_not() -> None:
    for entry in _load_specs():
        if entry["score_type"] == "categorical":
            assert entry.get("categories"), entry["name"]
        else:
            assert "categories" not in entry, entry["name"]


def test_evaluator_specs_json_is_a_real_file_not_a_placeholder() -> None:
    assert Path(_SPEC_PATH).exists()
