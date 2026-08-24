"""`uses_refusal_appropriateness`.

Offline: the selection rule is dataset-field-driven and pure, so this file
never touches a live model -- matches `tests/test_e2e_check_expects.py`'s
own shape for `check_expects`.
"""

from __future__ import annotations

from tests.test_e2e import uses_refusal_appropriateness


def test_true_only_when_expects_out_of_scope_is_true() -> None:
    assert uses_refusal_appropriateness({"expects": {"out_of_scope": True}})


def test_false_when_expects_is_absent() -> None:
    assert not uses_refusal_appropriateness({})


def test_false_when_expects_out_of_scope_is_false_or_absent() -> None:
    assert not uses_refusal_appropriateness({"expects": {}})
    assert not uses_refusal_appropriateness({"expects": {"out_of_scope": False}})


def test_false_for_a_refuses_injection_case_not_also_out_of_scope() -> None:
    """The three `refuses_injection` cases (injection, jailbreak, poisoned
    KB) are expected to engage with an in-scope task while resisting the
    attack -- not to decline outright, so they must not pick up the
    refusal metric merely for sharing the `expects` field's presence."""
    assert not uses_refusal_appropriateness({"expects": {"refuses_injection": True}})
