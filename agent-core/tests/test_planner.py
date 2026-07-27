"""The plan is data, so it can be checked. These tests are that check."""

from __future__ import annotations

import pytest
from styleloom_core import build_plan
from styleloom_core.errors import PlanError
from styleloom_core.planner import STANDARD_STEPS, Plan
from styleloom_core.tools import registry


def test_standard_plan_validates():
    build_plan().validate()


def test_hook_runs_after_outline():
    """Ordering is load-bearing: a hook has to promise the outline's payoff, so
    generating it earlier would make it generic."""
    steps = list(STANDARD_STEPS)
    assert steps.index("outline") < steps.index("hook")


def test_reordering_is_rejected_before_anything_runs():
    bad = Plan(name="bad", steps=("ingest", "hook", "outline"))
    with pytest.raises(PlanError, match="reads"):
        bad.validate()


def test_unknown_tool_is_rejected():
    with pytest.raises(PlanError, match="unknown tool"):
        Plan(name="typo", steps=("ingest", "outlyne")).validate()


def test_duplicate_step_is_rejected():
    with pytest.raises(PlanError, match="twice"):
        Plan(name="dup", steps=("ingest", "ingest")).validate()


def test_empty_plan_is_rejected():
    with pytest.raises(PlanError, match="no steps"):
        Plan(name="empty", steps=()).validate()


def test_plan_without_preloaded_style_fails():
    """`style` is loaded by the runner rather than produced by a tool. If that
    ever stops happening, validation must catch it instead of the run dying at
    the outline stage."""
    with pytest.raises(PlanError, match="style"):
        build_plan().validate(available=frozenset())


def test_no_qc_plan_drops_only_qc():
    plan = build_plan(include_qc=False)
    plan.validate()
    assert "qc" not in plan.steps
    assert set(plan.steps) == set(STANDARD_STEPS) - {"qc"}


def test_every_registered_tool_declares_reachable_reads():
    """Guards against a new tool reading an artifact key nothing ever writes --
    a typo in `reads` would otherwise only surface at runtime."""
    writable = {s.writes for s in registry.REGISTRY.values() if s.writes} | {"style"}
    for spec in registry.REGISTRY.values():
        unknown = [key for key in spec.reads if key not in writable]
        assert not unknown, f"{spec.name} reads unproduced artifacts: {unknown}"
