"""Memory: the style asset, run records, and hook history."""

from __future__ import annotations

import pytest
from styleloom_core.errors import NotFoundError
from styleloom_core.schema import Choice, RunRecord, RunStatus, StyleSchema


def test_style_round_trip(ctx):
    original = StyleSchema(style_id="s1", notes="원본")
    ctx.styles.save(original)
    assert ctx.styles.load("s1").notes == "원본"
    assert ctx.styles.list_ids() == ["s1"]


def test_missing_style_names_itself(ctx):
    with pytest.raises(NotFoundError, match="nope"):
        ctx.styles.load("nope")


def test_run_records_list_newest_first_and_respect_limit(ctx):
    for i in range(5):
        ctx.runs.save(RunRecord(run_id=f"run_{i:02d}", style_id="s1"))
    ids = [r.run_id for r in ctx.runs.list_records(limit=3)]
    assert ids == ["run_04", "run_03", "run_02"]


def test_run_records_filter_by_style(ctx):
    ctx.runs.save(RunRecord(run_id="a", style_id="s1"))
    ctx.runs.save(RunRecord(run_id="b", style_id="s2"))
    assert [r.run_id for r in ctx.runs.list_records(style_id="s2")] == ["b"]


def test_history_returns_most_recent_first(ctx):
    for a in ["question", "reversal", "empathy"]:
        ctx.history.append("s1", Choice(run_id="r", kind="hook", value=a))
    assert ctx.history.recent_values("s1", "hook") == ["empathy", "reversal", "question"]


def test_history_honours_the_recency_window(ctx):
    for i in range(10):
        ctx.history.append("s1", Choice(run_id=f"r{i}", kind="hook", value=f"a{i}"))
    assert len(ctx.history.recent_values("s1", "hook")) == ctx.settings.hook_recency_window


def test_history_is_empty_for_an_unseen_style(ctx):
    assert ctx.history.recent_values("never_used", "hook") == []


def test_history_separates_kinds(ctx):
    """One log serves hook archetypes and casting choices, so a busy creator
    history must not crowd out the hook window or vice versa."""
    for i in range(6):
        ctx.history.append("s1", Choice(run_id=f"r{i}", kind="creator", value=f"c{i}"))
    ctx.history.append("s1", Choice(run_id="rh", kind="hook", value="question"))
    ctx.history.append("s1", Choice(run_id="rs", kind="setting", value="bathroom"))

    assert ctx.history.recent_values("s1", "hook") == ["question"]
    assert ctx.history.recent_values("s1", "setting") == ["bathroom"]
    assert ctx.history.recent_values("s1", "creator") == ["c5", "c4", "c3", "c2"]


def test_history_survives_a_truncated_line(ctx):
    """The tail read can land mid-line on a large file, and a crashed write can
    leave a partial record. Neither may take down the next run."""
    ctx.history.append("s1", Choice(run_id="r1", kind="hook", value="question"))
    path = ctx.history.path_for("s1")
    with path.open("a", encoding="utf-8") as fh:
        fh.write('{"run_id": "broken", "kin\n')
    ctx.history.append("s1", Choice(run_id="r2", kind="hook", value="reversal"))
    assert ctx.history.recent_values("s1", "hook") == ["reversal", "question"]


def test_run_status_transitions_persist(ctx):
    record = RunRecord(run_id="r1", style_id="s1")
    ctx.runs.save(record)
    ctx.runs.save(record.touch(status=RunStatus.DONE, stage="done", qc_score=0.9))
    reloaded = ctx.runs.load("r1")
    assert reloaded.status is RunStatus.DONE
    assert reloaded.qc_score == 0.9
