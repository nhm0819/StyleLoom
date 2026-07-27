"""The claim under test: different inputs through one unchanged system produce
different videos that still conform to the same style.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
from styleloom_core import RunInputs, RunStatus, build_plan, run_batch, run_once
from styleloom_core.errors import NotFoundError
from styleloom_core.events import EventKind
from styleloom_core.media import probe_video
from styleloom_core.planner import STANDARD_STEPS
from styleloom_core.schema import HookResult, QCReport, Storyboard

INPUTS = [
    "회사에서 아무도 안 알려주는 엑셀 단축키",
    "자취 3년차가 후회하는 가전 구매 순서",
    "러닝 첫 달에 무릎이 아픈 진짜 이유",
]


def test_single_run_produces_a_playable_video(ctx, style):
    record = run_once(ctx, style.style_id, RunInputs(text=INPUTS[0]))
    assert record.status is RunStatus.DONE, record.error

    final = ctx.runs.dir_for(record.run_id) / "final.mp4"
    assert final.exists() and final.stat().st_size > 0
    # Playable, not merely present: decoding it back is the actual check.
    assert probe_video(final)["duration"] > 0


def test_run_writes_one_inspectable_artifact_per_stage(ctx, style):
    """`data/runs/<id>/` is the deliverable, so a finished run has to be readable
    end to end without the code that produced it."""
    record = run_once(ctx, style.style_id, RunInputs(text=INPUTS[0]))
    run_dir = ctx.runs.dir_for(record.run_id)

    for name in (
        "run.json", "inputs.json", "style_used.json", "brief.json", "outline.json",
        "casting.json", "hook.json", "storyboard.json", "qc_report.json", "final.mp4",
    ):
        assert (run_dir / name).exists(), f"missing artifact: {name}"

    # And they parse back into their contracts.
    HookResult.model_validate_json((run_dir / "hook.json").read_text(encoding="utf-8"))
    Storyboard.model_validate_json((run_dir / "storyboard.json").read_text(encoding="utf-8"))
    QCReport.model_validate_json((run_dir / "qc_report.json").read_text(encoding="utf-8"))


def test_style_is_snapshotted_into_the_run(ctx, style):
    """style.json is meant to be hand-corrected between runs, so a run must record
    the version it actually used."""
    record = run_once(ctx, style.style_id, RunInputs(text=INPUTS[0]))
    snapshot = ctx.runs.dir_for(record.run_id) / "style_used.json"
    assert style.style_id in snapshot.read_text(encoding="utf-8")


def test_stages_run_in_the_planned_order(ctx, style, sink):
    run_once(ctx, style.style_id, RunInputs(text=INPUTS[0]))
    assert sink.stages() == list(STANDARD_STEPS)
    assert sink.kinds()[0] == EventKind.RUN_STARTED.value
    assert sink.kinds()[-1] == EventKind.RUN_FINISHED.value


def test_three_inputs_diverge_through_one_pipeline(ctx, style):
    records = run_batch(ctx, style.style_id, [RunInputs(text=t) for t in INPUTS])

    assert all(r.status is RunStatus.DONE for r in records), [r.error for r in records]
    # Different content, which is the point of the batch.
    assert len({r.hook_text for r in records}) == 3
    assert len({r.run_id for r in records}) == 3
    # Same style, which is the other half of the point.
    assert {r.style_id for r in records} == {style.style_id}


def test_batch_feeds_choice_history_forward(ctx, style):
    """Each run has to see what the previous one chose, or the recency penalty
    cannot do anything. This checks the wiring, not the resulting distribution."""
    run_batch(ctx, style.style_id, [RunInputs(text=t) for t in INPUTS])
    for kind in ("hook", "creator", "setting"):
        history = ctx.history.recent(style.style_id, kind, limit=10)
        assert len(history) == 3, f"{kind} history did not record every run"
        assert all(h.value for h in history)


def test_output_conforms_to_the_style_it_claims(ctx, style):
    record = run_once(ctx, style.style_id, RunInputs(text=INPUTS[0]))
    report = QCReport.model_validate_json(
        (ctx.runs.dir_for(record.run_id) / "qc_report.json").read_text(encoding="utf-8")
    )
    by_name = {c.name: c for c in report.checks}
    # Pacing and runtime are reproducible by construction. Colour grade is not,
    # because the offline provider makes no attempt to match it -- see the README
    # limitations. Asserting those would make QC meaningless.
    assert by_name["avg_shot_sec"].passed
    assert by_name["total_duration"].passed
    assert by_name["hook_window_shots"].passed


def test_run_records_the_hook_it_used(ctx, style):
    record = run_once(ctx, style.style_id, RunInputs(text=INPUTS[0]))
    board = Storyboard.model_validate_json(
        (ctx.runs.dir_for(record.run_id) / "storyboard.json").read_text(encoding="utf-8")
    )
    assert record.hook_text == board.hook_text
    assert record.hook_archetype == board.hook_archetype


def test_no_qc_plan_skips_qc_but_still_delivers(ctx, style):
    record = run_once(
        ctx, style.style_id, RunInputs(text=INPUTS[0]), plan=build_plan(include_qc=False)
    )
    assert record.status is RunStatus.DONE
    assert record.qc_score is None
    assert not (ctx.runs.dir_for(record.run_id) / "qc_report.json").exists()
    assert (ctx.runs.dir_for(record.run_id) / "final.mp4").exists()


def test_a_missing_style_fails_before_a_run_is_created(ctx):
    """Asking for something impossible raises; a failed render does not."""
    with pytest.raises(NotFoundError):
        run_once(ctx, "does_not_exist", RunInputs(text="x"))
    assert ctx.runs.list_records() == []


def test_a_failing_stage_is_recorded_rather_than_raised(ctx, style, sink, monkeypatch):
    """A batch must survive one bad input, so run_once reports failure in the
    record instead of propagating."""
    from styleloom_core.tools import render as render_tool

    def boom(provider, shot, out_dir, persona_ref=None):
        raise RuntimeError("provider exploded")

    monkeypatch.setattr(render_tool, "render_shot", boom)
    record = run_once(ctx, style.style_id, RunInputs(text=INPUTS[0]))

    assert record.status is RunStatus.FAILED
    assert record.stage == "render"
    assert "exploded" in (record.error or "")
    assert EventKind.RUN_FAILED.value in sink.kinds()
    # The stages before the failure still left their artifacts behind.
    assert (ctx.runs.dir_for(record.run_id) / "hook.json").exists()


def test_batch_continues_past_a_failed_input(ctx, style, monkeypatch):
    from styleloom_core.tools import registry

    original = registry.REGISTRY["ingest"]
    calls = {"n": 0}

    def flaky(ctx_, session):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("bad input")
        return original.fn(ctx_, session)

    # setitem, not assignment: the registry is process-global, so a test that
    # replaces a tool without restoring it corrupts every test after it.
    monkeypatch.setitem(registry.REGISTRY, "ingest", replace(original, fn=flaky))

    records = run_batch(ctx, style.style_id, [RunInputs(text=t) for t in INPUTS])
    statuses = [r.status for r in records]
    assert statuses.count(RunStatus.DONE) == 2
    assert statuses[1] is RunStatus.FAILED
