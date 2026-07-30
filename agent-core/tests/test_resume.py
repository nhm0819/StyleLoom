"""Resuming a run partway through.

A stage that finishes writes its artifact to disk before the next one runs, which
is what makes a mid-run failure recoverable at all: `resume_run` reopens what
already ran and re-enters the plan at the point that broke, instead of redoing
work that already produced a correct result and was billed for it.
"""

from __future__ import annotations

import json

import pytest
from styleloom_core import RunInputs, RunStatus, resume_run, resume_session, run_once
from styleloom_core.errors import NotFoundError, ToolError
from styleloom_core.schema import AssembleResult, RenderResult
from styleloom_core.tools.registry import REGISTRY

INPUT = "겨울철 건조한 피부 관리 루틴 총정리"


def _break_stage(name: str, error: Exception):
    """Context manager-free swap: register a version of tool `name` that raises,
    and hand back a restore function.

    Registering a real, correctly-shaped ToolSpec rather than monkeypatching the
    module function, because the registry -- not the module attribute -- is what
    `execute` actually calls through.
    """
    real = REGISTRY[name]

    def boom(ctx, session):
        raise error

    REGISTRY[name] = real.__class__(name=name, reads=real.reads, writes=real.writes, fn=boom)
    return real


def _restore(name: str, real) -> None:
    REGISTRY[name] = real


def test_a_run_that_fails_after_render_can_resume_from_assemble(ctx, style):
    """The scenario this feature exists for: render finished and was billed for,
    assemble broke on something unrelated, and re-rendering would waste it."""
    real = _break_stage("assemble", FileNotFoundError("shots/win_00.mp4 missing"))
    try:
        failed = run_once(ctx, style.style_id, RunInputs(text=INPUT))
    finally:
        _restore("assemble", real)

    assert failed.status is RunStatus.FAILED
    assert failed.stage == "assemble"
    run_dir = ctx.runs.dir_for(failed.run_id)
    assert (run_dir / "render.json").exists()
    assert not (run_dir / "assemble.json").exists()

    resumed = resume_run(ctx, failed.run_id)

    assert resumed.status is RunStatus.DONE, resumed.error
    assert resumed.run_id == failed.run_id
    assert (run_dir / "assemble.json").exists()
    assert (run_dir / "final.mp4").exists()


def test_resume_does_not_redo_the_stages_before_it(ctx, style):
    """The point of resuming rather than rerunning: earlier stages must not be
    called a second time."""
    real = _break_stage("assemble", FileNotFoundError("x"))
    try:
        failed = run_once(ctx, style.style_id, RunInputs(text=INPUT))
    finally:
        _restore("assemble", real)

    calls: list[str] = []
    watched = ("ingest", "casting", "outline", "hook", "storyboard", "keyframe", "render")
    originals = {name: REGISTRY[name] for name in watched}
    for name, spec in originals.items():
        def wrapped(ctx, session, n=name, f=spec.fn):
            calls.append(n)
            return f(ctx, session)

        REGISTRY[name] = spec.__class__(
            name=name, reads=spec.reads, writes=spec.writes, fn=wrapped
        )
    try:
        resume_run(ctx, failed.run_id)
    finally:
        REGISTRY.update(originals)

    assert calls == [], f"stages re-ran that should have been skipped: {calls}"


def test_resume_defaults_to_the_stage_the_run_failed_at(ctx, style):
    real = _break_stage("assemble", RuntimeError("boom"))
    try:
        failed = run_once(ctx, style.style_id, RunInputs(text=INPUT))
    finally:
        _restore("assemble", real)

    _, plan = resume_session(ctx, failed.run_id)
    assert plan.steps[0] == "assemble"


def test_from_stage_can_redo_an_already_succeeded_stage(ctx, style):
    """Naming a stage explicitly re-runs it even though it did not fail -- for a
    clip that rendered fine but turned out wrong, say."""
    done = run_once(ctx, style.style_id, RunInputs(text=INPUT))
    assert done.status is RunStatus.DONE

    run_dir = ctx.runs.dir_for(done.run_id)
    before = (run_dir / "assemble.json").read_text(encoding="utf-8")
    (run_dir / "assemble.json").write_text("garbage", encoding="utf-8")

    resumed = resume_run(ctx, done.run_id, from_stage="assemble")

    assert resumed.status is RunStatus.DONE, resumed.error
    after = (run_dir / "assemble.json").read_text(encoding="utf-8")
    assert after != "garbage"
    assert json.loads(after) == json.loads(before)


def test_plan_steps_survive_a_second_resume_from_further_along(ctx, style):
    """Regression: `execute` used to overwrite `plan_steps` on every call, so
    after one resume it held only the remaining stages. A second failure further
    along would then record a plan that had already forgotten `render` was ever
    part of this run -- and a later `--from-stage render`, even though
    render.json still exists, would be refused as not part of the plan."""
    real = _break_stage("assemble", FileNotFoundError("x"))
    try:
        failed = run_once(ctx, style.style_id, RunInputs(text=INPUT))
    finally:
        _restore("assemble", real)

    real_qc = _break_stage("qc", RuntimeError("boom"))
    try:
        second = resume_run(ctx, failed.run_id)
    finally:
        _restore("qc", real_qc)
    assert second.status is RunStatus.FAILED
    assert second.stage == "qc"

    record = ctx.runs.load(failed.run_id)
    assert "render" in record.plan_steps
    assert "assemble" in record.plan_steps

    # And render's artifact is still reachable through resume_session, precisely
    # because plan_steps was not shrunk out from under it.
    _, plan = resume_session(ctx, failed.run_id, from_stage="render")
    assert plan.steps[0] == "render"


def test_a_completed_run_clears_the_error_from_an_earlier_failed_attempt(ctx, style):
    """Regression: `finish()` used to leave a stale `error` string on the record,
    so a run that failed and was then resumed to completion showed status=DONE
    next to an error message from the attempt before it."""
    real = _break_stage("assemble", FileNotFoundError("x"))
    try:
        failed = run_once(ctx, style.style_id, RunInputs(text=INPUT))
    finally:
        _restore("assemble", real)
    assert failed.error

    resumed = resume_run(ctx, failed.run_id)
    assert resumed.status is RunStatus.DONE
    assert resumed.error is None


def test_resuming_an_unknown_run_id_is_a_clean_error(ctx):
    with pytest.raises(NotFoundError):
        resume_run(ctx, "no-such-run")


def test_naming_a_stage_outside_the_recorded_plan_is_refused(ctx, style):
    real = _break_stage("assemble", RuntimeError("boom"))
    try:
        failed = run_once(ctx, style.style_id, RunInputs(text=INPUT))
    finally:
        _restore("assemble", real)

    with pytest.raises(ToolError, match="not a step"):
        resume_run(ctx, failed.run_id, from_stage="not_a_real_stage")


def test_a_run_with_no_recorded_plan_is_inferred_not_refused(ctx, style):
    """Runs from before `plan_steps` existed still have every artifact on disk,
    which is the whole premise of keeping them as readable files. Refusing them
    over a missing bookkeeping field threw away a finished, already-billed
    render."""
    real = _break_stage("assemble", FileNotFoundError("x"))
    try:
        failed = run_once(ctx, style.style_id, RunInputs(text=INPUT))
    finally:
        _restore("assemble", real)

    record = ctx.runs.load(failed.run_id)
    record.plan_steps = []
    ctx.runs.save(record)

    resumed = resume_run(ctx, failed.run_id)
    assert resumed.status is RunStatus.DONE, resumed.error
    # And the inferred plan is written back, so a later resume needs no inference.
    assert ctx.runs.load(failed.run_id).plan_steps


def test_inference_only_includes_optional_stages_that_left_a_file(ctx, style, settings, sink):
    """An optional stage before the resume point is in the inferred plan only if
    its artifact is on disk. Guessing it in would make `seed_from_disk` reach for
    a file that was never written."""
    from styleloom_core import build_context

    no_keyframe = build_context(
        settings.model_copy(update={"use_first_frame": False}), events=sink
    )
    real = _break_stage("assemble", FileNotFoundError("x"))
    try:
        failed = run_once(no_keyframe, style.style_id, RunInputs(text=INPUT))
    finally:
        _restore("assemble", real)
    assert not (no_keyframe.runs.dir_for(failed.run_id) / "keyframe.json").exists()

    record = no_keyframe.runs.load(failed.run_id)
    record.plan_steps = []
    no_keyframe.runs.save(record)

    _, plan = resume_session(no_keyframe, failed.run_id)
    assert "keyframe" not in plan.steps
    assert resume_run(no_keyframe, failed.run_id).status is RunStatus.DONE


def test_resuming_past_a_deleted_artifact_names_the_missing_file(ctx, style):
    """If a stage's own file is gone -- not just the one that failed -- the error
    has to say which artifact is missing rather than surfacing as a raw
    FileNotFoundError from deep inside whichever tool tries to read it."""
    real = _break_stage("assemble", FileNotFoundError("x"))
    try:
        failed = run_once(ctx, style.style_id, RunInputs(text=INPUT))
    finally:
        _restore("assemble", real)

    (ctx.runs.dir_for(failed.run_id) / "storyboard.json").unlink()

    with pytest.raises(ToolError, match="storyboard.*missing"):
        resume_run(ctx, failed.run_id)


def test_the_resumed_result_is_identical_in_shape_to_an_uninterrupted_run(ctx, style):
    """Resuming must not change what the finished artifacts look like -- only
    when they were produced."""
    real = _break_stage("assemble", FileNotFoundError("x"))
    try:
        failed = run_once(ctx, style.style_id, RunInputs(text=INPUT))
    finally:
        _restore("assemble", real)
    resumed = resume_run(ctx, failed.run_id)

    run_dir = ctx.runs.dir_for(failed.run_id)
    assembled = AssembleResult.model_validate_json(
        (run_dir / "assemble.json").read_text(encoding="utf-8")
    )
    rendered = RenderResult.model_validate_json(
        (run_dir / "render.json").read_text(encoding="utf-8")
    )
    assert assembled.final_video.exists()
    assert len(rendered.segments) > 0
    assert resumed.artifacts.get("final_video") == str(assembled.final_video)
