"""The render path: one generation per window of cuts.

There used to be two strategies here and these tests compared them. `per_shot` is
gone, so what is left is the consequence of that: the cuts land inside the model's
output rather than on file boundaries, which makes the requested timeline a request
rather than a guarantee. These tests pin down the windowing arithmetic, the caption
placement that follows from it, and the drift check that is now the only thing
standing between a delivered video and a silently wrong one.
"""

from __future__ import annotations

import pytest
from styleloom_core import RunInputs, RunStatus, run_once
from styleloom_core.errors import ToolError
from styleloom_core.media import probe_video
from styleloom_core.providers import MotionShot
from styleloom_core.schema import (
    AssembleResult,
    Casting,
    ClipSegment,
    QCReport,
    RenderResult,
    Shot,
    Storyboard,
)
from styleloom_core.tools import assemble as assemble_tool
from styleloom_core.tools import qc as qc_tool
from styleloom_core.tools import render as render_tool

INPUT = "모공 관리 3일차 후기"


def make_shots(durations: list[float]) -> list[Shot]:
    return [
        Shot(
            index=i,
            role="context",
            duration_sec=d,
            shot_size="CU",
            camera_move="static",
            action="a",
            caption=f"자막 {i}",
            scene_prompt=f"frame {i}",
            motion_prompt=f"motion {i}",
        )
        for i, d in enumerate(durations)
    ]


# --- the capability is now a requirement ------------------------------------- #


def test_an_incapable_provider_is_refused_up_front(ctx, style, monkeypatch):
    """Carrying several cuts in one request is no longer optional, so a provider
    that cannot do it has to be refused rather than sent one cut per request and
    billed for the rest."""
    monkeypatch.setattr(type(ctx.video), "supports_multi_shot", property(lambda self: False))

    from styleloom_core.runner import prepare_session

    run = prepare_session(ctx, style.style_id, RunInputs(text=INPUT))
    run.artifacts["storyboard"] = Storyboard(
        style_id=style.style_id, hook_text="h", hook_archetype="question",
        shots=make_shots([1.2, 1.5]),
    )
    run.artifacts["casting"] = Casting(
        creator={"id": "c"}, setting={"id": "s"}  # type: ignore[arg-type]
    )
    with pytest.raises(ToolError, match="several cuts in one generation"):
        render_tool.render(ctx, run)


# --- window splitting -------------------------------------------------------- #


def test_shots_are_grouped_into_windows_that_fit():
    shots = make_shots([4.0, 4.0, 4.0, 4.0, 4.0])
    windows = render_tool.split_windows(shots, window_sec=15.0)
    assert [len(w) for w in windows] == [3, 2]
    for window in windows:
        assert sum(s.duration_sec for s in window) <= 15.0


def test_a_cut_is_never_split_across_two_generations():
    shots = make_shots([2.0, 2.0, 2.0])
    windows = render_tool.split_windows(shots, window_sec=3.0)
    flattened = [s.index for w in windows for s in w]
    assert flattened == [0, 1, 2]
    assert all(len(w) >= 1 for w in windows)


def test_no_window_ceiling_means_one_generation():
    windows = render_tool.split_windows(make_shots([1.0, 1.0]), window_sec=0)
    assert len(windows) == 1


def test_a_single_overlong_shot_still_gets_its_own_window():
    windows = render_tool.split_windows(make_shots([30.0, 1.0]), window_sec=15.0)
    assert [len(w) for w in windows] == [1, 1]


# --- provider ---------------------------------------------------------------- #


def test_the_offline_provider_really_renders_a_multi_cut_clip(ctx, tmp_path):
    """Not a stub. If the mock could not do this, the whole multi_shot path would be
    unreachable offline and therefore untested."""
    assert ctx.video.supports_multi_shot
    shots = [MotionShot(prompt=f"shot {i}", duration=d) for i, d in enumerate([1.2, 1.5, 1.3])]

    clip = ctx.video.generate_sequence(shots, tmp_path / "seq.mp4")
    measured = probe_video(clip)
    assert measured["duration"] == pytest.approx(4.0, abs=0.3)
    # Visually distinct shots, or the cuts would not exist to be detected.
    assert len(measured["cuts"]) >= 2


# --- segments are the shared shape ------------------------------------------- #


def test_the_segments_cover_every_shot_in_order(settings, sink, style, ctx):
    """Windows are an implementation detail of billing; no cut may be lost in them."""
    record = run_once(ctx, style.style_id, RunInputs(text=INPUT))
    assert record.status is RunStatus.DONE

    run_dir = ctx.runs.dir_for(record.run_id)
    result = RenderResult.model_validate_json(
        (run_dir / "render.json").read_text(encoding="utf-8")
    )
    board = Storyboard.model_validate_json(
        (run_dir / "storyboard.json").read_text(encoding="utf-8")
    )
    covered = [i for seg in result.segments for i in seg.shot_indices]
    assert covered == [s.index for s in board.shots]


def test_windowing_uses_fewer_generations_than_cuts(settings, sink, style, ctx):
    record = run_once(ctx, style.style_id, RunInputs(text=INPUT))
    assert record.status is RunStatus.DONE

    result = RenderResult.model_validate_json(
        (ctx.runs.dir_for(record.run_id) / "render.json").read_text(encoding="utf-8")
    )
    shots = sum(len(seg.shot_indices) for seg in result.segments)
    assert shots > len(result.segments), "the point of multi_shot is fewer calls"
    assert any(seg.is_multi_shot for seg in result.segments)


def test_the_run_still_delivers_a_playable_video(settings, sink, style, ctx):
    record = run_once(ctx, style.style_id, RunInputs(text=INPUT))
    run_dir = ctx.runs.dir_for(record.run_id)
    final = run_dir / "final.mp4"
    assert final.exists()

    # Against the storyboard rather than style.total_duration: the outline enforces
    # a minimum body budget, so a very short reference produces a longer video by
    # design. What multi_shot has to preserve is the length it was asked for.
    board = Storyboard.model_validate_json(
        (run_dir / "storyboard.json").read_text(encoding="utf-8")
    )
    assert probe_video(final)["duration"] == pytest.approx(board.total_duration, abs=0.5)


# --- the cut timeline -------------------------------------------------------- #


def test_cut_timeline_runs_across_segment_boundaries():
    result = RenderResult(
        segments=[
            ClipSegment(path="a.mp4", shot_indices=[0, 1], requested_durations=[1.2, 1.5]),
            ClipSegment(path="b.mp4", shot_indices=[2], requested_durations=[2.0]),
        ],
    )
    # A segment join is a cut too; the final boundary is the end of the video.
    assert result.cut_timeline() == [1.2, 2.7]


def test_drift_is_zero_on_an_exact_match():
    assert qc_tool.cut_drift([1.2, 2.7], [1.2, 2.7]) == 0.0


def test_drift_measures_a_late_cut():
    assert qc_tool.cut_drift([1.0, 2.0], [1.2, 2.2]) == pytest.approx(0.2)


def test_drift_reports_a_full_miss_rather_than_a_pass():
    """No detected cuts must not read as perfect timing."""
    assert qc_tool.cut_drift([1.0, 3.0], []) == 3.0


def test_drift_uses_nearest_neighbour_not_positional_pairing(ctx):
    """Cut detection misses low-contrast cuts and invents none, so pairing by index
    would punish a correct timeline for one missed detection."""
    positional_would_be_large = qc_tool.cut_drift([1.0, 2.0, 3.0], [1.0, 3.0])
    assert positional_would_be_large < 1.0


def test_qc_checks_cut_timing(settings, sink, style, ctx):
    record = run_once(ctx, style.style_id, RunInputs(text=INPUT))
    qc_path = ctx.runs.dir_for(record.run_id) / "qc_report.json"
    report = QCReport.model_validate_json(qc_path.read_text(encoding="utf-8"))
    drift = {c.name: c for c in report.checks}["cut_timing_drift"]
    # ffmpeg cuts exactly where told and the offline sequence renderer is ffmpeg,
    # so drift offline is near zero. On a real endpoint this is the number that
    # decides whether the output is usable at all -- it cannot be verified here,
    # only reported.
    assert drift.passed, f"unexpected offline drift: {drift.actual}"


# --- captions ---------------------------------------------------------------- #


def test_captions_get_one_window_per_cut_inside_a_segment(style, brief):
    board = Storyboard(
        style_id="s",
        hook_text="h",
        hook_archetype="question",
        shots=make_shots([1.2, 1.5, 1.3]),
    )
    segment = ClipSegment(
        path="a.mp4", shot_indices=[0, 1, 2], requested_durations=[1.2, 1.5, 1.3]
    )
    cues = assemble_tool.cues_for(segment, board)

    assert [c.text for c in cues] == ["자막 0", "자막 1", "자막 2"]
    assert cues[0].start == 0.0
    assert cues[1].start == pytest.approx(1.2)
    assert cues[2].end == pytest.approx(4.0)


def test_a_single_shot_segment_yields_one_full_length_cue():
    board = Storyboard(
        style_id="s", hook_text="h", hook_archetype="question", shots=make_shots([1.4])
    )
    segment = ClipSegment(path="a.mp4", shot_indices=[0], requested_durations=[1.4])
    cues = assemble_tool.cues_for(segment, board)
    assert len(cues) == 1
    assert (cues[0].start, cues[0].end) == (0.0, pytest.approx(1.4))


def test_uncaptioned_shots_are_skipped_not_blanked():
    shots = make_shots([1.0, 1.0])
    shots[0].caption = ""
    board = Storyboard(style_id="s", hook_text="h", hook_archetype="question", shots=shots)
    segment = ClipSegment(path="a.mp4", shot_indices=[0, 1], requested_durations=[1.0, 1.0])
    cues = assemble_tool.cues_for(segment, board)
    assert [c.text for c in cues] == ["자막 1"]
    assert cues[0].start == pytest.approx(1.0)


def test_every_caption_is_burned(settings, sink, style, ctx):
    record = run_once(ctx, style.style_id, RunInputs(text=INPUT))
    run_dir = ctx.runs.dir_for(record.run_id)

    result = AssembleResult.model_validate_json(
        (run_dir / "assemble.json").read_text(encoding="utf-8")
    )
    board = Storyboard.model_validate_json(
        (run_dir / "storyboard.json").read_text(encoding="utf-8")
    )
    expected = len([s for s in board.shots if s.caption.strip()])
    assert result.captioned_count == expected


# --- failure isolation differs by mode --------------------------------------- #


def test_a_failed_window_costs_every_cut_in_it(settings, sink, style, ctx, monkeypatch):
    """The honest downside of paying the duration floor once per request instead of
    once per cut: a failure loses the whole window."""

    def boom(*args, **kwargs):
        raise RuntimeError("endpoint refused the sequence")

    monkeypatch.setattr(ctx.video, "generate_sequence", boom)
    record = run_once(ctx, style.style_id, RunInputs(text=INPUT))

    assert record.status is RunStatus.FAILED
    assert record.stage == "render"
    assert "all" in (record.error or "")


def test_windows_are_capped_by_shot_count_not_only_by_duration():
    """Fourteen 0.76s cuts total under 11s and still exceed Kling's six-shot cap.
    The endpoint rejects the whole request rather than truncating it, so the two
    limits have to bind independently."""
    shots = make_shots([0.76] * 14)
    windows = render_tool.split_windows(shots, window_sec=15.0, max_shots=6)

    assert [len(w) for w in windows] == [6, 6, 2]
    assert sum(len(w) for w in windows) == 14, "no cut may be dropped"
    assert [s.index for w in windows for s in w] == [s.index for s in shots]


def test_a_count_cap_alone_still_splits_when_no_window_is_declared():
    windows = render_tool.split_windows(make_shots([1.0] * 7), window_sec=0, max_shots=6)
    assert [len(w) for w in windows] == [6, 1]


# --- planning on the endpoint's grid ----------------------------------------- #


def test_the_storyboard_asks_only_for_lengths_the_endpoint_can_render(ctx, style, sink):
    """No silent rounding. The storyboard used to divide beats evenly and hand over
    0.76s cuts to an endpoint whose shortest renderable cut is 1s; the request
    quantised without complaint and the delivered video ran ~40% long."""
    record = run_once(ctx, style.style_id, RunInputs(text=INPUT))
    board = Storyboard.model_validate_json(
        (ctx.runs.dir_for(record.run_id) / "storyboard.json").read_text(encoding="utf-8")
    )
    requested = [s.duration_sec for s in board.shots]
    assert requested == ctx.video.plan_shot_durations(requested)


def test_the_total_survives_the_snapping(ctx, style):
    """Rounding each beat on its own accumulates. The error is carried forward, so
    the total stays within half a unit of the reference however many beats there are."""
    record = run_once(ctx, style.style_id, RunInputs(text=INPUT))
    board = Storyboard.model_validate_json(
        (ctx.runs.dir_for(record.run_id) / "storyboard.json").read_text(encoding="utf-8")
    )
    unit = ctx.video.shot_billed_duration(1e-6)
    total = sum(s.duration_sec for s in board.shots)
    assert abs(total - style.total_duration) <= unit


def test_a_pacing_floor_widens_the_tolerance_by_exactly_the_forced_gap():
    """Accepting the floor, not abandoning the check. A style whose cuts are longer
    than the floor is measured as strictly as before."""
    from styleloom_core.tools.qc import tolerance_for

    strict = tolerance_for("avg_shot_sec", 2.0, min_cut=1.0)
    assert strict == tolerance_for("avg_shot_sec", 2.0)

    widened = tolerance_for("avg_shot_sec", 0.76, min_cut=1.0)
    assert widened == pytest.approx((1.0 - 0.76) + 1.0 * 0.15, abs=0.001)
    assert widened > tolerance_for("avg_shot_sec", 0.76)


def test_a_pacing_floor_is_reported_rather_than_absorbed(ctx, style, sink):
    """The check passing must not read as "the pacing was reproduced".

    Pacing is set below the floor explicitly rather than taken from the fixture,
    whose 1.21s cuts sit above it -- the point of the test is the case where the
    endpoint cannot deliver what the reference does.
    """
    style.pacing.avg_shot_sec = 0.6
    ctx.styles.save(style)
    assert ctx.video.shot_billed_duration(1e-6) > style.pacing.avg_shot_sec

    run_once(ctx, style.style_id, RunInputs(text=INPUT))

    messages = [e.message for e in sink.events]
    assert any("does not mean the pacing was reproduced" in m for m in messages), messages
