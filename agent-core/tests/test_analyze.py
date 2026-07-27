"""Style extraction must recover the fixture's measurable properties."""

from __future__ import annotations

import pytest
from styleloom_core import extract_style
from styleloom_core.errors import ToolError
from styleloom_core.media import estimate_bpm, probe_video


def test_probe_recovers_cut_structure(reference_video, ref_spec):
    m = probe_video(reference_video)
    assert m["shot_count"] == pytest.approx(ref_spec["shot_count"], abs=1)
    assert m["avg_shot_sec"] == pytest.approx(ref_spec["segment_sec"], abs=0.3)


def test_extract_style_uses_measured_numbers_not_model_guesses(ctx, reference_video, ref_spec):
    style = extract_style(ctx, "measured", [reference_video])
    assert style.pacing.avg_shot_sec == pytest.approx(ref_spec["segment_sec"], abs=0.3)
    assert style.total_duration == pytest.approx(ref_spec["total_sec"], abs=0.5)
    assert 0.0 <= style.look.saturation <= 1.0
    assert style.source_refs == [reference_video.name]


def test_extract_style_names_qualitative_attributes(ctx, reference_video):
    style = extract_style(ctx, "named", [reference_video])
    assert style.look.grade
    assert style.camera.moves
    assert style.audio.voice_tone


def test_hook_window_cut_count_is_measured(ctx, reference_video):
    """The reference cuts every 1.2s, so a 3s window holds about two cuts."""
    style = extract_style(ctx, "hookwin", [reference_video])
    assert 1 <= style.hook_style.cut_count <= 3


def test_missing_reference_is_reported_clearly(ctx, tmp_path):
    with pytest.raises(ToolError, match="not found"):
        extract_style(ctx, "x", [tmp_path / "nope.mp4"])


def test_no_reference_is_rejected(ctx):
    with pytest.raises(ToolError, match="at least one"):
        extract_style(ctx, "x", [])


@pytest.mark.parametrize(
    "interval,expected",
    [(0.5, 120.0), (1.2, 100.0), (0.0, 120.0)],
)
def test_bpm_folds_into_a_musical_range(interval, expected):
    bpm = estimate_bpm(interval)
    assert 70 <= bpm <= 160
    assert bpm == pytest.approx(expected, abs=1.0)
