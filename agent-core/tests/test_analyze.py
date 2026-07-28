"""Style extraction must recover the fixture's measurable properties."""

from __future__ import annotations

import shutil
from pathlib import Path

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


def test_bare_filename_resolves_against_the_uploads_dir(ctx, reference_video):
    """The point of the uploads dir: a filename works from any cwd."""
    ctx.settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(reference_video, ctx.settings.uploads_dir / "ref01.mp4")

    style = extract_style(ctx, "byname", [Path("ref01.mp4")])
    assert style.source_refs == ["ref01.mp4"]


def test_path_outside_the_uploads_dir_still_works(ctx, reference_video):
    """Absolute paths are the fallback, not a rejected input."""
    assert reference_video.is_absolute()
    style = extract_style(ctx, "bypath", [reference_video])
    assert style.source_refs == [reference_video.name]


def test_missing_reference_is_reported_clearly(ctx, tmp_path):
    with pytest.raises(ToolError, match="not found"):
        extract_style(ctx, "x", [tmp_path / "nope.mp4"])


def test_missing_reference_names_every_place_it_looked(ctx):
    """A bare filename fails in two places; the message has to say both."""
    with pytest.raises(ToolError) as exc:
        extract_style(ctx, "x", [Path("nope.mp4")])
    message = str(exc.value)
    assert str(ctx.settings.uploads_dir / "nope.mp4") in message
    assert "nope.mp4" in message


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
