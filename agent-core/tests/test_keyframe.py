"""The keyframe stage.

This stage existed once, was removed, and is back in a different shape. The
critique that removed it is recorded in docs/TOOL_RATIONALE.md and it was correct:
every cut generated its own keyframe from its own text, so nothing was held across
cuts and the extra call bought only single-shot composition control.

So the assertions here are almost entirely about the difference. Not "does a JPEG
appear" -- that is easy and was true of the old design too -- but: is there exactly
one anchor, does every frame descend from it, and do the frames land on the keys
render will actually look up. The last one is the quiet one: keys that do not line
up means every request falls back to text-to-video *after* paying for the images,
and the video still comes out.
"""

from __future__ import annotations

import pytest
from styleloom_core.errors import ToolError
from styleloom_core.schema import (
    Casting,
    Keyframes,
    RunInputs,
    RunRecord,
    Shot,
    Storyboard,
)
from styleloom_core.session import RunSession
from styleloom_core.tools import ingest as ingest_tool
from styleloom_core.tools.keyframe import anchor_prompt, frame_prompt, keyframe, lead_shots
from styleloom_core.tools.render import plan_windows


def _board(count: int, duration: float = 1.2) -> Storyboard:
    return Storyboard(
        style_id="fixture",
        hook_text="hook",
        hook_archetype="question",
        shots=[
            Shot(
                index=i,
                role="body",
                duration_sec=duration,
                shot_size="CU",
                camera_move="static",
                action=f"action {i}",
                caption="",
                scene_prompt=f"scene {i}",
                motion_prompt="motion",
            )
            for i in range(count)
        ],
    )


def _session(ctx, style) -> RunSession:
    return RunSession(
        record=RunRecord(run_id="kf1", style_id=style.style_id),
        inputs=RunInputs(),
        store=ctx.runs,
    )


def _run(ctx, style, casting, board) -> Keyframes:
    session = _session(ctx, style)
    session.artifacts["style"] = style
    session.artifacts["casting"] = casting
    session.artifacts["storyboard"] = board
    # The frame prompts name what the run is about, so keyframe reads the brief.
    session.artifacts["brief"] = ingest_tool.ingest(ctx, session)
    return keyframe(ctx, session)


@pytest.fixture
def casting() -> Casting:
    from styleloom_core.schema import CastChoice

    return Casting(
        creator=CastChoice(id="c1", label="c", prompt="a woman in her twenties"),
        setting=CastChoice(id="s1", label="s", prompt="a sunlit kitchen"),
    )


# --- the anchor ------------------------------------------------------------ #


def test_there_is_exactly_one_anchor_for_the_whole_run(ctx, style, casting):
    """The property the previous design lacked. Twelve cuts, one identity."""
    frames = _run(ctx, style, casting, _board(12))
    assert frames.anchor is not None and frames.anchor.is_file()


def test_the_anchor_prompt_asks_for_a_reference_portrait_not_a_shot(style, casting):
    """This frame never appears in the video, so composition is irrelevant and a
    neutral unambiguous subject is what a reference image should be."""
    prompt = anchor_prompt(style, casting)
    assert "a woman in her twenties" in prompt
    assert "a sunlit kitchen" in prompt
    assert "facing camera" in prompt


def test_a_frame_prompt_defers_to_the_anchor_for_identity(style):
    """Restating the creator here would compete with the reference image rather
    than reinforce it, and where they disagree text tends to win."""
    prompt = frame_prompt(style, _board(1).shots[0])
    assert "Same person, same location" in prompt
    assert "action 0" in prompt


# --- the keys render will look up ----------------------------------------- #


def test_frames_land_on_window_leads_not_on_every_cut(ctx, style, casting):
    """One frame per generation, not per cut. This is the cost argument as well as
    the correctness one: a 30s video buys three or four images, not twenty."""
    board = _board(14, duration=0.8)
    frames = _run(ctx, style, casting, board)

    windows = plan_windows(ctx, board)
    assert sorted(frames.frames) == sorted(w[0].index for w in windows)
    assert len(frames.frames) < len(board.shots)


def test_lead_shots_uses_the_same_packing_rule_as_render(ctx):
    """Literally the same function, not a second copy of the rule. If the two ever
    disagreed, every request would fall back to text-to-video after the images had
    been billed."""
    board = _board(9, duration=2.0)
    leads = lead_shots(ctx, board)
    assert [s.index for s in leads] == [w[0].index for w in plan_windows(ctx, board)]


# --- refusal rather than a silent skip ------------------------------------ #


def test_a_provider_that_cannot_take_a_frame_is_refused_not_skipped(
    ctx, style, casting, monkeypatch
):
    """A silently skipped keyframe stage renders a perfectly good video in which
    the creator's face changes between cuts, which is the failure this stage
    exists to prevent. So it raises, the way render refuses multi_shot."""
    monkeypatch.setattr(
        type(ctx.video), "supports_first_frame", property(lambda self: False)
    )
    with pytest.raises(ToolError, match="cannot take a start frame"):
        _run(ctx, style, casting, _board(3))


# --- fallback ------------------------------------------------------------- #


def test_an_absent_keyframe_artifact_renders_from_text(ctx, style, casting):
    """`render` reads the artifact defensively, so a plan without this stage still
    produces a video rather than failing validation."""
    empty = Keyframes()
    assert empty.for_lead(0) is None
