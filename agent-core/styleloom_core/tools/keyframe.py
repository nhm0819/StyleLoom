"""Build the stills that open each render request.

Two levels, and the split is the entire point of this stage.

  anchor   One still per run, generated from the cast creator and the cast setting.
           This is the identity: one face, one wardrobe, one room, one grade.

  frames   One still per render request, each generated *from the anchor* as a
           reference image. These are the compositions: the hook's framing, the
           second window's angle, whatever the storyboard asked for.

An earlier revision of this project had a keyframe stage, removed it, and recorded
why in docs/TOOL_RATIONALE.md: every cut generated its own keyframe from its own
text, so nothing was held across cuts and the second call per cut bought only
single-shot composition control. That critique was correct and it is not an
argument against a keyframe stage -- it is an argument against *that* one. The fix
named in the same document is "one keyframe reused as every cut's start image",
which is what `anchor` is.

Why not literally reuse the anchor as every request's first frame: with
image-to-video the start frame is the output's first frame, so two windows opening
on identical pixels reads as the video restarting. Deriving each window's frame
from the anchor keeps the person and the grade while letting the composition move.

Cost: one image call per run plus one per render request. In multi_shot mode that
is one per window -- three or four for a 30s video, not one per cut.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ..errors import ToolError
from ..events import EventKind
from ..schema import Casting, Keyframes, Shot, StyleSchema, Storyboard
from .registry import tool
from .render import FALLBACK_WINDOW_SEC, split_windows
from .storyboard import look_tokens

if TYPE_CHECKING:
    from ..context import Context
    from ..session import RunSession


def anchor_prompt(style: StyleSchema, casting: Casting) -> str:
    """What the run's identity still asks for.

    Deliberately not a shot from the storyboard. This frame is never used directly
    in the video, so composition is irrelevant and a neutral, well-lit, unambiguous
    subject is what a reference image should be -- the same reason a casting
    headshot is not an action photo.
    """
    return (
        f"{casting.creator.prompt}, in {casting.setting.prompt}. "
        f"{look_tokens(style)}. "
        "Neutral relaxed expression, facing camera, even soft lighting, "
        "full head and shoulders visible, sharp focus. "
        "Reference portrait for character consistency."
    )


def frame_prompt(style: StyleSchema, shot: Shot) -> str:
    """What one request's opening still asks for.

    The anchor supplies who and where, so this supplies only framing and moment.
    Repeating the creator description here would compete with the reference image
    rather than reinforce it -- and where they disagree, text tends to win, which
    is exactly backwards.
    """
    return (
        f"Same person, same location, same wardrobe. {shot.action}. "
        f"{shot.shot_size} shot, {shot.camera_move}. {look_tokens(style)}. "
        "Single frame, no text, no captions, no watermark."
    )


def lead_shots(ctx: Context, board: Storyboard) -> list[Shot]:
    """The shots that begin a render request, under the configured render mode.

    Windows come from `render.split_windows` rather than a second copy of the
    packing rule. If this stage grouped shots differently from render, it would
    generate frames keyed to leads that render never asks about, and every request
    would silently fall back to text-to-video after paying for the images.
    """
    if ctx.settings.render_mode != "multi_shot":
        return list(board.shots)
    windows = split_windows(
        board.shots,
        ctx.video.max_shot_window_sec or FALLBACK_WINDOW_SEC,
        ctx.video.max_shots_per_request,
        billed=ctx.video.shot_billed_duration,
    )
    return [window[0] for window in windows]


@tool("keyframe", reads=("style", "casting", "storyboard"), writes="keyframe")
def keyframe(ctx: Context, session: RunSession) -> Keyframes:
    """Generate the run's identity anchor and one opening still per request."""
    style = session.get("style", StyleSchema)
    casting = session.get("casting", Casting)
    board = session.get("storyboard", Storyboard)

    if not ctx.video.supports_first_frame:
        # Refused rather than skipped, matching how render refuses multi_shot on a
        # provider that cannot do it. A silently skipped keyframe stage renders a
        # perfectly good video in which the creator's face changes between cuts,
        # which is the failure this stage exists to prevent.
        raise ToolError(
            f"use_first_frame is on, but the {ctx.video.name!r} video provider "
            "cannot take a start frame. Set STYLELOOM_USE_FIRST_FRAME=false to "
            "render from text alone, or pick an endpoint that declares "
            "first_frame in configs/kling_models.yaml."
        )

    out_dir = session.workspace("keyframes")
    anchor = ctx.video.generate_image(
        anchor_prompt(style, casting), out_dir / "anchor.jpg"
    )

    frames: dict[int, Path] = {}
    failures: dict[int, str] = {}
    for shot in lead_shots(ctx, board):
        try:
            frames[shot.index] = ctx.video.generate_image(
                frame_prompt(style, shot),
                out_dir / f"lead_{shot.index:02d}.jpg",
                reference=anchor,
            )
        except Exception as exc:
            # One frame failing costs that request its anchor, not the run. Render
            # falls back to text-to-video for it, which is worse-looking rather
            # than broken -- and far better than discarding the frames that worked.
            failures[shot.index] = f"{type(exc).__name__}: {exc}"

    if failures:
        session.save_raw(
            "keyframe_errors.json", {str(k): v for k, v in failures.items()}
        )
        ctx.emit(
            EventKind.WARNING,
            session.run_id,
            stage="keyframe",
            message=(
                f"{len(failures)} of {len(failures) + len(frames)} opening frames "
                "failed; those requests fall back to text-to-video"
            ),
        )

    return Keyframes(anchor=anchor, frames=frames, model=ctx.settings.kling_t2i_model)
