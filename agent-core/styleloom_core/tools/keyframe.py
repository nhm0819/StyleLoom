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

Cost: one image call per run plus one per render request -- one per window, so
three or four for a 30s video rather than one per cut.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ..errors import ToolError
from ..events import EventKind
from ..schema import Brief, Casting, Keyframes, Shot, Storyboard, StyleSchema
from .registry import tool
from .render import plan_windows
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

    The style's own lighting, texture and palette go in, because every later frame
    descends from this one and inherits whatever look it establishes. `subject` and
    `lens` are left out on purpose: this frame's framing is fixed by the sentence
    below, and a reference portrait wants everything sharp -- so "shallow depth of
    field" or "hands only, no talking head" would fight the one job it has.
    """
    detail = style.look.detail
    look = [f"{style.look.grade} colour grade"]
    if detail.any_set():
        look += [p for p in (detail.lighting, detail.texture, detail.palette) if p]
    else:
        # Pre-split style: a flat list cannot be filtered, so it goes in whole.
        look += list(style.look.keywords)
    return (
        f"{casting.creator.prompt}, in {casting.setting.prompt}. "
        f"{', '.join(look)}. "
        "Neutral relaxed expression, facing camera, even soft lighting, "
        "full head and shoulders visible, sharp focus. "
        "Reference portrait for character consistency."
    )


def frame_prompt(style: StyleSchema, shot: Shot, brief: Brief | None = None) -> str:
    """What one request's opening still asks for.

    The anchor supplies who and where, so this supplies only framing and moment.
    Repeating the creator description here would compete with the reference image
    rather than reinforce it -- and where they disagree, text tends to win, which
    is exactly backwards.

    Everything else the reference has to say goes in, because this is the one prompt
    in the pipeline with room for it. The image endpoint allows 2500 characters and
    this used to spend 212 of them; the video endpoint allows 512 per shot and is the
    binding constraint on the whole system. So the look is established *here*, where
    there is room to describe it, and image-to-video inherits it from the frame
    instead of being told about it in 512 characters it does not have.

    `brief` is what the run is actually about. The storyboard leaves the topic out of
    its shot prompts deliberately -- there a paraphrase costs characters the shot
    description needs -- but here it is free, and a still generated without it drifts
    toward a generic version of the beat.
    """
    parts = [
        "Same person, same location, same wardrobe.",
        f"{shot.action}.",
        f"{shot.shot_size} shot, {shot.camera_move}.",
    ]
    if brief is not None:
        # The subject of the video, so the still shows this product and this moment
        # rather than a plausible stand-in for them.
        parts.append(f"Subject of the video: {brief.topic}. {brief.key_message}.")
    parts.append(f"{look_tokens(style, for_image=True)}.")
    parts.append("Single frame, no text, no captions, no watermark.")
    return " ".join(parts)


def lead_shots(ctx: Context, board: Storyboard) -> list[Shot]:
    """The shots that begin a render request.

    Windows come from `render.plan_windows` rather than a second copy of the packing
    rule. If this stage grouped shots differently from render, it would generate
    frames keyed to leads that render never asks about, and every request would
    silently fall back to text-to-video after paying for the images.
    """
    return [window[0] for window in plan_windows(ctx, board)]


@tool("keyframe", reads=("style", "brief", "casting", "storyboard"), writes="keyframe")
def keyframe(ctx: Context, session: RunSession) -> Keyframes:
    """Generate the run's identity anchor and one opening still per request."""
    style = session.get("style", StyleSchema)
    casting = session.get("casting", Casting)
    board = session.get("storyboard", Storyboard)
    # Read for the prompts, not for the plan: the frame stills are the one place with
    # room to say what the video is about.
    brief = session.get("brief", Brief)

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
                frame_prompt(style, shot, brief),
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
