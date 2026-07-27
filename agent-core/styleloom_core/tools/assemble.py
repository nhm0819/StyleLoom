"""Burn captions and concatenate into the final video."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..media import CaptionCue, burn_captions, concat
from ..schema import AssembleResult, ClipSegment, RenderResult, Storyboard, StyleSchema
from .registry import tool

if TYPE_CHECKING:
    from ..context import Context
    from ..session import RunSession


def cues_for(segment: ClipSegment, board: Storyboard) -> list[CaptionCue]:
    """Caption windows within one segment, measured from the segment's own start.

    This is where the render mode stops mattering. In per_shot mode a segment holds
    one shot and yields one cue spanning the clip; in multi_shot mode it holds
    several and yields a cue per cut. Either way `assemble` sees a list of cues.

    The windows come from the *requested* durations. If a multi_shot endpoint drifts
    from the timeline it was given, captions drift with it -- which is exactly why
    `qc` measures that drift rather than assuming it is zero.
    """
    by_index = {s.index: s for s in board.shots}
    cues: list[CaptionCue] = []
    elapsed = 0.0
    for index, duration in zip(
        segment.shot_indices, segment.requested_durations, strict=True
    ):
        shot = by_index.get(index)
        if shot is not None and shot.caption.strip():
            cues.append(
                CaptionCue(text=shot.caption, start=elapsed, end=elapsed + duration)
            )
        elapsed += duration
    return cues


@tool("assemble", reads=("style", "storyboard", "render"), writes="assemble")
def assemble(ctx: Context, session: RunSession) -> AssembleResult:
    """Caption each segment, then concatenate in shot order.

    Captions are burned here rather than asked of the video model: models render
    text unreliably and inconsistently, and the caption spec in style.json (font,
    position, wrap width, stroke) has to be byte-identical across every video for
    the set to read as one channel. drawtext guarantees that; a model does not.
    """
    style = session.get("style", StyleSchema)
    board = session.get("storyboard", Storyboard)
    rendered = session.get("render", RenderResult)

    captioned_dir = session.workspace("captioned")
    captioned = []
    burned = 0

    for i, segment in enumerate(rendered.segments):
        cues = cues_for(segment, board)
        out = burn_captions(
            segment.path, cues, style.caption, captioned_dir / f"seg_{i:02d}.mp4"
        )
        if out != segment.path:
            burned += len(cues)
        captioned.append(out)

    final = concat(captioned, session.dir / "final.mp4", session.inputs.bgm)
    return AssembleResult(final_video=final, captioned_count=burned)
