"""Render the storyboard: one generation per window of cuts.

There used to be two strategies here, selected by `render_mode`. `per_shot` sent
one request per cut and trimmed each result to length, which made cut timing exact
by construction but paid the endpoint's duration floor on every cut -- at short-form
pacing, most of the bill. It is gone, and this is the only path:

  * Cuts are packed into windows that each fit one request, and each window is one
    generation carrying its own shot list and per-cut durations.
  * The floor is paid once per window instead of once per cut, and a 1.2s cut costs
    1.2s rather than the 3s minimum.
  * The cuts therefore land *inside* the model's output. The requested timeline is a
    request rather than a guarantee, and `qc` measures how far the delivered cuts
    drifted from it.

That last point is the trade the removal accepted. It also means the harness now
requires an endpoint that can carry several cuts in one request; `render` refuses
up front rather than silently sending one cut and billing for the rest.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from ..errors import ToolError
from ..events import EventKind
from ..providers import MotionShot
from ..schema import ClipSegment, Keyframes, RenderResult, Shot, Storyboard
from .registry import tool

if TYPE_CHECKING:
    from ..context import Context
    from ..session import RunSession

# Used when a provider reports no window ceiling of its own.
FALLBACK_WINDOW_SEC = 15.0


def split_windows(
    shots: list[Shot],
    window_sec: float,
    max_shots: int = 0,
    billed: Callable[[float], float] | None = None,
    min_sec: float = 0.0,
) -> list[list[Shot]]:
    """Group shots into generations that each fit inside one window.

    Greedy and order-preserving: a cut cannot be split across two generations, so
    a shot that would overflow the window starts the next one.

    Two limits bind independently and both have to be honoured. Kling accepts
    1-6 shots per request *and* 15s total; a 14-shot montage at 0.76s per cut
    satisfies the duration limit and violates the count one, and the endpoint
    rejects the whole request rather than truncating it. `max_shots` of 0 means the
    endpoint states no count limit.

    `billed` converts a requested duration into what the endpoint will actually
    run, since the window is measured against the latter: four 3.6s cuts request
    14.4s and deliver 16s, and the request is rejected whole.

    `min_sec` is the endpoint's floor for a whole request. It cannot be enforced by
    holding a window open -- that would breach the ceilings above -- so it is
    applied to the one window that can fall short, the trailing remainder, by
    merging it backwards when the window before it has room. A short request would
    otherwise have to be lifted to the floor, which buys footage nobody asked for.
    """
    if window_sec <= 0 and max_shots <= 0:
        return [list(shots)]
    length = billed or (lambda seconds: seconds)
    windows: list[list[Shot]] = []
    current: list[Shot] = []
    elapsed = 0.0
    for shot in shots:
        duration = length(shot.duration_sec)
        over_time = window_sec > 0 and elapsed + duration > window_sec
        over_count = max_shots > 0 and len(current) >= max_shots
        if current and (over_time or over_count):
            windows.append(current)
            current, elapsed = [], 0.0
        current.append(shot)
        elapsed += duration
    if current:
        windows.append(current)

    if min_sec > 0 and len(windows) > 1:
        tail, previous = windows[-1], windows[-2]
        total = sum(length(s.duration_sec) for s in tail + previous)
        fits_time = window_sec <= 0 or total <= window_sec
        fits_count = max_shots <= 0 or len(tail) + len(previous) <= max_shots
        if sum(length(s.duration_sec) for s in tail) < min_sec and fits_time and fits_count:
            windows[-2:] = [previous + tail]
    return windows


def window_shots(shots: list[Shot]) -> list[MotionShot]:
    """The prompt-and-duration pairs one request carries.

    Only the first entry establishes who and where: all of them are read by one
    generation, so repeating the presenter and the room in each would spend the
    per-shot character budget on what the model has already been told.
    """
    return [
        MotionShot(
            prompt=(s.video_prompt if i == 0 else s.continuation_video_prompt),
            duration=s.duration_sec,
        )
        for i, s in enumerate(shots)
    ]


def plan_windows(ctx: Context, board: Storyboard) -> list[list[Shot]]:
    """How the storyboard divides into requests, under the current endpoint.

    One function, called by both `render` and `keyframe`. If the two grouped shots
    differently, keyframe would generate opening frames keyed to leads that render
    never asks about, and every request would silently fall back to text-to-video
    after the images had been paid for.
    """
    return split_windows(
        board.shots,
        ctx.video.max_shot_window_sec or FALLBACK_WINDOW_SEC,
        ctx.video.max_shots_per_request,
        billed=ctx.video.shot_billed_duration,
        min_sec=ctx.video.min_clip_sec,
    )


@tool("render", reads=("storyboard",), writes="render")
def render(ctx: Context, session: RunSession) -> RenderResult:
    """One generation per window, isolating failures to the window that broke.

    Windows render sequentially rather than in parallel. A window is already
    several shots' worth of work, the endpoints that support this have low
    concurrency limits, and a failure costs a whole window -- so there is little to
    gain and a larger blast radius to manage.
    """
    board = session.get("storyboard", Storyboard)
    # Not declared in `reads`: the keyframe stage is optional, and declaring it
    # would make a plan without it fail validation rather than render from text.
    frames = (
        session.get("keyframe", Keyframes) if session.has("keyframe") else Keyframes()
    )
    if not ctx.video.supports_multi_shot:
        raise ToolError(
            f"the {ctx.video.name!r} video provider cannot carry several cuts in "
            "one generation, which is the only way this harness renders. Pick an "
            "endpoint that declares multi_prompt or semicolon_prompt in "
            "configs/kling_models.yaml (run `styleloom models` to see which do)."
        )

    out_dir = session.workspace("shots")
    segments: list[ClipSegment] = []
    errors: dict[int, str] = {}

    for w, shots in enumerate(plan_windows(ctx, board)):
        # What the endpoint will actually run, which is what captions are placed
        # against and what the delivered length will be. Read from the provider
        # rather than recomputed here: the quantisation and the request floor are
        # the endpoint's arithmetic, and a second copy of it would drift.
        planned = ctx.video.plan_shot_durations([s.duration_sec for s in shots])
        try:
            clip: Path = ctx.video.generate_sequence(
                window_shots(shots),
                out_dir / f"win_{w:02d}.mp4",
                # Keyed on the window's lead, which is the shot whose composition the
                # frame was generated for. `keyframe.lead_shots` packs windows with
                # the same function, so the keys line up by construction.
                frames.for_lead(shots[0].index),
            )
        except Exception as exc:
            # A window failure loses every cut in it. That is the price of paying the
            # duration floor once per request instead of once per cut.
            message = f"{type(exc).__name__}: {exc}"
            for s in shots:
                errors[s.index] = message
            continue

        segments.append(
            ClipSegment(
                path=clip,
                shot_indices=[s.index for s in shots],
                requested_durations=[s.duration_sec for s in shots],
                billed_durations=planned,
            )
        )

    if errors:
        session.save_raw(
            "render_errors.json", {str(k): v for k, v in errors.items()}
        )
        ctx.emit(
            EventKind.WARNING,
            session.run_id,
            stage="render",
            message=f"{len(errors)} of {len(board.shots)} shots failed",
        )
    if not segments:
        raise ToolError(f"all {len(board.shots)} shots failed to render: {errors}")

    billed = sum(sum(seg.billed_durations) for seg in segments)
    ctx.emit(
        EventKind.WARNING,
        session.run_id,
        stage="render",
        message=(
            f"{len(board.shots)} cuts in {len(segments)} generation(s), "
            f"{billed:.1f}s billed. Cut timing is the model's -- see the qc "
            "drift check."
        ),
    )
    return RenderResult(segments=segments, errors=errors)
