"""Render each shot: text -> keyframe -> motion clip.

Two strategies, selected by `render_mode`. Both produce `ClipSegment`s, so nothing
downstream branches on which one ran.

  per_shot    One generation per cut, trimmed to length. Cuts land on file
              boundaries, so shot durations are exact by construction. Pays the
              endpoint's duration floor on every cut -- at short-form pacing that
              is most of the bill.

  multi_shot  One generation carrying several cuts and their durations. Removes
              the floor, roughly halving cost, and lets a 1.2s cut actually cost
              1.2s. The trade is that the cuts land inside the model's output: the
              requested timeline becomes a request rather than a guarantee, and
              `qc` measures how far the delivered cuts drifted.

per_shot is the default because it is the verified path. multi_shot is opt-in.
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING

from ..errors import ToolError
from ..events import EventKind
from ..media import trim_to
from ..providers import BaseVideoProvider, MotionShot
from ..schema import Casting, ClipSegment, RenderResult, Shot, Storyboard
from .registry import tool

if TYPE_CHECKING:
    from ..context import Context
    from ..session import RunSession

# A clip within this margin of the target needs no trim pass.
TRIM_EPSILON = 0.05
# Used when a provider reports no window ceiling of its own.
FALLBACK_WINDOW_SEC = 15.0


def render_shot(
    provider: BaseVideoProvider,
    shot: Shot,
    out_dir: Path,
    persona_ref: Path | None = None,
) -> Path:
    """Render one shot at exactly `shot.duration_sec`.

    Every real image-to-video endpoint has a duration floor (Seedance 4s, Kling
    3s) well above a typical short-form shot, and short-form pacing is precisely
    what the style schema encodes. So request the floor, then trim. The waste is
    real and unavoidable in this mode: a 1.2s cut still costs a 3-4s generation.
    Stretching the shot to fill the floor instead would destroy the pacing, which
    is the one property the whole system exists to reproduce. `multi_shot` is the
    way out; see the module docstring.
    """
    keyframe = provider.keyframe(
        shot.image_prompt, out_dir / f"shot_{shot.index:02d}.jpg", ref_image=persona_ref
    )
    requested = max(shot.duration_sec, provider.min_clip_sec)
    final = out_dir / f"shot_{shot.index:02d}.mp4"

    if requested <= shot.duration_sec + TRIM_EPSILON:
        return provider.animate(keyframe, shot.motion_prompt, requested, final, persona_ref)

    raw = provider.animate(
        keyframe,
        shot.motion_prompt,
        requested,
        out_dir / "raw" / f"shot_{shot.index:02d}.mp4",
        persona_ref,
    )
    return trim_to(raw, shot.duration_sec, final)


def split_windows(
    shots: list[Shot],
    window_sec: float,
    max_shots: int = 0,
    billed: Callable[[float], float] | None = None,
) -> list[list[Shot]]:
    """Group shots into generations that each fit inside one window.

    Greedy and order-preserving: a cut cannot be split across two generations, so
    a shot that would overflow the window starts the next one.

    Two limits bind independently and both have to be honoured. Kling accepts
    1-6 shots per multi_prompt request *and* 15s total; a 14-shot montage at
    0.76s per cut satisfies the duration limit and violates the count one, and
    the endpoint rejects the whole request rather than truncating it. `max_shots`
    of 0 means the endpoint states no count limit.

    `billed` converts a requested duration into the length the endpoint will
    actually run, because the window is measured against the latter. Kling
    quantises per-cut durations to whole seconds, so four 3.6s cuts request 14.4s
    -- inside the window -- and deliver 16s, which is outside it. Packing on the
    requested figures builds a request the endpoint rejects whole.
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
    return windows


def estimated_windows(shots: list[Shot], window_sec: float) -> int:
    """How many generations multi_shot would need. Used by cost reporting."""
    return len(split_windows(shots, window_sec))


def _render_per_shot(
    ctx: Context, session: RunSession, board: Storyboard, persona_ref: Path | None
) -> RenderResult:
    out_dir = session.workspace("shots")
    results: dict[int, Path] = {}
    errors: dict[int, str] = {}

    def work(shot: Shot) -> None:
        try:
            results[shot.index] = render_shot(ctx.video, shot, out_dir, persona_ref)
        except Exception as exc:  # provider errors are expected and recoverable
            errors[shot.index] = f"{type(exc).__name__}: {exc}"

    # The provider's ceiling wins over the setting: Kling's concurrency limit is
    # a property of the account tier (1 on the entry plan) and exceeding it fails
    # the whole run.
    workers = max(min(ctx.settings.max_concurrent_renders, ctx.video.max_concurrency), 1)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(work, board.shots))

    segments = [
        ClipSegment(
            path=results[s.index],
            shot_indices=[s.index],
            requested_durations=[s.duration_sec],
        )
        for s in board.shots
        if s.index in results
    ]
    return RenderResult(mode="per_shot", segments=segments, errors=errors)


def _render_multi_shot(
    ctx: Context, session: RunSession, board: Storyboard, persona_ref: Path | None
) -> RenderResult:
    """One generation per window of shots.

    Windows render sequentially rather than in parallel. A window is already
    several shots' worth of work, the endpoints that support this have low
    concurrency limits, and a failure costs a whole window -- so there is little
    to gain and a larger blast radius to manage.
    """
    out_dir = session.workspace("shots")
    window_sec = ctx.video.max_shot_window_sec or FALLBACK_WINDOW_SEC
    windows = split_windows(
        board.shots,
        window_sec,
        ctx.video.max_shots_per_request,
        billed=ctx.video.shot_billed_duration,
    )

    segments: list[ClipSegment] = []
    errors: dict[int, str] = {}

    for w, shots in enumerate(windows):
        # The first shot's keyframe anchors the look for the whole window; the
        # per-shot prompts inside the request carry the rest.
        lead = shots[0]
        try:
            keyframe = ctx.video.keyframe(
                lead.image_prompt, out_dir / f"win_{w:02d}.jpg", ref_image=persona_ref
            )
            clip = ctx.video.animate_sequence(
                keyframe,
                [
                    MotionShot(prompt=s.motion_prompt, duration=s.duration_sec)
                    for s in shots
                ],
                out_dir / f"win_{w:02d}.mp4",
                persona_ref,
            )
        except Exception as exc:
            # A window failure loses every cut in it, unlike per_shot mode where
            # one bad shot costs one shot. That is the price of the cheaper path.
            message = f"{type(exc).__name__}: {exc}"
            for s in shots:
                errors[s.index] = message
            continue

        segments.append(
            ClipSegment(
                path=clip,
                shot_indices=[s.index for s in shots],
                requested_durations=[s.duration_sec for s in shots],
                billed_durations=[
                    ctx.video.shot_billed_duration(s.duration_sec) for s in shots
                ],
            )
        )

    return RenderResult(mode="multi_shot", segments=segments, errors=errors)


@tool("render", reads=("storyboard", "casting"), writes="render")
def render(ctx: Context, session: RunSession) -> RenderResult:
    """Render every shot, isolating failures as far as the chosen mode allows."""
    board = session.get("storyboard", Storyboard)
    # A caller-supplied reference wins: if someone hands us a specific person, the
    # cast portrait is not what they asked for.
    persona_ref = session.inputs.persona_ref or session.get("casting", Casting).creator_ref

    mode = ctx.settings.render_mode
    if mode == "multi_shot" and not ctx.video.supports_multi_shot:
        raise ToolError(
            f"render_mode=multi_shot, but the {ctx.video.name!r} video provider "
            "cannot render several cuts in one generation. Use render_mode=per_shot, "
            "or an endpoint that declares multi_prompt in configs/kling_models.yaml "
            "(run `styleloom models` to see which do)."
        )

    if mode == "multi_shot":
        result = _render_multi_shot(ctx, session, board, persona_ref)
    else:
        result = _render_per_shot(ctx, session, board, persona_ref)

    if result.errors:
        session.save_raw(
            "render_errors.json", {str(k): v for k, v in result.errors.items()}
        )
        ctx.emit(
            EventKind.WARNING,
            session.run_id,
            stage="render",
            message=f"{len(result.errors)} of {len(board.shots)} shots failed",
        )
    if not result.segments:
        raise ToolError(
            f"all {len(board.shots)} shots failed to render (mode={mode}): {result.errors}"
        )

    if mode == "multi_shot":
        billed = sum(sum(seg.requested_durations) for seg in result.segments)
        ctx.emit(
            EventKind.WARNING,
            session.run_id,
            stage="render",
            message=(
                f"multi_shot: {len(board.shots)} cuts in {len(result.segments)} "
                f"generation(s), {billed:.1f}s billed. Cut timing is the model's "
                "-- see the qc drift check."
            ),
        )
    return result
