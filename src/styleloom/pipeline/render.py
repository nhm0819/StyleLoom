"""Stage 6 - Shot rendering.

Each shot goes text -> keyframe -> clip. Shots are independent, so they render
concurrently; the bottleneck is provider latency, not CPU.

Failures are isolated per shot: a failed shot is recorded and can be re-rendered
alone without redoing the run.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ..config import settings
from ..providers.video import BaseVideoProvider
from ..schema import Shot, Storyboard
from . import post


TRIM_EPSILON = 0.05


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
    real and unavoidable; a 1.2s cut still costs a 4s generation.
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
    return post.trim_to(raw, shot.duration_sec, final)


def render_storyboard(
    provider: BaseVideoProvider,
    storyboard: Storyboard,
    out_dir: Path,
    persona_ref: Path | None = None,
    only: list[int] | None = None,
) -> tuple[list[Path], dict[int, str]]:
    """Returns (clip paths in shot order, {shot_index: error}) ."""
    out_dir.mkdir(parents=True, exist_ok=True)
    targets = [s for s in storyboard.shots if only is None or s.index in only]

    results: dict[int, Path] = {}
    errors: dict[int, str] = {}

    def work(shot: Shot) -> None:
        try:
            results[shot.index] = render_shot(provider, shot, out_dir, persona_ref)
        except Exception as exc:  # provider errors are expected and recoverable
            errors[shot.index] = f"{type(exc).__name__}: {exc}"

    workers = max(min(settings.max_concurrent_renders, provider.max_concurrency), 1)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(work, targets))

    ordered = [results[s.index] for s in storyboard.shots if s.index in results]
    return ordered, errors
