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


def render_shot(
    provider: BaseVideoProvider,
    shot: Shot,
    out_dir: Path,
    persona_ref: Path | None = None,
) -> Path:
    keyframe = provider.keyframe(
        shot.image_prompt, out_dir / f"shot_{shot.index:02d}.jpg", ref_image=persona_ref
    )
    return provider.animate(
        keyframe, shot.motion_prompt, shot.duration_sec, out_dir / f"shot_{shot.index:02d}.mp4"
    )


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

    with ThreadPoolExecutor(max_workers=settings.max_concurrent_renders) as pool:
        list(pool.map(work, targets))

    ordered = [results[s.index] for s in storyboard.shots if s.index in results]
    return ordered, errors
