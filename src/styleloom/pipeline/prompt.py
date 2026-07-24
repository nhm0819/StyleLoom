"""Stage 5 - Storyboard assembly.

Turns (style + hook + outline) into shot-level prompts. Style tokens are
injected into *every* shot prompt rather than stated once, because image and
video models have no memory across calls -- a look described only in shot 1
drifts by shot 6.
"""

from __future__ import annotations

import random

from ..schema import HookResult, Outline, Shot, Storyboard, StyleSchema


def _style_tokens(style: StyleSchema) -> str:
    look = style.look
    parts = [
        f"{look.grade} colour grade",
        f"saturation {look.saturation:.2f}",
        f"contrast {look.contrast:.2f}",
        "vertical 9:16 framing",
    ]
    parts += look.keywords[:4]
    return ", ".join(parts)


def _sizes_for(style: StyleSchema, count: int, rng: random.Random) -> list[str]:
    dist = style.camera.shot_size_dist or {"CU": 1.0}
    sizes = list(dist.keys())
    weights = [max(dist[s], 1e-6) for s in sizes]
    return rng.choices(sizes, weights=weights, k=count)


def build_storyboard(
    style: StyleSchema,
    hook: HookResult,
    outline: Outline,
    topic: str,
) -> Storyboard:
    rng = random.Random()
    tokens = _style_tokens(style)
    moves = style.camera.moves or ["static"]
    shots: list[Shot] = []

    # --- hook window ------------------------------------------------------- #
    hook_cuts = max(style.hook_style.cut_count, 1)
    hook_shot_sec = round(style.hook_style.window_sec / hook_cuts, 2)
    for i in range(hook_cuts):
        shots.append(
            Shot(
                index=len(shots),
                role="hook",
                duration_sec=hook_shot_sec,
                shot_size=style.hook_style.shot_size,
                camera_move=moves[i % len(moves)],
                action=hook.selected.visual,
                caption=hook.selected.text if i == 0 else "",
                image_prompt=(
                    f"{hook.selected.visual}. Subject of: {topic}. "
                    f"{style.hook_style.shot_size} shot. {tokens}. "
                    "Opening frame of a short-form video, immediately legible."
                ),
                motion_prompt=(
                    f"{moves[i % len(moves)]}, fast and attention-grabbing. "
                    f"{hook.selected.visual}. No text overlay in the generated footage."
                ),
            )
        )

    # --- body -------------------------------------------------------------- #
    for beat in outline.beats:
        count = max(int(round(beat.duration_sec / max(style.pacing.avg_shot_sec, 0.3))), 1)
        per = round(beat.duration_sec / count, 2)
        sizes = _sizes_for(style, count, rng)
        for j in range(count):
            shots.append(
                Shot(
                    index=len(shots),
                    role=beat.name,
                    duration_sec=per,
                    shot_size=sizes[j],  # type: ignore[arg-type]
                    camera_move=moves[(len(shots) + j) % len(moves)],
                    action=beat.content,
                    caption=beat.content if j == 0 else "",
                    image_prompt=(
                        f"{beat.content}. Subject of: {topic}. "
                        f"{sizes[j]} shot. {tokens}."
                    ),
                    motion_prompt=(
                        f"{moves[(len(shots) + j) % len(moves)]}. {beat.content}. "
                        "No text overlay in the generated footage."
                    ),
                )
            )

    return Storyboard(
        style_id=style.style_id,
        hook_text=hook.selected.text,
        hook_archetype=hook.archetype_sampled,
        shots=shots,
    )
