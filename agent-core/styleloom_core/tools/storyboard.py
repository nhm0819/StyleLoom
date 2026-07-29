"""Turn style + hook + outline into shot-level prompts."""

from __future__ import annotations

import math
import random
from typing import TYPE_CHECKING

from ..schema import (
    Brief,
    Casting,
    HookResult,
    Outline,
    Shot,
    ShotSize,
    Storyboard,
    StyleSchema,
)
from .registry import tool

if TYPE_CHECKING:
    from ..context import Context
    from ..session import RunSession

MIN_AVG_SHOT_SEC = 0.3


def style_tokens(style: StyleSchema, casting: Casting) -> str:
    """The look, the presenter and the location, compressed into prompt tokens.

    Injected into *every* shot prompt rather than stated once, because image and
    video models have no memory across calls -- a grade described only in shot 1
    has drifted by shot 6, and a presenter described only once becomes a different
    person by the third cut.

    Casting comes first: subject identity is the thing viewers notice breaking, so
    it should not sit at the tail of a long prompt where models weight it least.
    """
    look = style.look
    parts = [
        casting.creator.prompt,
        f"in {casting.setting.prompt}",
        f"{look.grade} colour grade",
        f"saturation {look.saturation:.2f}",
        f"contrast {look.contrast:.2f}",
        "vertical 9:16 framing",
    ]
    parts += look.keywords[:4]
    return ", ".join(p for p in parts if p)


def ensure_distinct_frames(shots: list[Shot]) -> list[Shot]:
    """Guarantee that no cut asks for the same image as the one before it.

    The camera move in the prompt handles this whenever a style has more than one
    move, since moves cycle by shot index. A style with a single move -- which the
    schema permits -- would still produce identical neighbours inside one beat, and
    an invisible cut is worse than a slightly redundant prompt token. So this is a
    cheap final pass rather than another rule inside the builder.

    Only adjacency is enforced. Returning to an earlier framing later in a video is
    ordinary editing, not a defect.
    """
    for previous, current in zip(shots, shots[1:], strict=False):
        if current.image_prompt == previous.image_prompt:
            current.image_prompt += " Alternate angle on the same moment."
    return shots


def shot_count_for(beat_sec: float, avg_shot_sec: float) -> int:
    """How many cuts to spend on a beat so the result sits closest to the style.

    A beat rarely divides evenly into the reference's cut length, so the count is
    a choice between the two neighbouring integers -- and they are not equally
    good. Rounding to nearest picks the closer *count*, which is not the same as
    the closer *pacing*: a 3s beat at 1.21s per cut rounds to 2 cuts of 1.50s,
    when 3 cuts of 1.00s sit nearer the target. That is 24% off instead of 17%,
    on the one number the whole schema exists to hold.

    So both candidates are measured against the thing being reproduced. This was
    invisible while the QC tolerance on avg_shot_sec was 0.6s absolute, which on
    a 1.2s reference accepted anything from 0.6s to 1.8s.
    """
    target = max(avg_shot_sec, MIN_AVG_SHOT_SEC)
    ideal = beat_sec / target
    candidates = {max(1, math.floor(ideal)), max(1, math.ceil(ideal))}
    return min(candidates, key=lambda n: abs(beat_sec / n - target))


def sizes_for(style: StyleSchema, count: int, rng: random.Random) -> list[ShotSize]:
    dist = style.camera.shot_size_dist or {"CU": 1.0}
    sizes = list(dist.keys())
    weights = [max(dist[s], 1e-6) for s in sizes]
    return rng.choices(sizes, weights=weights, k=count)  # type: ignore[return-value]


@tool(
    "storyboard",
    reads=("style", "brief", "casting", "outline", "hook"),
    writes="storyboard",
)
def storyboard(ctx: Context, session: RunSession) -> Storyboard:
    """Expand beats into individually promptable shots at the reference's pacing."""
    style = session.get("style", StyleSchema)
    brief = session.get("brief", Brief)
    outline = session.get("outline", Outline)
    hook = session.get("hook", HookResult)
    casting = session.get("casting", Casting)

    rng = random.Random()
    tokens = style_tokens(style, casting)
    moves = style.camera.moves or ["static"]
    topic = brief.topic
    shots: list[Shot] = []

    # --- hook window -------------------------------------------------------
    hook_cuts = max(style.hook_style.cut_count, 1)
    hook_shot_sec = round(style.hook_style.window_sec / hook_cuts, 2)
    for i in range(hook_cuts):
        move = moves[i % len(moves)]
        shots.append(
            Shot(
                index=len(shots),
                role="hook",
                duration_sec=hook_shot_sec,
                shot_size=style.hook_style.shot_size,
                camera_move=move,
                action=hook.selected.visual,
                # Caption only on the first hook shot: the text should land once
                # and hold, not re-animate on every cut inside the window.
                caption=hook.selected.text if i == 0 else "",
                # The camera move belongs in the still prompt, not only in the
                # motion prompt. Without it every cut inside the hook window asks
                # for byte-identical imagery, so a "2-cut hook" is one frame shown
                # twice and the cut is invisible -- which also made it undetectable
                # to the QC drift check.
                image_prompt=(
                    f"{hook.selected.visual}. Subject of: {topic}. "
                    f"{style.hook_style.shot_size} shot, {move}. {tokens}. "
                    "Opening frame of a short-form video, immediately legible."
                ),
                motion_prompt=(
                    f"{move}, fast and attention-grabbing. {hook.selected.visual}. "
                    "No text overlay in the generated footage."
                ),
            )
        )

    # --- body --------------------------------------------------------------
    for beat in outline.beats:
        count = max(
            int(round(beat.duration_sec / max(style.pacing.avg_shot_sec, MIN_AVG_SHOT_SEC))),
            1,
        )
        per = round(beat.duration_sec / count, 2)
        sizes = sizes_for(style, count, rng)
        for j in range(count):
            move = moves[(len(shots) + j) % len(moves)]
            shots.append(
                Shot(
                    index=len(shots),
                    role=beat.name,
                    duration_sec=per,
                    shot_size=sizes[j],
                    camera_move=move,
                    action=beat.content,
                    caption=beat.content if j == 0 else "",
                    image_prompt=(
                        f"{beat.content}. Subject of: {topic}. "
                        f"{sizes[j]} shot, {move}. {tokens}."
                    ),
                    motion_prompt=(
                        f"{move}. {beat.content}. "
                        "No text overlay in the generated footage."
                    ),
                )
            )

    return Storyboard(
        style_id=style.style_id,
        hook_text=hook.selected.text,
        hook_archetype=hook.archetype_sampled,
        shots=ensure_distinct_frames(shots),
    )
