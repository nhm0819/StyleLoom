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


def look_tokens(style: StyleSchema) -> str:
    """Grade only. QC measures these, so they go in every entry even when
    identity does not -- a continuation inherits the person, not the grade."""
    look = style.look
    return (
        f"{look.grade} colour grade, "
        f"saturation {look.saturation:.2f}, contrast {look.contrast:.2f}"
    )


def style_tokens(style: StyleSchema, casting: Casting, budget: int = 0) -> str:
    """The look, the presenter and the location, compressed into prompt tokens.

    Casting comes first: identity is what viewers notice breaking, so it should
    not sit at the tail where models weight it least.

    `budget` caps the result. Emission order and drop order are deliberately
    different: casting is emitted first so the model weights it heavily, and given
    up first because colour is what QC measures. Only whole clauses are removed --
    cutting to a character count would hand the model half a sentence.
    """
    look = style.look
    parts = [
        casting.creator.prompt,
        f"in {casting.setting.prompt}",
        f"{look.grade} colour grade",
        f"saturation {look.saturation:.2f}",
        f"contrast {look.contrast:.2f}",
    ]
    # No framing token. Under text-to-video `aspect_ratio` is an explicit request
    # field, so spelling "vertical 9:16" into the prompt spends characters
    # restating what the API already knows -- and inside a 512-character
    # storyboard entry those characters cost `contrast`, which QC measures.
    fixed = len(parts)
    parts += look.keywords[:4]
    parts = [p for p in parts if p]

    if budget > 0:
        # Index order to give up, most expendable first: the descriptive keywords,
        # then location, then the presenter, and only then the measured colour.
        give_up = list(range(fixed, len(parts))) + [1, 0, 4, 3, 2]
        dropped: set[int] = set()
        for index in give_up:
            if len(", ".join(p for i, p in enumerate(parts) if i not in dropped)) <= budget:
                break
            if index < len(parts):
                dropped.add(index)
        parts = [p for i, p in enumerate(parts) if i not in dropped]
    return ", ".join(parts)


def ensure_distinct_frames(shots: list[Shot]) -> list[Shot]:
    """Guarantee that no cut asks for the same shot as the one before it.

    The camera move in the prompt handles this whenever a style has more than one
    move, since moves cycle by shot index. A style with a single move -- which the
    schema permits -- would still produce identical neighbours inside one beat, and
    an invisible cut is worse than a slightly redundant prompt token. So this is a
    cheap final pass rather than another rule inside the builder.

    Only adjacency is enforced. Returning to an earlier framing later in a video is
    ordinary editing, not a defect.
    """
    for previous, current in zip(shots, shots[1:], strict=False):
        if current.scene_prompt == previous.scene_prompt:
            current.scene_prompt += " Alternate angle on the same moment."
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
    session.get("brief", Brief)  # declared dependency
    outline = session.get("outline", Outline)
    hook = session.get("hook", HookResult)
    casting = session.get("casting", Casting)

    rng = random.Random()
    # A multi-shot entry allows 512 characters; a single-cut prompt allows
    # thousands. Measured against the whole video_prompt, since the beat text
    # varies per shot -- a fixed token/remainder split is wrong for the long one.
    limit = ctx.video.max_shot_prompt_chars if ctx.settings.render_mode == "multi_shot" else 0
    tokens = style_tokens(style, casting)
    looks = look_tokens(style)

    def fit(scene: str, motion: str) -> str:
        """Shrink the shared tokens until scene + motion fits the entry limit."""
        if not limit or len(f"{scene} {motion}") <= limit:
            return scene
        head, _, tail = scene.partition(tokens)
        for size in range(len(tokens), 0, -16):
            shorter = style_tokens(style, casting, budget=size)
            candidate = f"{head}{shorter}{tail}"
            if len(f"{candidate} {motion}") <= limit:
                return candidate
        return f"{head}{style_tokens(style, casting, budget=1)}{tail}"
    moves = style.camera.moves or ["static"]
    # brief.topic is deliberately not injected: the beat text already describes
    # this shot of this subject, and the paraphrase cost `contrast` in a 512-
    # character entry.
    shots: list[Shot] = []

    # --- hook window -------------------------------------------------------
    hook_cuts = max(style.hook_style.cut_count, 1)
    hook_shot_sec = round(style.hook_style.window_sec / hook_cuts, 2)
    for i in range(hook_cuts):
        move = moves[i % len(moves)]
        hook_scene = (
            f"{hook.selected.visual}. {style.hook_style.shot_size} shot, "
            f"{move}. {tokens}. Opening shot."
        )
        hook_motion = "Fast and attention-grabbing."
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
                scene_prompt=fit(hook_scene, hook_motion),
                motion_prompt=hook_motion,
                continuation_prompt=(
                    f"{hook.selected.visual}. {style.hook_style.shot_size} shot, "
                    f"{move}. Same subject and location. {looks}."
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
            body_scene = (
                f"{beat.content}. {sizes[j]} shot, {move}. {tokens}."
            )
            # Not the move -- that is already in the scene line above.
            body_motion = "Natural continuous motion, no cut inside the shot."
            shots.append(
                Shot(
                    index=len(shots),
                    role=beat.name,
                    duration_sec=per,
                    shot_size=sizes[j],
                    camera_move=move,
                    action=beat.content,
                    caption=beat.content if j == 0 else "",
                    scene_prompt=fit(body_scene, body_motion),
                    motion_prompt=body_motion,
                    continuation_prompt=(
                        f"{beat.content}. {sizes[j]} shot, {move}. "
                        f"Same subject and location. {looks}."
                    ),
                )
            )

    return Storyboard(
        style_id=style.style_id,
        hook_text=hook.selected.text,
        hook_archetype=hook.archetype_sampled,
        shots=ensure_distinct_frames(shots),
    )
