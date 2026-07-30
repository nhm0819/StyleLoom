"""Turn style + hook + outline into shot-level prompts."""

from __future__ import annotations

import math
import random
from typing import TYPE_CHECKING

from ..budget import BODY_MOTION, HOOK_MOTION, look_tokens, plan_shot_text, style_tokens
from ..events import EventKind
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
    # One shot's `words` inside a shot list allows 512 characters where the whole
    # prompt field allows thousands, and every prompt is built to the tighter number
    # because any shot can end up sharing a window with five others. Measured against
    # the whole video_prompt, since the beat text varies per shot -- a fixed
    # token/remainder split is wrong for the long one.
    limit = ctx.video.max_shot_prompt_chars
    # The same split the outline and the hook were given, so the tokens here are
    # the ones their character budgets were computed against.
    tokens, _ = plan_shot_text(limit, style, casting)
    looks = look_tokens(style)

    squeezed: list[int] = []

    def fit(scene: str, motion: str) -> str:
        """Shrink the shared tokens until scene + motion fits the entry limit.

        The last resort, not the mechanism. `outline` and `hook` are given the
        character budget up front precisely so this does not run -- when it does,
        it pays for the room by dropping style clauses, and the first thing it gives
        up is the colour grade that QC measures. So a run where it binds is a run
        whose grade score is being spent on an over-long sentence, and it says so.
        """
        if not limit or len(f"{scene} {motion}") <= limit:
            return scene
        squeezed.append(len(scene))
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
        hook_motion = HOOK_MOTION
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
            body_motion = BODY_MOTION
            shots.append(
                Shot(
                    index=len(shots),
                    role=beat.name,
                    duration_sec=per,
                    shot_size=sizes[j],
                    camera_move=move,
                    action=beat.content,
                    # The beat's own caption, which was generated to the burn-in
                    # budget. `content` is the fallback for a model that did not
                    # supply one, and it is the long string, so it is the case that
                    # gets truncated on screen.
                    caption=(beat.caption or beat.content) if j == 0 else "",
                    scene_prompt=fit(body_scene, body_motion),
                    motion_prompt=body_motion,
                    continuation_prompt=(
                        f"{beat.content}. {sizes[j]} shot, {move}. "
                        f"Same subject and location. {looks}."
                    ),
                )
            )

    if squeezed:
        ctx.emit(
            EventKind.WARNING,
            session.run_id,
            stage="storyboard",
            message=(
                f"{len(squeezed)} of {len(shots)} shot prompts were over the "
                f"{limit}-character budget and had style clauses dropped to fit "
                f"(longest {max(squeezed)}). The colour grade is given up first "
                "and qc measures it -- check the saturation and contrast checks."
            ),
        )

    return Storyboard(
        style_id=style.style_id,
        hook_text=hook.selected.text,
        hook_archetype=hook.archetype_sampled,
        shots=ensure_distinct_frames(shots),
    )
