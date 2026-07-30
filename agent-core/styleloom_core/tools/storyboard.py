"""Turn style + hook + outline into shot-level prompts."""

from __future__ import annotations

import math
import random
from collections.abc import Callable
from typing import TYPE_CHECKING

from ..budget import look_tokens, motion_sentence, plan_shot_text, style_tokens
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

# A duration small enough that any real grid rounds it up to one unit, and that a
# provider stating no grid returns unchanged. Used to ask a quantiser what its
# smallest renderable cut is without adding a second method to the interface.
_TINY = 1e-6

# What `shot_billed_duration` is, seen from here: a function that maps a wanted cut
# length onto one the endpoint will actually run.
Quantiser = Callable[[float], float]


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


def whole_spans(spans: list[float], quantise: Quantiser) -> list[float]:
    """Snap a sequence of spans onto the endpoint's grid, carrying the error forward.

    Rounding each span on its own accumulates: three beats each rounded up by 0.4s
    make the video 1.2s long, and the total runtime is one of the properties being
    reproduced. Diffusing the error means a beat that rounded up makes the next one
    round down, so the sum stays within half a unit of the target however many beats
    there are.
    """
    out: list[float] = []
    carry = 0.0
    for span in spans:
        wanted = span + carry
        snapped = quantise(wanted)
        carry = wanted - snapped
        out.append(snapped)
    return out


def cut_plan(
    span: float, quantise: Quantiser, count: int = 0, avg: float = 0.0
) -> list[float]:
    """Cut lengths covering `span` that the endpoint can render exactly.

    The storyboard used to divide a span evenly and hand over whatever came out --
    0.76s cuts against an endpoint whose shortest renderable cut is 1s. Nothing
    rejected that: the request quantised silently, so the delivered video ran ~40%
    longer than the storyboard and the drift only showed up in QC on a paid run.

    So the division happens on the endpoint's grid. Every returned length is a
    multiple of the smallest renderable cut, which makes it a fixed point of
    `shot_billed_duration` -- what is asked for is what is billed and what is
    delivered.

    The pacing cost is real and is not hidden: on a reference whose cuts run 0.76s
    the plan plays them at 1s, 31% slower, and QC's `avg_shot_sec` check reports it
    against a tolerance that knows about the floor.

    `count` fixes the number of cuts, for a hook window whose cut count the style
    declares; otherwise `avg` picks it. Either way the count is capped at what the
    span can hold, since a 3s window cannot hold four 1s cuts.
    """
    unit = quantise(_TINY)
    n = count or shot_count_for(span, avg)
    if unit <= _TINY:
        # The endpoint states no grid, so there is nothing to snap to.
        n = max(1, n)
        return [round(span / n, 2)] * n
    units = max(int(round(span / unit)), 1)
    n = max(1, min(n, units))
    base, extra = divmod(units, n)
    return [unit * (base + (1 if i < extra else 0)) for i in range(n)]


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
    quantise = ctx.video.shot_billed_duration
    hook_lengths = cut_plan(
        style.hook_style.window_sec, quantise, count=max(style.hook_style.cut_count, 1)
    )
    for i, hook_shot_sec in enumerate(hook_lengths):
        move = moves[i % len(moves)]
        hook_scene = (
            f"{hook.selected.visual}. {style.hook_style.shot_size} shot, "
            f"{move}. {tokens}. Opening shot."
        )
        hook_motion = motion_sentence(style, hook=True)
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
    beat_spans = whole_spans([b.duration_sec for b in outline.beats], quantise)
    for beat, beat_span in zip(outline.beats, beat_spans, strict=True):
        lengths = cut_plan(
            beat_span, quantise, avg=max(style.pacing.avg_shot_sec, MIN_AVG_SHOT_SEC)
        )
        count = len(lengths)
        sizes = sizes_for(style, count, rng)
        for j, per in enumerate(lengths):
            move = moves[(len(shots) + j) % len(moves)]
            body_scene = (
                f"{beat.content}. {sizes[j]} shot, {move}. {tokens}."
            )
            # Not the move -- that is already in the scene line above.
            body_motion = motion_sentence(style)
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
