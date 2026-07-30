"""How long generated sentences may be, derived from the constraints they land in.

A peer of `media.py` rather than a helper inside `tools/`, for the same reason: the
budgets are needed by three stages -- `outline`, `hook` and `storyboard` -- and a
tool importing another tool would make the execution order implicit in the import
graph, which a declared plan is supposed to make explicit.

Every number here comes from a limit that already existed and used to be applied
only *after* the text was written:

  * The render endpoint refuses a shot prompt over its per-shot limit (512 on Kling
    3.0). Before that it degrades: `storyboard.fit` buys room by dropping style
    clauses, and the first thing it gives up is the colour grade that QC measures.
  * The caption burn-in keeps the first `CAPTION_MAX_LINES` wrapped lines and
    discards the rest. Text past that is not shortened, it is missing, and nothing
    reports it -- you find out by watching the output.

Neither is caught by a validator, so both are computed before generation and stated
in the prompt. Validation still runs afterwards; this is what stops it from being
the thing that decides the output.

The two budgets are far apart -- roughly 200 characters of prompt against 42 of
caption -- which is why a beat carries a separate sentence for each. One string
serving both is either a thin prompt or a truncated caption.
"""

from __future__ import annotations

from .media import CAPTION_MAX_LINES
from .schema import Casting, StyleSchema

# The motion sentences a shot prompt carries alongside the generated action. They
# live here, not in `storyboard`, because they are part of the fixed cost this
# module has to subtract -- and `outline` needs the figure before `storyboard` runs.
HOOK_MOTION = "Fast and attention-grabbing."
BODY_MOTION = "Natural continuous motion, no cut inside the shot."

# The largest shot-size token, so the budget holds for whichever size is drawn.
_WIDEST_SIZE = "EWS"

# A floor on what a generating stage is asked for. If the fixed part of a prompt is
# so long that nothing is left, the answer is to shorten the style tokens, not to
# ask a model for a four-character sentence -- so the budget bottoms out here and
# `storyboard.fit` does the shrinking instead.
MIN_TEXT_CHARS = 40

# What one shot's action description needs to be a description rather than a label.
# On the bundled reference the fixed tokens ran 415 characters of a 512 budget,
# leaving 40 -- so the generated sentence was starved and `storyboard.fit` covered it
# up per shot by dropping style clauses. The split is decided once here instead.
TARGET_TEXT_CHARS = 120

# What compression may give up: the descriptive keywords, and nothing else. The
# presenter, the location and the three measured colour figures all stay. Identity is
# what a viewer notices breaking, and the colour figures are what QC scores, so
# trading either for a longer sentence would buy prompt detail with the two
# properties the system exists to reproduce.
#
# So the floor is not a number -- it is the tokens minus their keywords, whatever
# those happen to be for this style and this draw.


def caption_chars(style: StyleSchema) -> int:
    """What one caption can hold before the burn-in silently drops the rest."""
    return max(style.caption.max_chars_per_line, 1) * CAPTION_MAX_LINES


def _fixed_overhead(style: StyleSchema) -> int:
    """Everything in a shot prompt that is neither the style tokens nor the action.

    Measured against the *longest* move and motion sentence a run can draw, not an
    average. A budget that only holds for the shortest one is not a budget: the move
    cycles by shot index, so any shot in the video can be the expensive one.
    """
    move = max(style.camera.moves or ["static"], key=len)
    motion = max(HOOK_MOTION, BODY_MOTION, key=len)
    return len(f". {_WIDEST_SIZE} shot, {move}. . {motion}")


def plan_shot_text(
    prompt_limit: int, style: StyleSchema, casting: Casting
) -> tuple[str, int]:
    """Split one shot prompt between the fixed style tokens and the generated action.

    Returns the tokens to use and the characters left for the action, decided once
    for the whole run so that `outline`, `hook` and `storyboard` are working to the
    same numbers. Previously each stage assumed the full tokens and `storyboard.fit`
    reconciled the difference per shot, which meant the reconciliation -- dropping
    the colour grade that QC measures -- was the normal case rather than the
    exception.

    When both cannot fit, the tokens are compressed before the sentence is starved,
    down to a floor: below that they stop naming the presenter and the grade, and
    identity is what a viewer notices breaking.
    """
    full = style_tokens(style, casting)
    if prompt_limit <= 0:
        return full, 0
    room = prompt_limit - _fixed_overhead(style)
    if room - len(full) >= TARGET_TEXT_CHARS:
        return full, room - len(full)
    floor = len(style_tokens(style, casting, keywords=False))
    wanted = max(room - TARGET_TEXT_CHARS, floor)
    tokens = style_tokens(style, casting, budget=wanted)
    return tokens, max(room - len(tokens), MIN_TEXT_CHARS)


def shot_text_chars(prompt_limit: int, style: StyleSchema, casting: Casting) -> int:
    """Just the character budget from `plan_shot_text`, for the generating stages."""
    return plan_shot_text(prompt_limit, style, casting)[1]


def look_tokens(style: StyleSchema) -> str:
    """Grade only. QC measures these, so they go in every entry even when
    identity does not -- a continuation inherits the person, not the grade."""
    look = style.look
    return (
        f"{look.grade} colour grade, "
        f"saturation {look.saturation:.2f}, contrast {look.contrast:.2f}"
    )


def style_tokens(
    style: StyleSchema, casting: Casting, budget: int = 0, keywords: bool = True
) -> str:
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
    if keywords:
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
