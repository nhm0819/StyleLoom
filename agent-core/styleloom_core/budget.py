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

# Kept when the reference has its own motion description: "no cut inside the shot" is
# an instruction the model needs in multi-shot mode, where it decides where the cuts
# inside a window fall. The rest of those two sentences is flavour.
_NO_INNER_CUT = "No cut inside the shot."


def motion_sentence(style: StyleSchema, hook: bool = False) -> str:
    """What a shot prompt says about movement.

    The reference's own `motion_feel` when it has one. The two constants above were
    the fallback *and* the only thing ever sent -- every video this system produced
    said "Natural continuous motion, no cut inside the shot", which is exactly the
    filler that makes output read as generic.

    This is also where the phrase belongs on a budget. A start frame carries the
    look, so the scene tokens do not need to describe it, but no still can express
    cutting rhythm -- and putting it here replaces a sentence of similar length
    instead of competing with the presenter for the same 512 characters.
    """
    feel = style.look.detail.motion_feel.strip()
    if not feel:
        return HOOK_MOTION if hook else BODY_MOTION
    return feel if hook else f"{feel}. {_NO_INNER_CUT}"

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


def clamp_phrase(text: str, limit: int) -> str:
    """Trim a returned phrase to its budget on a clause boundary.

    Whole clauses, never a character count: a phrase cut mid-word is worse in a
    prompt than a shorter one, and the leading clause is the one that carries the
    field's meaning. Falls back to the first clause when even that is over.

    Here rather than in `analyze`, where it was written, because `casting` now
    needs it too and tools must not import each other. It belongs in this module
    on its own terms: it is the enforcement half of the budgets above.
    """
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    clauses = [c.strip() for c in text.split(",") if c.strip()]
    kept: list[str] = []
    for clause in clauses:
        candidate = ", ".join([*kept, clause])
        if kept and len(candidate) > limit:
            break
        kept.append(clause)
    return ", ".join(kept)[:limit].rstrip(", ")


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
    motion = max(motion_sentence(style, hook=True), motion_sentence(style), key=len)
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
    floor = len(essential_tokens(style, casting))
    wanted = max(room - TARGET_TEXT_CHARS, floor)
    tokens = style_tokens(style, casting, budget=wanted)
    return tokens, max(room - len(tokens), MIN_TEXT_CHARS)


def shot_text_chars(prompt_limit: int, style: StyleSchema, casting: Casting) -> int:
    """Just the character budget from `plan_shot_text`, for the generating stages."""
    return plan_shot_text(prompt_limit, style, casting)[1]


def descriptive_phrases(style: StyleSchema, for_image: bool) -> list[str]:
    """The reference's visual vocabulary, for the stage that can act on it.

    Falls back to `look.keywords` when `look.detail` is empty, which is every style
    extracted before the split existed. The fallback is deliberately unfiltered: a
    flat keyword list cannot be sorted into image and motion phrases after the fact,
    so an old style sends all of it and a re-extracted one sends the right subset.
    """
    detail = style.look.detail
    if detail.any_set():
        return detail.image_phrases() if for_image else detail.video_phrases()
    return list(style.look.keywords)


def look_tokens(style: StyleSchema, for_image: bool = False) -> str:
    """The grade, named, plus whatever description the stage can use.

    The measured `saturation`/`contrast` figures used to be here and are gone. They
    were put in prompts because QC scores them, which confused two different jobs: QC
    reads them from `style.json`, and no image or video model interprets
    "saturation 0.35". They cost 30 characters of a 512-character shot budget and
    displaced the phrases that actually drive colour -- the grade name, the palette
    and the lighting. The grade *name* stays, because that is language.
    """
    parts = [f"{style.look.grade} colour grade"]
    parts += descriptive_phrases(style, for_image=for_image)
    return ", ".join(p for p in parts if p)


def token_parts(style: StyleSchema, casting: Casting) -> list[str]:
    """The three things a shot prompt names, in the order the model reads them."""
    return [
        casting.creator.prompt,
        f"in {casting.setting.prompt}",
        f"{style.look.grade} colour grade",
    ]


def essential_tokens(style: StyleSchema, casting: Casting) -> str:
    """The shortest form that still names all three: each one's head clause."""
    return ", ".join(p.split(",")[0].strip() for p in token_parts(style, casting))


def style_tokens(
    style: StyleSchema, casting: Casting, budget: int = 0
) -> str:
    """The look, the presenter and the location, compressed into prompt tokens.

    Casting comes first: identity is what viewers notice breaking, so it should
    not sit at the tail where models weight it least.

    `budget` caps the result. Emission order and drop order are deliberately
    different: casting is emitted first so the model weights it heavily, and given
    up first because colour is what QC measures. Only whole clauses are removed --
    cutting to a character count would hand the model half a sentence.
    """
    parts = token_parts(style, casting)
    # No framing token. Under text-to-video `aspect_ratio` is an explicit request
    # field, so spelling "vertical 9:16" into the prompt spends characters restating
    # what the API already knows.
    #
    # No saturation/contrast figures either -- see `look_tokens`. They were the last
    # thing this compression gave up, on the grounds that QC measures them, which
    # meant a 512-character budget spent its final characters on two numbers no model
    # reads while dropping the phrases that describe the shot.
    #
    # No descriptive phrases: those go to the image prompts, which have 2500
    # characters, and the motion one goes in `motion_sentence`.
    parts = [p for p in parts if p]

    if budget > 0:
        # Trailing clauses go before whole tokens do. All three of these are
        # comma-separated prose -- a location prompt runs past 110 characters and is a
        # head ("in a sunlit kitchen") followed by qualifiers -- so trimming from the
        # tail keeps the thing being named while freeing most of its length. Dropping
        # the token outright loses the place entirely, which is a far bigger loss than
        # losing its adjectives, and at 512 characters the difference decides whether
        # the grade survives at all.
        #
        # Order is expendability, ranked by how completely the start frame that opens
        # the generation already carries each one: the image *is* its colour grade,
        # the location is visible in it, and the presenter is what a viewer notices
        # breaking -- so the presenter is trimmed last and dropped last.
        order = [2, 1, 0]

        def joined() -> str:
            return ", ".join(p for p in parts if p)

        while len(joined()) > budget:
            trimmed = False
            for i in order:
                if "," in parts[i]:
                    parts[i] = parts[i].rsplit(",", 1)[0].strip()
                    trimmed = True
                    break
            if trimmed:
                continue
            # Every token is down to its head and it still does not fit.
            for i in order:
                if parts[i]:
                    parts[i] = ""
                    break
            else:
                break
        parts = [p for p in parts if p]
    return ", ".join(parts)
