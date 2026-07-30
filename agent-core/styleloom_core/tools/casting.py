"""Casting: who is on screen, and where.

The assignment lists creator (person) and background alongside the hook as
elements that should differ between outputs, so that three videos from one style
are three different videos rather than one video with new captions.

Two stages, because they answer two different questions.

  1. *Draw.* A weighted sample from an editable pool, penalised by what this style
     used recently, entropy from `secrets.SystemRandom` -- the same machinery as
     the hook archetype. This is what makes three runs three different castings.
  2. *Specialise.* One LLM call that rewrites the drawn description against this
     run's brief. This is what stops the result being generic.

The draw alone produced the genericness this stage was reported for. A pool entry
has to fit any topic, so it is written neutrally -- "warm and approachable
expression", "clean glowy makeup" -- and neutral phrasing is what makes a frame
look like stock. It was also the source of the reported script mismatch: the pool
could cast a man for a brief whose every beat describes a woman, because nothing
connected the two.

So the seed supplies variety and the brief supplies specificity, and where they
disagree about the person, the brief wins -- it is the thing the script was
written about. Two consequences worth being explicit about:

  * The choices are still *system-generated*. Only the brief's content fields
    reach this stage, never the user's raw text, so writing "cast a man in his
    20s" in the input still does not pin the casting.
  * They are still non-deterministic. The seed is drawn per run, and the run's
    `casting.json` records both the pools it drew from and the seed text each
    description was specialised from.

`id` remains the pool entry's id, not the specialised text: recency, the pool
listing and the history are all keyed on it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..budget import clamp_phrase
from ..errors import LLMError
from ..sampling import ENTROPY_SOURCE, load_pool, new_rng, sample_with_recency
from ..schema import Brief, CastChoice, Casting, Choice, StyleSchema
from .registry import tool

if TYPE_CHECKING:
    from ..context import Context
    from ..session import RunSession

CREATOR_KEY = "creators"
SETTING_KEY = "settings"

# Stated in the prompt and enforced on the way back. These two descriptions are
# carried in *every* shot prompt against a 512-character per-shot cap, so an
# unbounded one does not bloat a single request -- `budget.plan_shot_text` pays for
# it by starving the shot's own action sentence, in every shot of the video.
#
# 160 rather than a round number: the longest bundled pool entry is 152 characters
# and the budget arithmetic is tuned around entries of that size, so this leaves
# room to be more specific without moving the figure the rest of the run assumes.
MAX_CAST_CHARS = 160

# Low. This call is not where run-to-run variety comes from -- the draw is -- and a
# hot specialisation would wander off the seed, which is the thing keeping three
# runs visibly different from each other.
SPECIALISE_TEMPERATURE = 0.4


def _to_choice(entry: dict) -> CastChoice:
    prompt = _flatten(entry.get("prompt", ""))
    return CastChoice(
        id=str(entry["id"]),
        label=str(entry.get("label", entry["id"])),
        prompt=prompt,
        seed_prompt=prompt,
    )


def _flatten(text: object) -> str:
    """Collapse YAML folded scalars: a stray newline inside a prompt is a token
    boundary for some image models."""
    return " ".join(str(text or "").split())


def specialise(
    ctx: Context, brief: Brief, style: StyleSchema, creator: CastChoice, setting: CastChoice
) -> tuple[CastChoice, CastChoice]:
    """Rewrite the two drawn descriptions against this run's brief.

    Returns the pair unchanged if the model gives nothing usable. The draw is a
    complete casting on its own -- it was the whole stage until now -- so a failure
    here degrades to the previous behaviour rather than failing the run, and a
    provider with no LLM at all still produces video.
    """
    facts = "; ".join(brief.facts[:4])
    try:
        parsed = ctx.llm.complete_json(
            task="casting_detail",
            system=(
                "You cast the on-screen presenter and the location for one "
                "short-form video, by rewriting two draft descriptions so they suit "
                "this specific video.\n"
                "Return JSON: {creator: str, setting: str}. Each is a "
                "comma-separated list of visual tokens in English, never a "
                "sentence -- image models weight comma-separated tokens more "
                "evenly than prose.\n"
                "Keep the draft's broad casting -- who this person roughly is, what "
                "kind of place this is -- and replace its generic filler with "
                "detail this topic makes concrete: what they are wearing for this, "
                "what is within reach, what the place is being used for. Generic "
                "praise of a face is what makes a frame look like stock; specifics "
                "are what stop it.\n"
                "Where the brief states something about the person the draft "
                "contradicts -- their gender, their age, their occupation -- the "
                "brief wins and you change the draft. The brief is what the script "
                "was written about; the draft is only a starting point.\n"
                f"HARD LIMIT: each field at most {MAX_CAST_CHARS} characters. Both "
                "are carried in every shot prompt against a 512-character cap, so "
                "characters spent here come out of the shot's own description. "
                "Write short rather than truncating a clause."
            ),
            user=(
                f"DRAFT_CREATOR: {creator.prompt}\n"
                f"DRAFT_SETTING: {setting.prompt}\n\n"
                f"TOPIC: {brief.topic}\n"
                f"KEY_MESSAGE: {brief.key_message}\n"
                f"AUDIENCE: {brief.audience}\n"
                f"FACTS: {facts}\n\n"
                f"REFERENCE_SUBJECT: {style.look.detail.subject}\n"
                f"REFERENCE_LIGHTING: {style.look.detail.lighting}\n"
                f"REFERENCE_PALETTE: {style.look.detail.palette}\n"
                f"MAX_CHARS: {MAX_CAST_CHARS}\n"
            ),
            temperature=SPECIALISE_TEMPERATURE,
        )
    except LLMError:
        return creator, setting

    return (
        _specialised(creator, parsed.get("creator")),
        _specialised(setting, parsed.get("setting")),
    )


def _specialised(seed: CastChoice, text: object) -> CastChoice:
    """One returned description, clamped, or the seed if it is unusable.

    Clamped rather than trusted: the limit is stated in the prompt and a model can
    ignore it. Whole clauses only -- `clamp_phrase` -- because a description cut
    mid-word reaches the render endpoint as half a token list.
    """
    written = _flatten(text)
    if not written:
        return seed
    return seed.model_copy(update={"prompt": clamp_phrase(written, MAX_CAST_CHARS)})


@tool("casting", reads=("style", "brief"), writes="casting")
def casting(ctx: Context, session: RunSession) -> Casting:
    """Cast the on-screen creator and the location for this run."""
    style = session.get("style", StyleSchema)
    brief = session.get("brief", Brief)

    settings = ctx.settings
    path = settings.resolve_config(settings.casting_path)
    rng = new_rng()

    creator_pool = load_pool(path, CREATOR_KEY)
    setting_pool = load_pool(path, SETTING_KEY)

    creator = _to_choice(
        sample_with_recency(
            creator_pool,
            ctx.history.recent_values(session.style_id, "creator"),
            rng,
            penalty=settings.hook_recency_penalty,
        )
    )
    setting = _to_choice(
        sample_with_recency(
            setting_pool,
            ctx.history.recent_values(session.style_id, "setting"),
            rng,
            penalty=settings.hook_recency_penalty,
        )
    )

    creator, setting = specialise(ctx, brief, style, creator, setting)

    result = Casting(
        creator=creator,
        setting=setting,
        creator_pool=[c["id"] for c in creator_pool],
        setting_pool=[s["id"] for s in setting_pool],
        entropy_source=ENTROPY_SOURCE,
    )

    # No reference portrait: text-to-video has no image input to feed one into.
    # The creator reaches the output as the description below, which fixes the
    # person's type across cuts, not their face. See README.

    # Recorded immediately, so the next run in a batch sees these choices even if
    # this run later fails during render.
    for kind, choice in (("creator", creator), ("setting", setting)):
        ctx.history.append(
            session.style_id,
            Choice(
                run_id=session.run_id,
                kind=kind,  # type: ignore[arg-type]
                value=choice.id,
                detail=choice.label,
            ),
        )
    return result
