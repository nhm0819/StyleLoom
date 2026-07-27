"""Casting: who is on screen, and where.

The assignment lists creator (person) and background alongside the hook as
elements that should differ between outputs, so that three videos from one style
are three different videos rather than one video with new captions.

They are drawn with the same machinery as the hook archetype -- weighted sample
from an editable pool, penalised by what this style used recently, entropy from
`secrets.SystemRandom`. Two consequences worth being explicit about:

  * The choices are *system-generated*. Nothing about the creator or the location
    comes from the user's input, which is only a topic.
  * They are non-deterministic. Rerunning the same input can cast a different
    presenter in a different room, and the run's `casting.json` records the pools
    it drew from.

When the video provider can consume a reference image, this tool also generates one
portrait of the cast creator and hands it to every shot, so the presenter stays the
same person across cuts. Providers that cannot use it never pay for it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..events import EventKind
from ..sampling import ENTROPY_SOURCE, load_pool, new_rng, sample_with_recency
from ..schema import Brief, CastChoice, Casting, Choice, StyleSchema
from .registry import tool

if TYPE_CHECKING:
    from ..context import Context
    from ..session import RunSession

CREATOR_KEY = "creators"
SETTING_KEY = "settings"


def _to_choice(entry: dict) -> CastChoice:
    return CastChoice(
        id=str(entry["id"]),
        label=str(entry.get("label", entry["id"])),
        # Collapse YAML folded scalars: a stray newline inside a prompt is a
        # token boundary for some image models.
        prompt=" ".join(str(entry.get("prompt", "")).split()),
    )


def creator_portrait_prompt(creator: CastChoice, style: StyleSchema) -> str:
    """A neutral reference portrait, deliberately without the scene.

    The reference image exists to fix identity, so including the background would
    fight the per-shot setting prompt rather than support it.
    """
    return (
        f"{creator.prompt}. Head and shoulders portrait, facing camera, "
        f"neutral seamless background, even soft lighting, photorealistic, "
        f"{style.look.grade} colour grade, vertical 9:16 framing."
    )


@tool("casting", reads=("style", "brief"), writes="casting")
def casting(ctx: Context, session: RunSession) -> Casting:
    """Cast the on-screen creator and the location for this run."""
    style = session.get("style", StyleSchema)
    session.get("brief", Brief)  # declared dependency; casting is topic-agnostic

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

    result = Casting(
        creator=creator,
        setting=setting,
        creator_pool=[c["id"] for c in creator_pool],
        setting_pool=[s["id"] for s in setting_pool],
        entropy_source=ENTROPY_SOURCE,
    )

    # Identity anchor, but only where it can be used. Generating a portrait for a
    # provider that ignores reference images would bill a keyframe for nothing.
    if ctx.video.supports_persona and session.inputs.persona_ref is None:
        try:
            result.creator_ref = ctx.video.keyframe(
                creator_portrait_prompt(creator, style),
                session.workspace("cast") / "creator.jpg",
            )
        except Exception as exc:  # noqa: BLE001 - degrade, do not fail the run
            ctx.emit(
                EventKind.WARNING,
                session.run_id,
                stage="casting",
                message=f"creator reference portrait failed, continuing without it: {exc}",
            )

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
