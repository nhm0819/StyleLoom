"""Hook generation. The graded core of the system.

Requirement: the hook must be produced by the system rather than typed by the
user, must fit the whole video's context, and must be able to differ across
reruns of the same input.

Relying on LLM `temperature` alone would satisfy the last part only by accident.
Variation is engineered at three independent points instead:

  1. Archetype sampling   weighted draw from configs/archetypes.yaml, with a
                          recency penalty read from this style's history. A
                          different archetype gives a structurally different
                          opening, not a paraphrase.
  2. Candidate generation N candidates, temperature 0.9, no fixed seed.
  3. Stochastic selection top-k softmax sample instead of argmax, so the best
                          candidate is favoured but not guaranteed.

Entropy comes from `secrets.SystemRandom` -- OS entropy, unseeded -- so a run can
never accidentally repeat. Every decision is written to `hook.json`, which makes
the non-determinism auditable rather than asserted.

This tool runs *after* the outline on purpose. A hook is a promise the body has
to pay off; generating it first would produce a generic attention grabber, while
generating it here lets it reference the actual payoff.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..config import Settings
from ..errors import ToolError
from ..sampling import (
    ENTROPY_SOURCE,
    load_pool,
    new_rng,
    sample_with_recency,
    softmax_sample,
)
from ..schema import Brief, Choice, HookCandidate, HookResult, Outline, StyleSchema
from .registry import tool

if TYPE_CHECKING:
    from ..context import Context
    from ..session import RunSession

# Weights for combining the model's self-reported sub-scores into one number.
# context_fit dominates because a hook that does not match the body is the one
# failure mode that makes the whole video misleading.
SCORE_WEIGHTS = {"context_fit": 0.5, "style_fit": 0.3, "novelty": 0.2}


def load_archetypes(settings: Settings, path: Path | None = None) -> list[dict[str, Any]]:
    """Load the archetype pool.

    `STYLELOOM_ARCHETYPES_PATH` wins, so someone can drop in their own archetype
    set for their own reference channel without touching the repo.
    """
    src = path or settings.resolve_config(settings.archetypes_path)
    return load_pool(src, "archetypes")


def dedupe(candidates: list[HookCandidate]) -> list[HookCandidate]:
    """Keep the best-scoring candidate per distinct text.

    Duplicates make the softmax draw a no-op: sampling among five copies of the
    same line is indistinguishable from argmax, which would quietly remove one of
    the three randomisation points.
    """
    best: dict[str, HookCandidate] = {}
    for c in candidates:
        if not c.text:
            continue
        if c.text not in best or c.score > best[c.text].score:
            best[c.text] = c
    return list(best.values())


def generate(
    ctx: Context,
    brief: Brief,
    outline: Outline,
    style: StyleSchema,
    recent_archetypes: list[str] | None = None,
    candidate_count: int | None = None,
) -> HookResult:
    """Generate one hook. Callable directly, which is how `hook preview` gets N
    independent generations without creating N runs."""
    settings = ctx.settings
    rng = new_rng()  # OS entropy: not seedable, never repeats a run
    archetypes = load_archetypes(settings)
    n = candidate_count or settings.hook_candidate_count

    chosen = sample_with_recency(
        archetypes,
        recent_archetypes or [],
        rng,
        penalty=settings.hook_recency_penalty,
    )

    body_summary = " / ".join(f"[{b.name}] {b.content}" for b in outline.beats)
    parsed = ctx.llm.complete_json(
        task="hook_candidates",
        system=(
            f"You write the first {style.hook_style.window_sec:.0f} seconds of "
            f"short-form videos, in {brief.language}. The hook must promise exactly "
            "what the body delivers -- never bait something the video does not pay "
            "off.\n"
            "Return JSON: candidates = list of "
            "{archetype, text, visual, context_fit, style_fit, novelty, rationale}. "
            "context_fit / style_fit / novelty are floats in [0,1]. "
            "`text` is the on-screen caption, max 16 characters per line. "
            "`visual` describes what is on screen, in one sentence."
        ),
        user=(
            f"ARCHETYPE: {chosen['id']}\n"
            f"ARCHETYPE_RULE: {chosen.get('pattern', '')}\n"
            f"ARCHETYPE_EXAMPLES: {chosen.get('fewshot', [])}\n"
            f"CANDIDATE_COUNT: {n}\n\n"
            f"TOPIC: {brief.topic}\n"
            f"KEY_MESSAGE: {brief.key_message}\n"
            f"AUDIENCE: {brief.audience}\n"
            f"BODY_BEATS: {body_summary}\n"
            f"PAYOFF_THE_HOOK_MUST_PROMISE: {outline.payoff}\n\n"
            f"STYLE_SHOT_SIZE: {style.hook_style.shot_size}\n"
            f"STYLE_CUTS_IN_WINDOW: {style.hook_style.cut_count}\n"
            f"STYLE_CAMERA_MOVES: {style.camera.moves}\n"
            f"STYLE_TONE: {style.audio.voice_tone}\n"
        ),
        temperature=settings.hook_temperature,
    )

    candidates: list[HookCandidate] = []
    for c in parsed.get("candidates", []):
        score = sum(float(c.get(key, 0.5)) * w for key, w in SCORE_WEIGHTS.items())
        candidates.append(
            HookCandidate(
                archetype=str(c.get("archetype", chosen["id"])),
                text=str(c.get("text", "")).strip(),
                visual=str(c.get("visual", "")).strip(),
                score=round(score, 4),
                rationale=str(c.get("rationale", "")),
            )
        )

    candidates = dedupe(candidates)
    if not candidates:
        raise ToolError("hook generation produced no usable candidates")

    selected, method = softmax_sample(
        candidates, settings.hook_top_k, settings.hook_softmax_temp, rng
    )

    return HookResult(
        archetype_pool=[a["id"] for a in archetypes],
        archetype_sampled=chosen["id"],
        candidates=candidates,
        selected=selected,
        selection_method=method,
        temperature=settings.hook_temperature,
        entropy_source=ENTROPY_SOURCE,
    )


@tool("hook", reads=("style", "brief", "outline"), writes="hook")
def hook(ctx: Context, session: RunSession) -> HookResult:
    """Generate the first three seconds, non-deterministically."""
    style = session.get("style", StyleSchema)
    brief = session.get("brief", Brief)
    outline = session.get("outline", Outline)

    result = generate(
        ctx,
        brief,
        outline,
        style,
        recent_archetypes=ctx.history.recent_values(session.style_id, "hook"),
    )

    # Recorded immediately rather than at the end of the run: the next run in a
    # batch reads this history to avoid repeating the archetype, and a run that
    # fails during render still consumed this choice.
    ctx.history.append(
        session.style_id,
        Choice(
            run_id=session.run_id,
            kind="hook",
            value=result.archetype_sampled,
            detail=result.selected.text,
        ),
    )
    session.mark(
        hook_text=result.selected.text,
        hook_archetype=result.archetype_sampled,
    )
    return result
