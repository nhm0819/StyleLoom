"""Build the body beats, and the payoff the hook will have to promise."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..schema import Beat, Brief, Outline, StyleSchema
from .registry import tool

if TYPE_CHECKING:
    from ..context import Context
    from ..session import RunSession

MIN_BEAT_SEC = 0.5
# Enough room for a few beats at MIN_BEAT_SEC, and nothing more. It exists only to
# stop a degenerate reference producing a zero-length body; whenever it binds, the
# reference's own total runtime has been overruled.
#
# It was 6.0s, which is a long-form number in a short-form system: any reference
# under (6 + hook window) seconds silently got a longer body than it measured. A
# 7.25s reference with a 3s hook needs a 4.25s body and was given 6s, so the
# output ran 9s against a style that says 7.25 -- 24% over, on a property QC is
# supposed to be checking. It went unnoticed because the total_duration tolerance
# was 6.0s absolute, wide enough to swallow the entire error.
MIN_BODY_BUDGET = 2.0


def fit_to_budget(beats: list[Beat], budget: float) -> list[Beat]:
    """Rescale beat durations so the body fits the reference's total length.

    Models routinely ignore a stated per-beat budget, and total runtime is one of
    the style properties being reproduced. Enforcing it here keeps it out of the
    prompt's hands: proportions stay as the model intended, the sum does not.
    """
    total = sum(b.duration_sec for b in beats)
    if total <= 0:
        return beats
    factor = budget / total
    for beat in beats:
        beat.duration_sec = round(max(beat.duration_sec * factor, MIN_BEAT_SEC), 2)
    return beats


@tool("outline", reads=("style", "brief"), writes="outline")
def outline(ctx: Context, session: RunSession) -> Outline:
    """Turn the Brief into body beats plus an explicit payoff."""
    style = session.get("style", StyleSchema)
    brief = session.get("brief", Brief)

    body_beats = [b for b in style.beat_pattern if b != "hook"] or ["context", "payoff"]
    body_budget = max(style.total_duration - style.hook_style.window_sec, MIN_BODY_BUDGET)
    per_beat = round(body_budget / len(body_beats), 2)

    parsed = ctx.llm.complete_json(
        task="outline",
        system=(
            f"You write short-form video outlines in {brief.language}. "
            "Return JSON: payoff (string), beats "
            "(list of {name, intent, content, duration_sec}). "
            "Do NOT write the hook -- it is generated separately."
        ),
        user=(
            f"TOPIC: {brief.topic}\n"
            f"KEY_MESSAGE: {brief.key_message}\n"
            f"AUDIENCE: {brief.audience}\n"
            f"FACTS: {brief.facts}\n"
            f"BEAT_NAMES: {body_beats}\n"
            f"SECONDS_PER_BEAT: {per_beat}\n"
        ),
        temperature=0.7,
    )

    raw_beats = parsed.get("beats") or []
    beats = [
        Beat(
            name=str(b.get("name", body_beats[min(i, len(body_beats) - 1)])),
            intent=str(b.get("intent", "")),
            content=str(b.get("content", "")),
            duration_sec=max(float(b.get("duration_sec", per_beat)), MIN_BEAT_SEC),
        )
        for i, b in enumerate(raw_beats)
    ]
    if not beats:
        beats = [
            Beat(name=n, intent="", content=brief.key_message, duration_sec=per_beat)
            for n in body_beats
        ]

    return Outline(
        beats=fit_to_budget(beats, body_budget),
        payoff=str(parsed.get("payoff", brief.key_message)),
    )
