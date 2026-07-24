"""Stage 2-3 - Ingest and outline.

Ingest collapses text / image / video input into one `Brief`. Outline turns the
Brief into body beats **and a payoff**.

Ordering matters: the outline runs *before* the hook. A hook is a promise that
the body pays off, so generating it first would make it a generic attention
grabber rather than one that fits this specific video.
"""

from __future__ import annotations

from pathlib import Path

import cv2

from ..providers.llm import BaseLLM
from ..schema import Beat, Brief, Outline, StyleSchema


def _video_first_frame_note(path: Path) -> str:
    cap = cv2.VideoCapture(str(path))
    ok, _ = cap.read()
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    if not ok:
        return f"video input ({path.name}), unreadable"
    return f"video input ({path.name}), ~{total / fps:.1f}s"


def ingest(
    llm: BaseLLM,
    text: str = "",
    file_path: Path | None = None,
    language: str = "ko",
) -> Brief:
    kind = "text"
    description = text
    if file_path is not None:
        suffix = file_path.suffix.lower()
        if suffix in {".mp4", ".mov", ".webm", ".mkv"}:
            kind = "video"
            description = f"{text}\n{_video_first_frame_note(file_path)}".strip()
        else:
            kind = "image"
            description = f"{text}\nimage input ({file_path.name})".strip()

    images = None
    if kind == "image" and file_path is not None:
        images = [file_path.read_bytes()]

    parsed = llm.complete_json(
        task="ingest",
        system=(
            f"Extract a short-form video brief. Answer in {language}. "
            "Return JSON: topic, audience, key_message, facts (list of strings)."
        ),
        user=f"TOPIC: {description.strip().splitlines()[0] if description.strip() else 'unknown'}\n\n{description}",
        temperature=0.4,
        images=images,
    )

    return Brief(
        input_kind=kind,  # type: ignore[arg-type]
        raw_text=text,
        source_path=str(file_path) if file_path else None,
        topic=parsed.get("topic", "")[:80],
        audience=parsed.get("audience", "general"),
        key_message=parsed.get("key_message", ""),
        facts=[str(f) for f in parsed.get("facts", [])][:6],
        language=language,
    )


def build_outline(llm: BaseLLM, brief: Brief, style: StyleSchema) -> Outline:
    body_beats = [b for b in style.beat_pattern if b != "hook"] or ["context", "payoff"]
    body_budget = max(style.total_duration - style.hook_style.window_sec, 6.0)
    per_beat = round(body_budget / len(body_beats), 2)

    parsed = llm.complete_json(
        task="outline",
        system=(
            f"You write short-form video outlines in {brief.language}. "
            "Return JSON: payoff (string), beats (list of {name, intent, content, duration_sec}). "
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
            duration_sec=max(float(b.get("duration_sec", per_beat)), 0.5),
        )
        for i, b in enumerate(raw_beats)
    ]
    if not beats:
        beats = [
            Beat(name=n, intent="", content=brief.key_message, duration_sec=per_beat)
            for n in body_beats
        ]

    return Outline(
        beats=_fit_to_budget(beats, body_budget),
        payoff=str(parsed.get("payoff", brief.key_message)),
    )


def _fit_to_budget(beats: list[Beat], budget: float) -> list[Beat]:
    """Rescale beat durations so the body fits the reference's total length.

    Models routinely ignore a stated per-beat budget, and total runtime is one of
    the style properties being reproduced. Enforcing it here keeps it out of the
    prompt's hands -- proportions stay as the model intended, the sum does not.
    """
    total = sum(b.duration_sec for b in beats)
    if total <= 0:
        return beats
    factor = budget / total
    for beat in beats:
        beat.duration_sec = round(max(beat.duration_sec * factor, 0.5), 2)
    return beats
