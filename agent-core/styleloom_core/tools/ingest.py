"""Normalise arbitrary input into a Brief."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..media import sample_frames, video_duration_note
from ..schema import Brief, InputKind
from .registry import tool

if TYPE_CHECKING:
    from ..context import Context
    from ..session import RunSession

VIDEO_SUFFIXES = {".mp4", ".mov", ".webm", ".mkv", ".m4v"}
MAX_FACTS = 6
MAX_TOPIC_CHARS = 80
# Beginning, middle and end. More frames buy detail the four-field brief cannot
# use.
VIDEO_FRAME_SAMPLES = 3


def classify(path_suffix: str) -> InputKind:
    return "video" if path_suffix.lower() in VIDEO_SUFFIXES else "image"


@tool("ingest", writes="brief")
def ingest(ctx: Context, session: RunSession) -> Brief:
    """Collapse text / image / video input into one Brief."""
    inputs = session.inputs
    kind: InputKind = "text"
    description = inputs.text
    images: list[bytes] | None = None

    if inputs.file_path is not None:
        kind = classify(inputs.file_path.suffix)
        if kind == "video":
            images = sample_frames(inputs.file_path, VIDEO_FRAME_SAMPLES) or None
            note = video_duration_note(inputs.file_path)
            if images:
                note += (
                    f". The {len(images)} attached stills are frames from this "
                    "video in order, from near its beginning, middle and end."
                )
            else:
                note += ". Frames could not be read, so no stills are attached."
        else:
            note = f"image input ({inputs.file_path.name})"
            images = [inputs.file_path.read_bytes()]
        description = f"{inputs.text}\n{note}".strip()

    first_line = description.strip().splitlines()[0] if description.strip() else "unknown"
    parsed = ctx.llm.complete_json(
        task="ingest",
        system=(
            f"Extract a short-form video brief. Answer in {inputs.language}. "
            "Any attached images describe the subject to make a video about, not "
            "a style to copy. Return JSON: topic, audience, key_message, facts "
            "(list of strings)."
        ),
        user=f"TOPIC: {first_line}\n\n{description}",
        temperature=0.4,
        images=images,
    )

    return Brief(
        input_kind=kind,
        raw_text=inputs.text,
        source_path=str(inputs.file_path) if inputs.file_path else None,
        topic=str(parsed.get("topic", ""))[:MAX_TOPIC_CHARS],
        audience=str(parsed.get("audience", "general")),
        key_message=str(parsed.get("key_message", "")),
        facts=[str(f) for f in parsed.get("facts", [])][:MAX_FACTS],
        language=inputs.language,
    )
