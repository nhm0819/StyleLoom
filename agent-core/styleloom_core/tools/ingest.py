"""Normalise arbitrary input into a Brief."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..media import video_duration_note
from ..schema import Brief, InputKind
from .registry import tool

if TYPE_CHECKING:
    from ..context import Context
    from ..session import RunSession

VIDEO_SUFFIXES = {".mp4", ".mov", ".webm", ".mkv", ".m4v"}
MAX_FACTS = 6
MAX_TOPIC_CHARS = 80


def classify(path_suffix: str) -> InputKind:
    return "video" if path_suffix.lower() in VIDEO_SUFFIXES else "image"


@tool("ingest", writes="brief")
def ingest(ctx: Context, session: RunSession) -> Brief:
    """Collapse text / image / video input into one Brief."""
    inputs = session.inputs
    kind: InputKind = "text"
    description = inputs.text

    if inputs.file_path is not None:
        kind = classify(inputs.file_path.suffix)
        if kind == "video":
            note = video_duration_note(inputs.file_path)
        else:
            note = f"image input ({inputs.file_path.name})"
        description = f"{inputs.text}\n{note}".strip()

    images = None
    if kind == "image" and inputs.file_path is not None:
        # Only stills go to the vision call. A video's frames would need
        # sampling, which is the reference-analysis job, not the ingest job.
        images = [inputs.file_path.read_bytes()]

    first_line = description.strip().splitlines()[0] if description.strip() else "unknown"
    parsed = ctx.llm.complete_json(
        task="ingest",
        system=(
            f"Extract a short-form video brief. Answer in {inputs.language}. "
            "Return JSON: topic, audience, key_message, facts (list of strings)."
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
