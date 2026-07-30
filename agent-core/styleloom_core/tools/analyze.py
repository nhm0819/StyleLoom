"""Reference videos -> style.json.

Not a registered pipeline tool: this produces the reusable *asset* that runs
consume, so it has its own entry point (`styleloom style extract`) and is not
part of any run's plan.

The division of labour here is the point. Pacing and grade are numbers, so they
are measured from pixels with OpenCV; an LLM answering "fast cuts, punchy colour"
is not reproducible and cannot be checked in QC. The LLM is called once, only to
*name* the things that are genuinely linguistic -- the grade, the camera
vocabulary, the tone -- and is told explicitly not to invent numbers.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ..errors import ToolError
from ..media import estimate_bpm, probe_video
from ..schema import (
    AudioStyle,
    Camera,
    CaptionStyle,
    HookStyle,
    Look,
    LookDetail,
    Pacing,
    StyleSchema,
)

if TYPE_CHECKING:
    from ..context import Context

MAX_KEYFRAMES_TO_LLM = 6

# Stated in the extraction prompt and enforced on the way back. These fields are
# carried in *every* shot prompt, against a 512-character per-shot cap the endpoint
# will not raise, so an unbounded phrase here does not merely bloat one request --
# it makes every request in every future run of this style fail budgeting.
#
# The numbers come from what has to coexist in one shot prompt: the presenter, the
# location, the grade, the action and the motion sentence. An extraction that
# answered with a 165-character `motion_feel` and a 120-character `grade` pushed
# the continuation prompt to 630 characters and the endpoint refused every request.
MAX_GRADE_CHARS = 60
MAX_MOTION_CHARS = 80
MAX_DETAIL_CHARS = 90


def clamp_phrase(text: str, limit: int) -> str:
    """Trim a returned phrase to its budget on a clause boundary.

    Whole clauses, never a character count: a phrase cut mid-word is worse in a
    prompt than a shorter one, and the leading clause is the one that carries the
    field's meaning. Falls back to the first clause when even that is over.
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
MAX_CUT_TIMES_STORED = 40


def merge_metrics(metrics: list[dict]) -> dict:
    """Average across reference videos.

    Cut *times* are not averaged -- they are not comparable across clips of
    different lengths -- so the first video's are kept as an illustrative sample.
    """

    def avg(key: str) -> float:
        return round(sum(m[key] for m in metrics) / len(metrics), 3)

    return {
        "duration": avg("duration"),
        "avg_shot_sec": avg("avg_shot_sec"),
        "median_cut_interval": avg("median_cut_interval"),
        "saturation": avg("saturation"),
        "contrast": avg("contrast"),
        "warmth": avg("warmth"),
        "shot_count": int(round(avg("shot_count"))),
        "cuts_in_first_3s": max(1, int(round(avg("cuts_in_first_3s")))),
        "cuts": metrics[0]["cuts"],
    }


def extract_style(
    ctx: Context,
    style_id: str,
    ref_paths: list[Path],
    notes: str = "",
) -> StyleSchema:
    if not ref_paths:
        raise ToolError("at least one reference video is required")

    resolved = [(p, ctx.settings.resolve_ref(p)) for p in ref_paths]
    missing = [given for given, found in resolved if found is None]
    if missing:
        tried = [
            f"{given} (tried {', '.join(str(c) for c in ctx.settings.ref_candidates(given))})"
            for given in missing
        ]
        raise ToolError("reference not found: " + "; ".join(tried))
    found_paths = [found for _, found in resolved if found is not None]

    per_video = [probe_video(p) for p in found_paths]
    keyframes = [kf for m in per_video for kf in m["keyframes"]][:MAX_KEYFRAMES_TO_LLM]
    m = merge_metrics(per_video)

    named = ctx.llm.complete_json(
        task="style_synthesis",
        system=(
            "You are a short-form video editor. Describe an existing style precisely "
            "and reusably. Do not invent numbers -- only name qualitative "
            "attributes.\n"
            "Be specific and physical. `soft warm key light from the upper left` is "
            "usable; `nice lighting` is not. These phrases are pasted into image "
            "generation prompts, so write what a camera would see, not how the video "
            "feels to watch."
        ),
        user=(
            "You are given measured statistics from short-form reference videos.\n"
            f"MEASURED: avg_shot_sec={m['avg_shot_sec']}, duration={m['duration']}, "
            f"saturation={m['saturation']}, contrast={m['contrast']}, "
            f"warmth={m['warmth']}, shot_count={m['shot_count']}, "
            f"cuts_in_first_3s={m['cuts_in_first_3s']}\n"
            "Name the visual style. Return JSON with keys: grade, cut_style, "
            "moves (list), voice_tone, keywords (list), notes, and these five "
            "single-phrase fields describing what the frames look like:\n"
            "  subject   -- what is on screen and how it is framed\n"
            "  lighting  -- direction, hardness and colour of the light\n"
            "  texture   -- surface qualities: skin, product, fabric, moisture\n"
            "  palette   -- the actual colours present, not a measurement\n"
            "  lens      -- depth of field, focal length feel, camera-as-device look\n"
            "plus one more for motion, which a still cannot express:\n"
            "  motion_feel -- what drives the cutting and the movement\n"
            "The five visual fields go into image generation prompts and "
            "motion_feel into video prompts, so keep each to its own concern.\n"
            f"LENGTH LIMITS, all hard: grade at most {MAX_GRADE_CHARS} characters, "
            f"motion_feel at most {MAX_MOTION_CHARS}, each of the five visual "
            f"fields at most {MAX_DETAIL_CHARS}. A video shot prompt is capped at "
            "512 characters by the endpoint and these are carried in every one of "
            "them, so a long phrase here costs the shot its own description. "
            "Write a phrase, not a sentence."
        ),
        temperature=0.3,
        images=keyframes or None,
    )

    return StyleSchema(
        style_id=style_id,
        source_refs=[p.name for p in found_paths],
        total_duration=m["duration"],
        pacing=Pacing(
            avg_shot_sec=max(m["avg_shot_sec"], 0.3),
            shot_count=max(m["shot_count"], 3),
            cut_style=named.get("cut_style", "jump_cut"),
            cut_times=m["cuts"][:MAX_CUT_TIMES_STORED],
        ),
        camera=Camera(moves=named.get("moves") or ["handheld_micro_shake"]),
        look=Look(
            grade=clamp_phrase(str(named.get("grade", "neutral")), MAX_GRADE_CHARS),
            saturation=m["saturation"],
            contrast=m["contrast"],
            warmth=m["warmth"],
            keywords=named.get("keywords") or [],
            detail=LookDetail(
                # Clamped on the way in, not trusted from the prompt. The limit is
                # stated to the model and a model can ignore it -- and this value is
                # written to style.json and reused by every future run, so an
                # over-long phrase accepted once keeps costing shot prompts forever.
                **{
                    field: clamp_phrase(
                        str(named.get(field) or "").strip(),
                        MAX_MOTION_CHARS if field == "motion_feel" else MAX_DETAIL_CHARS,
                    )
                    for field in LookDetail.model_fields
                }
            ),
        ),
        caption=CaptionStyle(),
        audio=AudioStyle(
            bpm=estimate_bpm(m["median_cut_interval"]),
            voice_tone=named.get("voice_tone", "neutral"),
        ),
        hook_style=HookStyle(cut_count=m["cuts_in_first_3s"]),
        notes=(notes + " " + named.get("notes", "")).strip(),
    )
