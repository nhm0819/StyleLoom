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
    Pacing,
    StyleSchema,
)

if TYPE_CHECKING:
    from ..context import Context

MAX_KEYFRAMES_TO_LLM = 6
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
    missing = [p for p in ref_paths if not p.exists()]
    if missing:
        raise ToolError(f"reference not found: {', '.join(str(p) for p in missing)}")

    per_video = [probe_video(p) for p in ref_paths]
    keyframes = [kf for m in per_video for kf in m["keyframes"]][:MAX_KEYFRAMES_TO_LLM]
    m = merge_metrics(per_video)

    named = ctx.llm.complete_json(
        task="style_synthesis",
        system=(
            "You are a short-form video editor. Describe an existing style precisely "
            "and reusably. Do not invent numbers -- only name qualitative attributes."
        ),
        user=(
            "You are given measured statistics from short-form reference videos.\n"
            f"MEASURED: avg_shot_sec={m['avg_shot_sec']}, duration={m['duration']}, "
            f"saturation={m['saturation']}, contrast={m['contrast']}, "
            f"warmth={m['warmth']}, shot_count={m['shot_count']}, "
            f"cuts_in_first_3s={m['cuts_in_first_3s']}\n"
            "Name the visual style. Return JSON with keys: grade, cut_style, "
            "moves (list), voice_tone, keywords (list), notes."
        ),
        temperature=0.3,
        images=keyframes or None,
    )

    return StyleSchema(
        style_id=style_id,
        source_refs=[p.name for p in ref_paths],
        total_duration=m["duration"],
        pacing=Pacing(
            avg_shot_sec=max(m["avg_shot_sec"], 0.3),
            shot_count=max(m["shot_count"], 3),
            cut_style=named.get("cut_style", "jump_cut"),
            cut_times=m["cuts"][:MAX_CUT_TIMES_STORED],
        ),
        camera=Camera(moves=named.get("moves") or ["handheld_micro_shake"]),
        look=Look(
            grade=named.get("grade", "neutral"),
            saturation=m["saturation"],
            contrast=m["contrast"],
            warmth=m["warmth"],
            keywords=named.get("keywords") or [],
        ),
        caption=CaptionStyle(),
        audio=AudioStyle(
            bpm=estimate_bpm(m["median_cut_interval"]),
            voice_tone=named.get("voice_tone", "neutral"),
        ),
        hook_style=HookStyle(cut_count=m["cuts_in_first_3s"]),
        notes=(notes + " " + named.get("notes", "")).strip(),
    )
