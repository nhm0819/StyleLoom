"""Stage 1 - Style decomposition.

Reference video(s) -> `style.json`.

Measured with OpenCV rather than asked of an LLM, because pacing and grade are
numbers: an LLM guessing "fast cuts" is not reproducible, `avg_shot_sec=1.34`
is. The LLM is used only for the parts that genuinely need language (naming the
grade, the camera vocabulary, the tone).

BPM is estimated from cut intervals rather than audio onsets. Short-form edits
overwhelmingly cut on the beat, and this avoids a heavy audio dependency. It is
an estimate and is labelled as such.
"""

from __future__ import annotations

import statistics
from pathlib import Path

import cv2
import numpy as np

from ..providers.llm import BaseLLM
from ..schema import (
    AudioStyle,
    Camera,
    CaptionStyle,
    HookStyle,
    Look,
    Pacing,
    StyleSchema,
)

CUT_THRESHOLD = 0.38  # normalised HSV histogram distance
MIN_SHOT_SEC = 0.25


def _hist(frame: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
    return cv2.normalize(h, h).flatten()


def probe_video(path: Path, sample_stride: int = 2) -> dict:
    """Single pass: cut times + colour statistics + keyframes."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"cannot open video: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = total / fps if total else 0.0

    cuts: list[float] = []
    sats: list[float] = []
    vals: list[float] = []
    warms: list[float] = []
    keyframes: list[bytes] = []

    prev_hist = None
    last_cut = 0.0
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % sample_stride == 0:
            t = idx / fps
            hist = _hist(frame)
            if prev_hist is not None:
                dist = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA)
                if dist > CUT_THRESHOLD and (t - last_cut) >= MIN_SHOT_SEC:
                    cuts.append(round(t, 3))
                    last_cut = t
                    if len(keyframes) < 8:
                        ok_enc, buf = cv2.imencode(".jpg", frame)
                        if ok_enc:
                            keyframes.append(buf.tobytes())
            prev_hist = hist

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            sats.append(float(hsv[..., 1].mean()) / 255.0)
            vals.append(float(frame.std()) / 128.0)
            b, g, r = (float(frame[..., i].mean()) for i in range(3))
            warms.append(r / max(r + b, 1e-6))
        idx += 1
    cap.release()

    if duration == 0.0:
        duration = idx / fps

    shot_count = len(cuts) + 1
    avg_shot = duration / shot_count if shot_count else duration
    intervals = [b - a for a, b in zip(cuts, cuts[1:])] if len(cuts) > 1 else []
    median_interval = statistics.median(intervals) if intervals else avg_shot

    return {
        "duration": round(duration, 2),
        "fps": round(fps, 2),
        "cuts": cuts,
        "shot_count": shot_count,
        "avg_shot_sec": round(avg_shot, 3),
        "median_cut_interval": round(median_interval, 3),
        "saturation": round(float(np.mean(sats)) if sats else 0.5, 3),
        "contrast": round(min(float(np.mean(vals)) if vals else 0.5, 1.0), 3),
        "warmth": round(float(np.mean(warms)) if warms else 0.5, 3),
        "cuts_in_first_3s": len([c for c in cuts if c <= 3.0]),
        "keyframes": keyframes,
    }


def _estimate_bpm(median_interval: float) -> float:
    """Assume cuts land on beats; fold into a musical 70-160 range."""
    if median_interval <= 0:
        return 120.0
    bpm = 60.0 / median_interval
    while bpm < 70:
        bpm *= 2
    while bpm > 160:
        bpm /= 2
    return round(bpm, 1)


def _merge(metrics: list[dict]) -> dict:
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
    style_id: str,
    ref_paths: list[Path],
    llm: BaseLLM,
    notes: str = "",
) -> StyleSchema:
    per_video = [probe_video(p) for p in ref_paths]
    keyframes = [kf for m in per_video for kf in m["keyframes"]][:6]
    m = _merge(per_video)

    user = (
        "You are given measured statistics from short-form reference videos.\n"
        f"MEASURED: avg_shot_sec={m['avg_shot_sec']}, duration={m['duration']}, "
        f"saturation={m['saturation']}, contrast={m['contrast']}, warmth={m['warmth']}, "
        f"shot_count={m['shot_count']}, cuts_in_first_3s={m['cuts_in_first_3s']}\n"
        "Name the visual style. Return JSON with keys: grade, cut_style, moves (list), "
        "voice_tone, keywords (list), notes."
    )
    named = llm.complete_json(
        task="style_synthesis",
        system=(
            "You are a short-form video editor. Describe an existing style precisely and "
            "reusably. Do not invent numbers -- only name qualitative attributes."
        ),
        user=user,
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
            cut_times=m["cuts"][:40],
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
            bpm=_estimate_bpm(m["median_cut_interval"]),
            voice_tone=named.get("voice_tone", "neutral"),
        ),
        hook_style=HookStyle(cut_count=m["cuts_in_first_3s"]),
        notes=(notes + " " + named.get("notes", "")).strip(),
    )
