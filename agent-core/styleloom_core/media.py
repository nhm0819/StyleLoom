"""Media primitives: ffmpeg calls and OpenCV measurement.

Why this module exists as a peer of `tools/` rather than living inside them:
in the previous layout the render stage imported the post stage (for `trim_to`)
and the QC stage imported the analyze stage (for `probe_video`). Tools calling
tools makes the execution order implicit in the import graph, which is exactly
what a declared plan is supposed to make explicit. So the shared operations moved
down here and tools depend only on this.

Everything here is a pure function of its arguments plus `Settings`. No session,
no artifacts, no provider.
"""

from __future__ import annotations

import statistics
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .config import Settings
from .errors import MediaError
from .schema import CaptionStyle

# --------------------------------------------------------------------------- #
# ffmpeg
# --------------------------------------------------------------------------- #

FONT_CANDIDATES = [
    # Weight before platform. A short-form caption is read at a glance over moving
    # footage, so Bold is the target and Regular is a fallback rather than an equal
    # alternative -- every Bold candidate is tried before any Regular one, so a
    # machine with both installed always lands on Bold.
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",  # Debian, Ubuntu
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Bold.ttc",
    "C:/Windows/Fonts/malgunbd.ttf",  # Windows, 맑은 고딕 Bold
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "C:/Windows/Fonts/malgun.ttf",  # Windows, 맑은 고딕
    # Latin-only last resort. Hangul renders as boxes here, but a non-Korean
    # caption still burns instead of being dropped silently.
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]

Y_BY_POS = {
    "top": "h*0.12",
    "center": "(h-text_h)/2",
    "center_lower": "h*0.62",
    "bottom": "h*0.82",
}

_ENCODE = ["-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p"]


def run_ffmpeg(args: list[str]) -> None:
    """Run ffmpeg, raising with the tail of stderr on failure.

    Only the tail: ffmpeg's banner is long and the useful line is always last.
    """
    proc = subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", *args], capture_output=True, text=True
    )
    if proc.returncode != 0:
        raise MediaError(f"ffmpeg failed: {proc.stderr[-600:].strip()}")


def resolve_font() -> str | None:
    for path in FONT_CANDIDATES:
        if Path(path).exists():
            return path
    return None


def filter_path(path: str | Path) -> str:
    """A filesystem path, escaped for use as a filtergraph option value.

    Needed the moment a Windows font path enters `drawtext`. ffmpeg reads `:` as an
    option separator, so `fontfile=C:/Windows/Fonts/malgunbd.ttf` parses as
    `fontfile=C` followed by a bare `/Windows/Fonts/malgunbd.ttf`, which drawtext
    takes as its shorthand `text` option and then rejects the whole run with
    "Both text and text file provided".

    Two backslashes, not one: escaping is applied twice, once by the filtergraph
    parser and once by the filter's own argument parser. A single backslash is
    consumed by the first and the colon reaches the second bare, failing
    identically to no escaping at all. Both forms were run against ffmpeg before
    this was written.

    A no-op on POSIX paths, which contain neither a colon nor a backslash.
    """
    return str(path).replace("\\", "/").replace(":", "\\\\:")


def wrap_caption(text: str, width: int, max_lines: int = 3) -> str:
    """Character-count wrap.

    Korean does not put spaces at predictable places, so a word wrapper leaves
    ragged lines. Short-form captions wrap on visual width instead.
    """
    words = text.split()
    if not words:
        return ""
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) <= width or not current:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return "\n".join(lines[:max_lines])


@dataclass(frozen=True)
class CaptionCue:
    """One caption and the window it occupies within a clip."""

    text: str
    start: float
    end: float


def burn_captions(
    clip: Path, cues: list[CaptionCue], style: CaptionStyle, out_path: Path
) -> Path:
    """Burn captions into `clip`, each shown only during its own window.

    One function serves both render modes. A single-shot clip is the degenerate
    case: one cue spanning the whole file. A multi-shot clip gets one drawtext
    filter per cue, gated by `enable=between(t, start, end)` -- which is why the
    cut moving from a file boundary to a timestamp does not need a second code
    path here.

    Text goes through `textfile=` rather than inline, which sidesteps drawtext
    escaping entirely -- Korean captions routinely contain colons, quotes and
    commas that would otherwise each need escaping.

    Returns the input clip unchanged if there is nothing to draw, or if no
    CJK-capable font is installed: shipping uncaptioned beats failing the run.
    """
    usable = [c for c in cues if c.text.strip()]
    font = resolve_font()
    if not usable or font is None:
        return clip

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fade_in = 0.12 if style.anim == "pop_in" else 0.0

    filters: list[str] = []
    text_paths: list[Path] = []
    for i, cue in enumerate(usable):
        text_path = out_path.with_suffix(f".{i:02d}.txt")
        text_path.write_text(
            wrap_caption(cue.text, style.max_chars_per_line), encoding="utf-8"
        )
        text_paths.append(text_path)

        # Fade is relative to the cue's own start, not the clip's, or every caption
        # after the first would appear already fully opaque.
        if fade_in:
            local = f"(t-{cue.start:.3f})"
            alpha = f"if(lt({local},{fade_in}),{local}/{fade_in},1)"
        else:
            alpha = "1"

        filters.append(
            f"drawtext=fontfile={filter_path(font)}"
            f":textfile={filter_path(text_path)}"
            f":fontcolor={style.color}"
            f":fontsize=h/18"
            f":borderw=6:bordercolor={style.stroke_color}"
            f":line_spacing=10"
            f":x=(w-text_w)/2"
            f":y={Y_BY_POS.get(style.pos, 'h*0.62')}"
            f":alpha='{alpha}'"
            f":enable='between(t,{cue.start:.3f},{cue.end:.3f})'"
        )

    try:
        run_ffmpeg(["-i", str(clip), "-vf", ",".join(filters), *_ENCODE, str(out_path)])
    finally:
        for path in text_paths:
            path.unlink(missing_ok=True)
    return out_path


def trim_to(clip: Path, seconds: float, out_path: Path) -> Path:
    """Cut a clip to an exact length.

    Re-encodes rather than stream-copying so the cut lands on the requested
    frame instead of the nearest keyframe. Pacing is the property the style
    schema exists to reproduce, so frame-accuracy here is load-bearing.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    run_ffmpeg(["-i", str(clip), "-t", f"{seconds:.3f}", *_ENCODE, "-an", str(out_path)])
    return out_path


def concat(clips: list[Path], out_path: Path, bgm: Path | None = None) -> Path:
    if not clips:
        raise MediaError("nothing to concatenate: no clips survived rendering")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    listing = out_path.parent / "concat.txt"
    listing.write_text(
        "\n".join(f"file '{c.resolve()}'" for c in clips) + "\n", encoding="utf-8"
    )

    args = ["-f", "concat", "-safe", "0", "-i", str(listing)]
    if bgm and bgm.exists():
        args += ["-i", str(bgm), "-shortest", "-c:a", "aac", "-b:a", "128k"]
    args += [*_ENCODE, str(out_path)]
    run_ffmpeg(args)
    return out_path


def gradient_keyframe(prompt_hash: str, out_path: Path, settings: Settings) -> Path:
    """A deterministic placeholder still, coloured from a prompt hash so distinct
    shots stay visually distinct. Used by the offline mock provider."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    c0, c1 = f"0x{prompt_hash[0:6]}", f"0x{prompt_hash[6:12]}"
    run_ffmpeg(
        [
            "-f", "lavfi",
            "-i", (
                f"gradients=s={settings.width}x{settings.height}"
                f":c0={c0}:c1={c1}:duration=1:speed=0.1"
            ),
            "-frames:v", "1",
            str(out_path),
        ]
    )
    return out_path


def push_in(image_path: Path, duration: float, out_path: Path, settings: Settings) -> Path:
    """Animate a still with a slow push-in. Used by the offline mock provider."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames = max(int(duration * settings.fps), 1)
    zoom = (
        f"zoompan=z='min(zoom+0.0012,1.18)':d={frames}"
        f":s={settings.width}x{settings.height}:fps={settings.fps}"
    )
    run_ffmpeg(
        [
            "-loop", "1", "-i", str(image_path),
            "-vf", f"{zoom},format=yuv420p",
            "-t", f"{duration:.2f}",
            "-r", str(settings.fps),
            *_ENCODE,
            str(out_path),
        ]
    )
    return out_path


# --------------------------------------------------------------------------- #
# OpenCV measurement
# --------------------------------------------------------------------------- #

CUT_THRESHOLD = 0.38  # normalised HSV histogram Bhattacharyya distance
MIN_SHOT_SEC = 0.25
MAX_KEYFRAMES = 8


def _hist(frame: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
    return cv2.normalize(h, h).flatten()


def probe_video(path: Path, sample_stride: int = 2) -> dict:
    """One decode pass yielding cut times, colour statistics and keyframes.

    Measured rather than described by a model: an LLM answering "fast cuts" is
    not reproducible, `avg_shot_sec=1.34` is, and it can be checked against the
    output in QC. Cuts, colour stats and keyframes all come from the same pass,
    which is why a second cut-detection library was not worth adding.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise MediaError(f"cannot open video: {path}")

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
    try:
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
                        if len(keyframes) < MAX_KEYFRAMES:
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
    finally:
        cap.release()

    if duration == 0.0:
        duration = idx / fps

    shot_count = len(cuts) + 1
    avg_shot = duration / shot_count if shot_count else duration
    # Pairwise over consecutive cuts: the tail is one shorter by design.
    intervals = (
        [b - a for a, b in zip(cuts, cuts[1:], strict=False)]
        if len(cuts) > 1
        else []
    )
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


def video_duration_note(path: Path) -> str:
    """A one-line description of a video input, for the ingest prompt."""
    cap = cv2.VideoCapture(str(path))
    ok, _ = cap.read()
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    if not ok:
        return f"video input ({path.name}), unreadable"
    return f"video input ({path.name}), ~{total / fps:.1f}s"


def estimate_bpm(median_interval: float) -> float:
    """Assume cuts land on beats; fold into a musical 70-160 range.

    A heuristic, not a measurement. Audio onset detection would be correct but
    costs a heavy dependency for one number, and short-form edits cut on the
    beat often enough for this to be useful. Wrong on footage that does not.
    """
    if median_interval <= 0:
        return 120.0
    bpm = 60.0 / median_interval
    while bpm < 70:
        bpm *= 2
    while bpm > 160:
        bpm /= 2
    return round(bpm, 1)
