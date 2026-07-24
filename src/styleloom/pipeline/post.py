"""Stage 7 - Captions and assembly.

Captions are burned here, not asked of the video model: models render text
unreliably and inconsistently, and the caption spec in `style.json` (font,
position, wrap width) is exactly the kind of thing that must be identical across
every video for the set to look like one channel.

Caption text is passed via `textfile=` so no drawtext escaping is needed for
Korean punctuation, colons or quotes.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from ..schema import CaptionStyle, Shot

FONT_CANDIDATES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]

Y_BY_POS = {
    "top": "h*0.12",
    "center": "(h-text_h)/2",
    "center_lower": "h*0.62",
    "bottom": "h*0.82",
}


class PostError(RuntimeError):
    pass


def _run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise PostError(f"ffmpeg failed: {proc.stderr[-600:]}")


def resolve_font() -> str | None:
    for path in FONT_CANDIDATES:
        if Path(path).exists():
            return path
    return None


def wrap(text: str, width: int) -> str:
    """Character-count wrap. Korean has no spaces at predictable places, so a
    word wrapper leaves ragged lines; short-form captions wrap on width."""
    words = text.split()
    if not words:
        return ""
    lines, current = [], ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) <= width or not current:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return "\n".join(lines[:3])


def burn_caption(clip: Path, shot: Shot, style: CaptionStyle, out_path: Path) -> Path:
    if not shot.caption.strip():
        return clip
    font = resolve_font()
    if font is None:
        return clip  # no CJK-capable font available; ship uncaptioned rather than fail

    text_path = out_path.with_suffix(".txt")
    text_path.write_text(wrap(shot.caption, style.max_chars_per_line), encoding="utf-8")

    fade_in = 0.12 if style.anim == "pop_in" else 0.0
    alpha = f"if(lt(t,{fade_in}),t/{fade_in},1)" if fade_in else "1"

    drawtext = (
        f"drawtext=fontfile={font}"
        f":textfile={text_path}"
        f":fontcolor={style.color}"
        f":fontsize=h/18"
        f":borderw=6:bordercolor={style.stroke_color}"
        f":line_spacing=10"
        f":x=(w-text_w)/2"
        f":y={Y_BY_POS.get(style.pos, 'h*0.62')}"
        f":alpha='{alpha}'"
    )
    _run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(clip),
            "-vf", drawtext,
            "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
            str(out_path),
        ]
    )
    text_path.unlink(missing_ok=True)
    return out_path


def trim_to(clip: Path, seconds: float, out_path: Path) -> Path:
    """Cut a clip down to an exact length, re-encoding rather than stream-copying
    so the cut lands on the requested frame instead of the nearest keyframe."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(clip),
            "-t", f"{seconds:.3f}",
            "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
            "-an",
            str(out_path),
        ]
    )
    return out_path


def concat(clips: list[Path], out_path: Path, bgm: Path | None = None) -> Path:
    if not clips:
        raise PostError("nothing to concatenate: all shots failed to render")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    listing = out_path.parent / "concat.txt"
    listing.write_text(
        "\n".join(f"file '{c.resolve()}'" for c in clips) + "\n", encoding="utf-8"
    )

    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(listing)]
    if bgm and bgm.exists():
        cmd += ["-i", str(bgm), "-shortest", "-c:a", "aac", "-b:a", "128k"]
    cmd += ["-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p", str(out_path)]
    _run(cmd)
    return out_path


def finish(
    clips: list[Path],
    shots: list[Shot],
    caption_style: CaptionStyle,
    work_dir: Path,
    out_path: Path,
    bgm: Path | None = None,
) -> Path:
    captioned_dir = work_dir / "captioned"
    captioned_dir.mkdir(parents=True, exist_ok=True)
    by_index = {s.index: s for s in shots}

    captioned = []
    for clip in clips:
        idx = int(clip.stem.split("_")[-1])
        shot = by_index.get(idx)
        if shot is None:
            captioned.append(clip)
            continue
        captioned.append(
            burn_caption(clip, shot, caption_style, captioned_dir / f"shot_{idx:02d}.mp4")
        )
    return concat(captioned, out_path, bgm)
