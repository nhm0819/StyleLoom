"""Filtergraph path escaping.

A Windows font path is the reason this exists: `C:/Windows/Fonts/malgunbd.ttf`
inside `drawtext` parses as `fontfile=C` plus a bare `/Windows/...`, because ffmpeg
reads `:` as an option separator.

The escaping is checked against a real ffmpeg rather than only as a string, since
the number of backslashes is the whole question and only ffmpeg settles it. A colon
is legal in a POSIX filename, so a directory named like a drive letter reproduces
the Windows condition on the machines that run this suite.
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import pytest
from styleloom_core.media import (
    FONT_CANDIDATES,
    CaptionCue,
    burn_captions,
    caption_fontsize_expr,
    filter_path,
    resolve_font,
    run_ffmpeg,
)
from styleloom_core.schema import CaptionStyle


def test_posix_paths_are_untouched():
    """The no-op guarantee: escaping must not change what already worked."""
    p = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
    assert filter_path(p) == p


def test_a_windows_path_gets_its_drive_colon_escaped():
    assert (
        filter_path(r"C:\Windows\Fonts\malgunbd.ttf")
        == "C\\\\:/Windows/Fonts/malgunbd.ttf"
    )


def test_bold_is_preferred_over_regular():
    """Weight ordering is the point of the candidate list, so no Regular face may
    precede a Bold one -- otherwise a machine carrying both silently gets the
    thinner face. `malgunbd.ttf` is Bold; `malgun.ttf` is its Regular sibling.

    Scoped to the CJK-capable entries. The Latin-only DejaVu fallback sits last
    despite being Bold, because there coverage outranks weight: it is the choice
    between a thin caption and a caption of empty boxes.
    """
    cjk = [c for c in FONT_CANDIDATES if "CJK" in c or "malgun" in c]
    assert len(cjk) == len(FONT_CANDIDATES) - 1, "expected exactly one non-CJK fallback"

    weights = [
        "bold" if ("bold" in c.lower() or c.endswith("malgunbd.ttf")) else "regular"
        for c in cjk
    ]
    assert weights == sorted(weights, key=lambda w: 0 if w == "bold" else 1), (
        f"a Regular candidate precedes a Bold one: {cjk}"
    )
    assert FONT_CANDIDATES[-1].endswith("DejaVuSans-Bold.ttf")


@pytest.mark.skipif(
    os.name == "nt", reason="a colon cannot appear in a Windows filename"
)
def test_captions_burn_when_the_path_contains_a_colon(tmp_path: Path):
    """End to end: unescaped, ffmpeg rejects the filtergraph outright."""
    if resolve_font() is None:
        pytest.skip("no CJK-capable font installed, so burn_captions returns early")

    drive_like = tmp_path / "C:fakedrive"
    drive_like.mkdir()
    clip = drive_like / "in.mp4"
    run_ffmpeg(
        [
            "-f", "lavfi",
            "-i", "color=c=navy:s=240x426:d=1:r=24",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", str(clip),
        ]
    )

    out = burn_captions(
        clip,
        [CaptionCue(text="자막 테스트", start=0.0, end=1.0)],
        CaptionStyle(),
        drive_like / "out.mp4",
    )
    assert out != clip, "burn_captions returned the input, so nothing was drawn"
    assert out.exists() and out.stat().st_size > 0


# --- caption fit ------------------------------------------------------------- #


def _solid_clip(path: Path, width: int, height: int, seconds: float = 2.0) -> Path:
    run_ffmpeg([
        "-f", "lavfi", "-i", f"color=c=black:s={width}x{height}:d={seconds}:r=12",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path),
    ])
    return path


def _drawn_extent(clip: Path, frame_png: Path) -> tuple[int, int, int, int]:
    """Bounding box of everything drawn onto an otherwise black clip."""
    run_ffmpeg(["-i", str(clip), "-vf", "select=eq(n\\,10)", "-vframes", "1", str(frame_png)])
    image = cv2.imread(str(frame_png), cv2.IMREAD_GRAYSCALE)
    cols = np.where(image.max(axis=0) > 40)[0]
    rows = np.where(image.max(axis=1) > 40)[0]
    return int(cols.min()), int(cols.max()), int(rows.min()), int(rows.max())


@pytest.mark.parametrize(
    ("width", "height", "chars"),
    [
        (720, 1280, 12),   # 9:16, the shape every bundled style renders at
        (720, 1280, 14),   # 9:16 at the schema's default chars-per-line
        (1080, 1920, 12),
        (1920, 1080, 12),  # 16:9, where the height ceiling binds instead
        (1080, 1080, 14),
    ],
)
def test_a_full_caption_line_stays_inside_the_frame(tmp_path, width, height, chars):
    """Measured in pixels, not asserted about the filter string.

    The font size used to come from `h/18`, which on a vertical frame is the long
    edge: at 720x1280 that is a 71px font, and twelve CJK glyphs is ~852px of text
    in a 720px frame. drawtext centres with `x=(w-text_w)/2`, so it went negative
    and the caption was clipped on both sides rather than shrunk -- and nothing
    failed, because ffmpeg is perfectly happy to draw outside the frame.

    A full line of CJK is the worst case on purpose: those glyphs are em squares,
    so `chars * fontsize` is the widest a line of that many characters can be.
    """
    if resolve_font() is None:
        pytest.skip("no CJK-capable font installed; burn_captions returns the clip")

    clip = _solid_clip(tmp_path / "base.mp4", width, height)
    out = burn_captions(
        clip,
        [CaptionCue(text="가" * chars, start=0.0, end=2.0)],
        CaptionStyle(max_chars_per_line=chars),
        tmp_path / "out.mp4",
    )
    left, right, top, bottom = _drawn_extent(out, tmp_path / "frame.png")

    assert left > 0, f"caption clipped at the left edge ({left})"
    assert right < width - 1, f"caption clipped at the right edge ({right} of {width})"
    assert bottom < height - 1, f"caption clipped at the bottom ({bottom} of {height})"
    # And it is not so small that the budget has collapsed to nothing.
    assert (right - left) > width * 0.25


def test_the_font_size_is_derived_from_width_not_height():
    """Width is the dimension the text has to fit, and `max_chars_per_line` is the
    setting that decides how much of it one line may use -- so the two have to be
    solved together rather than set independently."""
    narrow = caption_fontsize_expr(24)
    wide = caption_fontsize_expr(8)
    assert "w*" in narrow and "/24" in narrow
    # More characters per line must mean a smaller font, not the same one.
    assert narrow != wide
