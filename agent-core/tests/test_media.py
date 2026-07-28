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

import pytest
from styleloom_core.media import (
    FONT_CANDIDATES,
    CaptionCue,
    burn_captions,
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
