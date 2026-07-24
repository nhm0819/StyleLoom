"""Test fixtures.

`STYLELOOM_DATA_DIR` is set before importing the package because `settings` is a
module-level singleton -- importing first would bind the real data directory.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

import pytest

_TMP = Path(tempfile.mkdtemp(prefix="styleloom_test_"))
os.environ["STYLELOOM_DATA_DIR"] = str(_TMP)
os.environ["STYLELOOM_LLM_PROVIDER"] = "mock"
os.environ["STYLELOOM_VIDEO_PROVIDER"] = "mock"
os.environ["STYLELOOM_WIDTH"] = "360"
os.environ["STYLELOOM_HEIGHT"] = "640"
os.environ["STYLELOOM_FPS"] = "24"


@pytest.fixture(scope="session")
def client():
    from fastapi.testclient import TestClient

    from styleloom.api.main import app

    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="session")
def reference_video() -> Path:
    """Six hard cuts at ~1.2s -- a plausible short-form pacing signature."""
    out = _TMP / "reference.mp4"
    if out.exists():
        return out

    segments = []
    colours = ["red", "blue", "green", "orange", "purple", "teal"]
    for i, colour in enumerate(colours):
        seg = _TMP / f"seg{i}.mp4"
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-f", "lavfi", "-i", f"color=c={colour}:s=360x640:d=1.2:r=24",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", str(seg),
            ],
            check=True,
        )
        segments.append(seg)

    listing = _TMP / "segments.txt"
    listing.write_text("\n".join(f"file '{s}'" for s in segments), encoding="utf-8")
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "concat", "-safe", "0", "-i", str(listing),
            "-c", "copy", str(out),
        ],
        check=True,
    )
    return out
