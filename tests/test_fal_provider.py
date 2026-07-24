"""fal provider tests.

No network. `fal_client` is replaced with a stub that records the payload, which
is the only thing worth asserting here: whether StyleLoom builds the shape each
endpoint actually documents. Everything downstream of that is fal's problem.
"""

from __future__ import annotations

import subprocess
import sys
import types
from pathlib import Path

import pytest

from styleloom.config import settings
from styleloom.pipeline import render
from styleloom.providers.video import (
    BaseVideoProvider,
    FalVideoProvider,
    VideoProviderError,
)
from styleloom.schema import Shot

SEEDANCE = "bytedance/seedance-2.0/fast/image-to-video"
KLING = "fal-ai/kling-video/v3/pro/image-to-video"


class StubFalClient(types.SimpleNamespace):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[str, dict]] = []
        self.uploads: list[Path] = []

    def upload_file(self, path, **_):
        self.uploads.append(Path(path))
        return f"https://v3.fal.media/files/{Path(path).name}"

    def subscribe(self, application, arguments, **_):
        self.calls.append((application, arguments))
        if "image-to-video" in application:
            return {"video": {"url": "https://v3.fal.media/files/out.mp4"}}
        return {"images": [{"url": "https://v3.fal.media/files/out.jpg"}]}


@pytest.fixture
def fal(monkeypatch):
    stub = StubFalClient()
    monkeypatch.setitem(sys.modules, "fal_client", stub)
    monkeypatch.setattr(settings, "fal_key", "test-key")

    def build(i2v: str = SEEDANCE, t2i: str = "fal-ai/flux-2-flex") -> FalVideoProvider:
        monkeypatch.setattr(settings, "fal_i2v_model", i2v)
        monkeypatch.setattr(settings, "fal_t2i_model", t2i)
        provider = FalVideoProvider()
        monkeypatch.setattr(provider, "_download", lambda url, out: out)
        return provider

    return stub, build


def _shot(duration: float = 1.2) -> Shot:
    return Shot(
        index=0,
        role="hook",
        duration_sec=duration,
        shot_size="CU",
        camera_move="snap_zoom_in",
        action="a",
        caption="c",
        image_prompt="a keyframe",
        motion_prompt="push in",
    )


def test_unknown_model_id_is_rejected(fal):
    _, build = fal
    with pytest.raises(VideoProviderError, match="unknown image_to_video"):
        build(i2v="fal-ai/not-a-real-model")


def test_seedance_payload_matches_documented_schema(fal, tmp_path):
    stub, build = fal
    provider = build(i2v=SEEDANCE)
    provider.animate(tmp_path / "kf.jpg", "push in", 4.0, tmp_path / "out.mp4")

    _, payload = stub.calls[-1]
    assert "image_url" in payload  # Seedance's name...
    assert "start_image_url" not in payload  # ...not Kling's
    assert payload["duration"] == "4"  # string, not int
    assert payload["resolution"] == "720p"
    assert payload["aspect_ratio"] == "9:16"
    assert payload["generate_audio"] is False


def test_kling_payload_uses_its_own_parameter_names(fal, tmp_path):
    stub, build = fal
    provider = build(i2v=KLING)
    provider.animate(tmp_path / "kf.jpg", "push in", 5.0, tmp_path / "out.mp4")

    _, payload = stub.calls[-1]
    assert "start_image_url" in payload
    assert "image_url" not in payload
    # Kling infers aspect ratio from the start image, so sending one is wrong.
    assert "aspect_ratio" not in payload
    assert "negative_prompt" in payload


def test_duration_floor_is_respected(fal, tmp_path):
    """A 1.2s shot cannot be requested from an endpoint with a 4s minimum."""
    stub, build = fal
    provider = build(i2v=SEEDANCE)
    assert provider.min_clip_sec == 4.0

    provider.animate(tmp_path / "kf.jpg", "push in", 1.2, tmp_path / "out.mp4")
    assert stub.calls[-1][1]["duration"] == "4"


def test_persona_uses_kling_elements(fal, tmp_path):
    stub, build = fal
    persona = tmp_path / "creator.jpg"
    persona.write_bytes(b"x")

    provider = build(i2v=KLING)
    provider.animate(tmp_path / "kf.jpg", "walks in", 5.0, tmp_path / "out.mp4", persona)

    _, payload = stub.calls[-1]
    assert payload["elements"][0]["frontal_image_url"].endswith("creator.jpg")
    assert payload["prompt"].startswith("@Element1")


def test_persona_warns_when_endpoint_cannot_use_it(fal, tmp_path):
    """Silently dropping the creator reference would be worse than saying so."""
    _, build = fal
    persona = tmp_path / "creator.jpg"
    persona.write_bytes(b"x")

    provider = build(i2v=SEEDANCE)
    with pytest.warns(UserWarning, match="persona ignored"):
        provider.animate(tmp_path / "kf.jpg", "walks in", 4.0, tmp_path / "out.mp4", persona)


# --------------------------------------------------------------------------- #


class FloorBoundProvider(BaseVideoProvider):
    """Mimics a real endpoint: ignores sub-minimum requests and returns 5s."""

    @property
    def min_clip_sec(self) -> float:
        return 5.0

    def keyframe(self, prompt, out_path, ref_image=None):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
             "-i", "color=c=red:s=180x320:d=1", "-frames:v", "1", str(out_path)],
            check=True,
        )
        return out_path

    def animate(self, image_path, motion_prompt, duration, out_path, persona_ref=None):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
             "-i", "color=c=blue:s=180x320:d=5:r=24", "-t", "5",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out_path)],
            check=True,
        )
        return out_path


def _duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    )
    return float(out.stdout.strip())


def test_oversized_clip_is_trimmed_to_the_shot(tmp_path):
    """The floor must not leak into the cut. A 1.2s shot stays 1.2s even when
    the provider can only make 5s clips -- otherwise pacing, the thing the style
    schema encodes, is destroyed."""
    clip = render.render_shot(FloorBoundProvider(), _shot(1.2), tmp_path)
    assert _duration(clip) == pytest.approx(1.2, abs=0.15)
    assert (tmp_path / "raw" / "shot_00.mp4").exists()  # untrimmed original kept


def test_kling_concurrency_ceiling_is_enforced(fal):
    """fal caps Kling v3 at 1 concurrent request per user by default. Letting
    settings.max_concurrent_renders override that would fail the whole run."""
    _, build = fal
    assert build(i2v=KLING).max_concurrency == 1
    assert build(i2v=SEEDANCE).max_concurrency == 32  # no documented cap
