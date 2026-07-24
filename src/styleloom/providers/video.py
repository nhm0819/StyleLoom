"""Video provider: two-stage `text -> keyframe -> motion clip`.

Why two stages instead of straight text-to-video: shot-to-shot consistency of
person and colour grade is what makes a short-form set read as "one channel".
Locking the look at the image stage leaves the video model responsible only for
motion. See docs/TOOL_RATIONALE.md.

  * `mock` - ffmpeg only. Produces a real keyframe JPEG and a real MP4 with a
             slow push-in, so the whole harness is runnable with zero keys.
  * `fal`  - fal.ai queue API (submit -> poll -> fetch). Model IDs are NOT
             hardcoded; set STYLELOOM_FAL_T2I_MODEL / STYLELOOM_FAL_I2V_MODEL.
"""

from __future__ import annotations

import hashlib
import subprocess
import time
from pathlib import Path

import httpx

from ..config import settings


class VideoProviderError(RuntimeError):
    pass


class BaseVideoProvider:
    name = "base"

    def keyframe(self, prompt: str, out_path: Path, ref_image: Path | None = None) -> Path:
        raise NotImplementedError

    def animate(self, image_path: Path, motion_prompt: str, duration: float, out_path: Path) -> Path:
        raise NotImplementedError


# --------------------------------------------------------------------------- #


def _run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise VideoProviderError(f"ffmpeg failed: {proc.stderr[-500:]}")


class MockVideoProvider(BaseVideoProvider):
    name = "mock"

    def keyframe(self, prompt: str, out_path: Path, ref_image: Path | None = None) -> Path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # Prompt-derived colour keeps distinct shots visually distinct.
        h = hashlib.sha1(prompt.encode()).hexdigest()
        c0, c1 = f"0x{h[0:6]}", f"0x{h[6:12]}"
        w, hgt = settings.width, settings.height
        _run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-f", "lavfi",
                "-i", f"gradients=s={w}x{hgt}:c0={c0}:c1={c1}:duration=1:speed=0.1",
                "-frames:v", "1", str(out_path),
            ]
        )
        return out_path

    def animate(self, image_path: Path, motion_prompt: str, duration: float, out_path: Path) -> Path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fps, w, h = settings.fps, settings.width, settings.height
        frames = max(int(duration * fps), 1)
        zoom = "zoompan=z='min(zoom+0.0012,1.18)':d=%d:s=%dx%d:fps=%d" % (frames, w, h, fps)
        _run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-loop", "1", "-i", str(image_path),
                "-vf", f"{zoom},format=yuv420p",
                "-t", f"{duration:.2f}",
                "-r", str(fps),
                "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
                str(out_path),
            ]
        )
        return out_path


# --------------------------------------------------------------------------- #


class FalVideoProvider(BaseVideoProvider):
    """fal.ai hosts Seedance / Kling / Veo behind a uniform queue API.

    NOTE: model IDs change between releases -- they are configuration, not code.
    Verify the current ID on fal.ai before running in production.
    """

    name = "fal"
    QUEUE = "https://queue.fal.run"

    def __init__(self) -> None:
        if not settings.fal_key:
            raise VideoProviderError("STYLELOOM_FAL_KEY is required for video_provider=fal")
        if not (settings.fal_t2i_model and settings.fal_i2v_model):
            raise VideoProviderError(
                "set STYLELOOM_FAL_T2I_MODEL and STYLELOOM_FAL_I2V_MODEL"
            )
        self.headers = {"Authorization": f"Key {settings.fal_key}"}

    def _submit_and_wait(self, model: str, payload: dict, timeout: float = 600.0) -> dict:
        with httpx.Client(timeout=60.0) as client:
            r = client.post(f"{self.QUEUE}/{model}", headers=self.headers, json=payload)
            if r.status_code >= 400:
                raise VideoProviderError(f"fal submit {r.status_code}: {r.text[:300]}")
            job = r.json()
            status_url = job.get("status_url")
            response_url = job.get("response_url")
            if not status_url or not response_url:
                raise VideoProviderError(f"fal returned no queue URLs: {job}")

            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                s = client.get(status_url, headers=self.headers).json()
                status = s.get("status")
                if status == "COMPLETED":
                    return client.get(response_url, headers=self.headers).json()
                if status in {"FAILED", "CANCELLED"}:
                    raise VideoProviderError(f"fal job {status}: {s}")
                time.sleep(3.0)
        raise VideoProviderError(f"fal job timed out after {timeout}s")

    @staticmethod
    def _download(url: str, out_path: Path) -> Path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with httpx.stream("GET", url, timeout=300.0, follow_redirects=True) as r:
            r.raise_for_status()
            with out_path.open("wb") as fh:
                for chunk in r.iter_bytes():
                    fh.write(chunk)
        return out_path

    @staticmethod
    def _first_url(result: dict, *keys: str) -> str:
        for key in keys:
            node = result.get(key)
            if isinstance(node, dict) and node.get("url"):
                return node["url"]
            if isinstance(node, list) and node and isinstance(node[0], dict) and node[0].get("url"):
                return node[0]["url"]
        raise VideoProviderError(f"no media URL in fal result: {str(result)[:300]}")

    def keyframe(self, prompt: str, out_path: Path, ref_image: Path | None = None) -> Path:
        payload = {
            "prompt": prompt,
            "image_size": {"width": settings.width, "height": settings.height},
        }
        result = self._submit_and_wait(settings.fal_t2i_model, payload)
        return self._download(self._first_url(result, "images", "image"), out_path)

    def animate(self, image_path: Path, motion_prompt: str, duration: float, out_path: Path) -> Path:
        # fal image inputs accept data URIs; avoids needing a separate upload step.
        import base64

        b64 = base64.b64encode(image_path.read_bytes()).decode()
        payload = {
            "prompt": motion_prompt,
            "image_url": f"data:image/jpeg;base64,{b64}",
            "duration": str(int(round(duration))),
        }
        result = self._submit_and_wait(settings.fal_i2v_model, payload)
        return self._download(self._first_url(result, "video", "videos"), out_path)


def get_video_provider() -> BaseVideoProvider:
    if settings.video_provider == "fal":
        return FalVideoProvider()
    return MockVideoProvider()
