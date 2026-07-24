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
import os
import subprocess
import warnings
from pathlib import Path

import httpx
import yaml

from ..config import settings


class VideoProviderError(RuntimeError):
    pass


class BaseVideoProvider:
    name = "base"

    @property
    def min_clip_sec(self) -> float:
        """Shortest clip this provider can produce. Real i2v endpoints have a
        floor (3-4s) far above a typical short-form shot, so callers request the
        floor and trim down."""
        return 0.0

    @property
    def max_concurrency(self) -> int:
        """Upper bound on parallel renders this provider tolerates."""
        return 32

    def keyframe(self, prompt: str, out_path: Path, ref_image: Path | None = None) -> Path:
        raise NotImplementedError

    def animate(
        self,
        image_path: Path,
        motion_prompt: str,
        duration: float,
        out_path: Path,
        persona_ref: Path | None = None,
    ) -> Path:
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

    def animate(
        self,
        image_path: Path,
        motion_prompt: str,
        duration: float,
        out_path: Path,
        persona_ref: Path | None = None,
    ) -> Path:
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


def load_fal_specs() -> dict:
    path = settings.fal_models_path
    if not path.exists():
        path = Path(__file__).resolve().parents[3] / "configs" / "fal_models.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


class FalVideoProvider(BaseVideoProvider):
    """fal.ai, driven by the endpoint specs in configs/fal_models.yaml.

    Uses the official `fal_client` SDK rather than hand-rolled HTTP: it owns the
    queue protocol and, more importantly, CDN upload. fal's own docs discourage
    base64 data URIs above a few KB, and keyframes are far larger than that.

    Endpoint schemas genuinely differ (`image_url` vs `start_image_url`,
    different size parameters, different duration floors), so payloads are built
    from the spec instead of a single fixed shape.
    """

    name = "fal"

    def __init__(self) -> None:
        try:
            import fal_client
        except ImportError as exc:  # optional dependency
            raise VideoProviderError(
                "video_provider=fal requires `pip install fal-client`"
            ) from exc
        if not settings.fal_key:
            raise VideoProviderError("STYLELOOM_FAL_KEY is required for video_provider=fal")
        os.environ["FAL_KEY"] = settings.fal_key  # not setdefault: a stale env var must not win

        specs = load_fal_specs()
        self._client = fal_client
        self.t2i_id = settings.fal_t2i_model
        self.i2v_id = settings.fal_i2v_model
        self.t2i = self._spec(specs, "text_to_image", self.t2i_id)
        self.i2v = self._spec(specs, "image_to_video", self.i2v_id)

    @staticmethod
    def _spec(specs: dict, kind: str, model_id: str) -> dict:
        known = specs.get(kind, {})
        if model_id not in known:
            raise VideoProviderError(
                f"unknown {kind} endpoint {model_id!r}. "
                f"Known: {sorted(known)}. Add it to configs/fal_models.yaml."
            )
        return known[model_id]

    @property
    def min_clip_sec(self) -> float:
        """Shortest clip the configured endpoint will produce. Shots below this
        are requested at the floor and trimmed by the caller."""
        return float(self.i2v.get("min_duration", 0))

    @property
    def max_concurrency(self) -> int:
        """fal enforces a per-user concurrency limit per endpoint -- Kling v3
        defaults to 1. Exceeding it fails the run, so the ceiling comes from the
        endpoint, not from our settings."""
        return int(self.i2v.get("max_concurrency", 0)) or super().max_concurrency

    def _run(self, model_id: str, payload: dict) -> dict:
        try:
            return self._client.subscribe(
                model_id, arguments=payload, client_timeout=settings.fal_timeout_sec
            )
        except Exception as exc:
            raise VideoProviderError(f"fal {model_id} failed: {exc}") from exc

    @staticmethod
    def _media_url(result: dict, key: str) -> str:
        node = result.get(key)
        if isinstance(node, list):
            node = node[0] if node else None
        if isinstance(node, dict) and node.get("url"):
            return node["url"]
        raise VideoProviderError(f"no {key} URL in fal result: {str(result)[:300]}")

    @staticmethod
    def _download(url: str, out_path: Path) -> Path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with httpx.stream("GET", url, timeout=300.0, follow_redirects=True) as r:
            r.raise_for_status()
            with out_path.open("wb") as fh:
                for chunk in r.iter_bytes():
                    fh.write(chunk)
        return out_path

    def keyframe(self, prompt: str, out_path: Path, ref_image: Path | None = None) -> Path:
        payload: dict = {"prompt": prompt, **self.t2i.get("defaults", {})}
        if self.t2i.get("size_mode") == "image_size_object":
            payload["image_size"] = {"width": settings.width, "height": settings.height}
        result = self._run(self.t2i_id, payload)
        return self._download(self._media_url(result, self.t2i["output_key"]), out_path)

    def animate(
        self,
        image_path: Path,
        motion_prompt: str,
        duration: float,
        out_path: Path,
        persona_ref: Path | None = None,
    ) -> Path:
        spec = self.i2v
        seconds = int(round(max(duration, spec.get("min_duration", 1))))
        seconds = min(seconds, int(spec.get("max_duration", seconds)))

        payload: dict = {"prompt": motion_prompt, **spec.get("defaults", {})}
        payload[spec["image_param"]] = self._client.upload_file(image_path)
        payload[spec["duration_param"]] = str(seconds)  # string everywhere on fal

        if spec.get("size_mode") == "resolution_enum":
            payload["resolution"] = spec.get("resolution", "720p")
        if spec.get("aspect_param"):
            payload[spec["aspect_param"]] = "9:16" if settings.height > settings.width else "16:9"

        if persona_ref is not None:
            if spec.get("persona_mode") == "kling_elements":
                payload["elements"] = [
                    {"frontal_image_url": self._client.upload_file(persona_ref)}
                ]
                payload["prompt"] = f"@Element1 {motion_prompt}"
            else:
                warnings.warn(
                    f"{self.i2v_id} has no reference-image parameter; persona ignored. "
                    "Use a Kling v3 endpoint for creator consistency.",
                    stacklevel=2,
                )

        result = self._run(self.i2v_id, payload)
        return self._download(self._media_url(result, spec["output_key"]), out_path)


def get_video_provider() -> BaseVideoProvider:
    if settings.video_provider == "fal":
        return FalVideoProvider()
    return MockVideoProvider()
