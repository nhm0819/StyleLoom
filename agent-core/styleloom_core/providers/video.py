"""Video providers: two-stage `text -> keyframe -> motion clip`.

Why two stages instead of straight text-to-video: shot-to-shot consistency of
person and colour grade is what makes a short-form set read as one channel.
Locking the look at the image stage leaves the video model responsible only for
motion. Straight text-to-video re-rolls the look on every shot, and by shot six
the grade has drifted. See docs/TOOL_RATIONALE.md.

  * `mock` - ffmpeg only. Produces a real keyframe JPEG and a real MP4 with a
             slow push-in, so the whole harness runs with zero keys.
  * `fal`  - fal.ai queue API. Model IDs are NOT hardcoded; they are settings,
             and their payload differences are data in configs/fal_models.yaml.
"""

from __future__ import annotations

import hashlib
import math
import os
import warnings
from pathlib import Path

import httpx
import yaml

from ..config import Settings
from ..errors import VideoProviderError
from ..media import concat, gradient_keyframe, push_in
from .base import BaseVideoProvider, MotionShot


class MockVideoProvider(BaseVideoProvider):
    name = "mock"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @property
    def supports_persona(self) -> bool:
        """False, and deliberately not faked.

        The offline renderer draws gradients; it cannot honour an identity
        reference. Reporting True would produce a portrait file that nothing
        consumes and imply consistency that is not there. The creator still varies
        offline through the prompt tokens, which is the mechanism under test.
        """
        return False

    def keyframe(self, prompt: str, out_path: Path, ref_image: Path | None = None) -> Path:
        digest = hashlib.sha1(prompt.encode()).hexdigest()
        return gradient_keyframe(digest, out_path, self.settings)

    @property
    def supports_multi_shot(self) -> bool:
        """True, and genuinely implemented rather than stubbed.

        The offline renderer can produce a single clip containing several cuts, so
        the whole multi_shot path -- segment handling, captions placed by
        timestamp, the QC drift check -- runs and is testable without a key.

        What it cannot tell you is whether a *real* endpoint honours a requested
        cut timeline: ffmpeg cuts exactly where told, so drift here is always
        zero. This exercises the plumbing, not the model's obedience.
        """
        return True

    @property
    def max_shot_window_sec(self) -> float:
        # No real limit offline; matches the common endpoint ceiling so offline
        # runs split into the same number of windows a real one would.
        return 15.0

    def animate(
        self,
        image_path: Path,
        motion_prompt: str,
        duration: float,
        out_path: Path,
        persona_ref: Path | None = None,
    ) -> Path:
        return push_in(image_path, duration, out_path, self.settings)

    def animate_sequence(
        self,
        image_path: Path,
        shots: list[MotionShot],
        out_path: Path,
        persona_ref: Path | None = None,
    ) -> Path:
        """Build one clip whose shots are visually distinct.

        Only the first shot animates the supplied start image; the rest derive
        their own frame from their own prompt. That mirrors how a real multi-shot
        endpoint behaves -- the start image anchors the opening shot and the model
        generates the others -- and it matters for more than realism: if every
        shot looked the same, the output would contain no detectable cuts and the
        QC drift check would report a catastrophic miss on a correct timeline.
        """
        parts = []
        scratch = out_path.parent / f"{out_path.stem}_parts"
        for i, shot in enumerate(shots):
            frame = (
                image_path
                if i == 0
                else self.keyframe(shot.prompt, scratch / f"{i:02d}.jpg")
            )
            parts.append(
                push_in(frame, shot.duration, scratch / f"{i:02d}.mp4", self.settings)
            )
        return concat(parts, out_path)


# --------------------------------------------------------------------------- #


def load_fal_specs(settings: Settings) -> dict:
    path = settings.resolve_config(settings.fal_models_path)
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


class FalVideoProvider(BaseVideoProvider):
    """fal.ai, driven by the endpoint specs in configs/fal_models.yaml.

    fal fronts several video models behind one queue API, so one implementation
    covers Seedance, Kling and others. It does *not* front them behind one
    schema, which was the surprise. Verified against the official model pages:

        start image  Seedance `image_url`      Kling `start_image_url`
        duration     string on both, floors 4s / 3s
        aspect       Seedance param; Kling infers it from the start image
        persona      Kling `elements` only
        concurrency  Kling defaults to 1 per user

    A single hardcoded payload would send a wrong parameter name to every model
    except the one it was written against, and would do it silently. Hence the
    spec file: adding a model is a data change.

    The official `fal_client` SDK is used rather than hand-rolled HTTP because it
    owns the queue protocol and CDN upload -- fal's docs discourage base64 data
    URIs above a few KB, and keyframes are hundreds of KB.
    """

    name = "fal"

    def __init__(self, settings: Settings) -> None:
        try:
            import fal_client
        except ImportError as exc:  # optional dependency
            raise VideoProviderError(
                "video_provider=fal requires `pip install fal-client`"
            ) from exc
        if not settings.fal_key:
            raise VideoProviderError(
                "STYLELOOM_FAL_KEY is required for video_provider=fal"
            )
        # Assignment, not setdefault: a stale env var must not win over settings.
        os.environ["FAL_KEY"] = settings.fal_key

        self.settings = settings
        self._client = fal_client
        specs = load_fal_specs(settings)
        self.t2i_id = settings.fal_t2i_model
        self.i2v_id = settings.fal_i2v_model
        # Validated here rather than at the first request, so a typo fails the
        # run before any money is spent.
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
        return float(self.i2v.get("min_duration", 0))

    @property
    def supports_persona(self) -> bool:
        """Endpoint-dependent: only Kling v3 exposes a reference-image parameter."""
        return self.i2v.get("persona_mode", "none") != "none"

    @property
    def supports_multi_shot(self) -> bool:
        return "multi_prompt" in (self.i2v.get("capabilities") or [])

    @property
    def max_shot_window_sec(self) -> float:
        return float(self.i2v.get("max_shot_window_sec", 0))

    @property
    def max_shots_per_request(self) -> int:
        """Kling accepts 1-6 shots per multi_prompt request. Exceeding it is a
        hard rejection, so windowing has to cap on count as well as duration."""
        return int(self.i2v.get("max_shots_per_request", 0))

    @property
    def max_concurrency(self) -> int:
        """The ceiling comes from the endpoint, not from settings: fal enforces a
        per-user concurrency limit and exceeding it fails the run."""
        return int(self.i2v.get("max_concurrency", 0)) or super().max_concurrency

    def _submit(self, model_id: str, payload: dict) -> dict:
        try:
            return self._client.subscribe(
                model_id, arguments=payload, client_timeout=self.settings.fal_timeout_sec
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

    def build_keyframe_payload(self, prompt: str) -> dict:
        payload: dict = {"prompt": prompt, **self.t2i.get("defaults", {})}
        if self.t2i.get("size_mode") == "image_size_object":
            payload["image_size"] = {
                "width": self.settings.width,
                "height": self.settings.height,
            }
        return payload

    def build_animate_payload(
        self, image_url: str, motion_prompt: str, duration: float, persona_url: str | None
    ) -> dict:
        """Payload construction, separated from I/O so it is testable without
        network access. The parameter names are the part that silently breaks."""
        spec = self.i2v
        seconds = int(round(max(duration, spec.get("min_duration", 1))))
        seconds = min(seconds, int(spec.get("max_duration", seconds)))
        # Some endpoints expose duration as a sparse enum (Veo 3.1: 4, 6, 8), so a
        # clamped value can still be illegal. Round UP to a legal one: buying more
        # footage and trimming it is recoverable, buying less is not.
        choices = spec.get("duration_choices")
        if choices:
            longer = [c for c in sorted(choices) if c >= seconds]
            seconds = longer[0] if longer else max(choices)

        payload: dict = {"prompt": motion_prompt, **spec.get("defaults", {})}
        payload[spec["image_param"]] = image_url
        # `duration` is a string on every fal endpoint, but not the same string.
        # Seedance and Kling take a bare number ("4"); Veo 2 takes a suffixed enum
        # ("5s") and rejects anything else. Hence the format is spec data.
        payload[spec["duration_param"]] = (
            f"{seconds}s" if spec.get("duration_format") == "seconds_suffix"
            else str(seconds)
        )

        if spec.get("size_mode") == "resolution_enum":
            payload["resolution"] = spec.get("resolution", "720p")
        if spec.get("aspect_param"):
            payload[spec["aspect_param"]] = (
                "9:16" if self.settings.height > self.settings.width else "16:9"
            )

        if persona_url is not None:
            if spec.get("persona_mode") == "kling_elements":
                payload["elements"] = [{"frontal_image_url": persona_url}]
                payload["prompt"] = f"@Element1 {motion_prompt}"
            else:
                warnings.warn(
                    f"{self.i2v_id} has no reference-image parameter; persona ignored. "
                    "Use a Kling v3 endpoint for creator consistency.",
                    stacklevel=2,
                )
        return payload

    def build_sequence_payload(
        self,
        image_url: str,
        shots: list[MotionShot],
        persona_url: str | None,
    ) -> dict:
        """Payload for a multi-cut request, separated from I/O so it is testable.

        Neither the top-level `prompt` nor `duration` is sent: the shot list
        carries both, and adding a competing top-level value risks the endpoint
        preferring it. If a future endpoint needs them, that is a spec field, not
        a special case here.
        """
        spec = self.i2v
        param = spec.get("multi_prompt_param")
        if not param:
            raise VideoProviderError(
                f"{self.i2v_id} has no multi_prompt parameter. "
                "Use render_mode=per_shot, or add the parameter name to "
                "configs/fal_models.yaml after checking the model page."
            )

        duration_type = spec.get("multi_prompt_duration_type", "string")
        uses_elements = persona_url and spec.get("persona_mode") == "kling_elements"
        prefix = "@Element1 " if uses_elements else ""

        entries = []
        for shot in shots:
            if duration_type == "integer_string":
                # Kling's per-shot duration is an integer enum starting at 1, so a
                # sub-second cut has no legal representation. Nearest second with a
                # floor of 1 minimises drift; qc reports what the rounding cost
                # rather than this pretending the timeline was honoured.
                #
                # floor(x + 0.5), not round(): round() is banker's rounding, so
                # round(2.5) is 2 and a 2.5s cut would quietly lose half a second.
                seconds: float | int = max(1, math.floor(shot.duration + 0.5))
                value: str | float | int = str(seconds)
            elif duration_type == "string":
                value = str(round(shot.duration, 2))
            else:
                value = round(shot.duration, 2)
            entries.append({"prompt": f"{prefix}{shot.prompt}", "duration": value})

        payload: dict = {**spec.get("defaults", {}), param: entries}
        payload[spec["image_param"]] = image_url
        if spec.get("size_mode") == "resolution_enum":
            payload["resolution"] = spec.get("resolution", "720p")
        if spec.get("aspect_param"):
            payload[spec["aspect_param"]] = (
                "9:16" if self.settings.height > self.settings.width else "16:9"
            )
        if persona_url is not None and spec.get("persona_mode") == "kling_elements":
            payload["elements"] = [{"frontal_image_url": persona_url}]
        return payload

    def animate_sequence(
        self,
        image_path: Path,
        shots: list[MotionShot],
        out_path: Path,
        persona_ref: Path | None = None,
    ) -> Path:
        image_url = self._client.upload_file(image_path)
        persona_url = (
            self._client.upload_file(persona_ref) if persona_ref is not None else None
        )
        payload = self.build_sequence_payload(image_url, shots, persona_url)
        result = self._submit(self.i2v_id, payload)
        return self._download(self._media_url(result, self.i2v["output_key"]), out_path)

    def keyframe(self, prompt: str, out_path: Path, ref_image: Path | None = None) -> Path:
        result = self._submit(self.t2i_id, self.build_keyframe_payload(prompt))
        return self._download(self._media_url(result, self.t2i["output_key"]), out_path)

    def animate(
        self,
        image_path: Path,
        motion_prompt: str,
        duration: float,
        out_path: Path,
        persona_ref: Path | None = None,
    ) -> Path:
        image_url = self._client.upload_file(image_path)
        persona_url = (
            self._client.upload_file(persona_ref) if persona_ref is not None else None
        )
        payload = self.build_animate_payload(
            image_url, motion_prompt, duration, persona_url
        )
        result = self._submit(self.i2v_id, payload)
        return self._download(self._media_url(result, self.i2v["output_key"]), out_path)


def build_video_provider(settings: Settings) -> BaseVideoProvider:
    if settings.resolved_video_provider() == "fal":
        return FalVideoProvider(settings)
    return MockVideoProvider(settings)
