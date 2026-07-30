"""KlingAI Open Platform, called directly rather than through an aggregator.

Three things about the official API shape the code below, and all three are
different from the fal wrapper this replaces:

  * Auth is a short-lived JWT the client signs itself, not a static key. There is
    no long-lived bearer token to put in a header.
  * Generation is asynchronous. A request returns a `task_id`; the video arrives
    on a polling endpoint some minutes later.
  * Images go up as raw Base64 in the request body, so there is no upload step and
    no CDN dependency.

The endpoint specs -- paths, parameter names, floors -- are data in
configs/kling_models.yaml, for the same reason they were data before: a wrong
parameter name is accepted and ignored rather than rejected.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import math
import time
from pathlib import Path

import httpx
import yaml

from ..config import Settings
from ..errors import VideoProviderError
from .base import BaseVideoProvider, MotionShot


def _b64url(raw: bytes) -> str:
    """Base64url without padding, which is what JWT uses everywhere."""
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def encode_jwt(access_key: str, secret_key: str, issued_at: int, ttl_sec: int = 1800) -> str:
    """Sign the HS256 token Kling expects.

    Hand-rolled rather than adding PyJWT: HS256 is hmac-sha256 over
    `header.payload`, both base64url-encoded, and this keeps the provider free of
    optional dependencies.

    `nbf` is backdated five seconds on purpose -- a client clock one second ahead
    of Kling's would otherwise have its own token rejected as not-yet-valid, with
    a 401 that says nothing about clocks.
    """
    if not access_key or not secret_key:
        raise VideoProviderError(
            "KLING_ACCESS_KEY and KLING_SECRET_KEY are required for video_provider=kling"
        )
    header = _b64url(
        json.dumps({"alg": "HS256", "typ": "JWT"}, separators=(",", ":")).encode()
    )
    payload = _b64url(
        json.dumps(
            {"iss": access_key, "exp": issued_at + ttl_sec, "nbf": issued_at - 5},
            separators=(",", ":"),
        ).encode()
    )
    signing_input = f"{header}.{payload}".encode("ascii")
    signature = hmac.new(secret_key.encode(), signing_input, hashlib.sha256).digest()
    return f"{header}.{payload}.{_b64url(signature)}"


def load_kling_specs(settings: Settings) -> dict:
    path = settings.resolve_config(settings.kling_models_path)
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def aspect_ratio_for(width: int, height: int, choices: list[str]) -> str:
    """Nearest legal aspect ratio to the configured output size.

    The API has no width/height, only a ratio enum, so any size has to map onto
    one of the listed strings.
    """
    target = width / height
    return min(choices, key=lambda c: abs(target - _ratio(c)))


def _ratio(choice: str) -> float:
    w, _, h = choice.partition(":")
    return float(w) / float(h)


class KlingVideoProvider(BaseVideoProvider):
    """The official Open Platform: text to image, then image to video."""

    name = "kling"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        specs = load_kling_specs(settings)
        self.base_url = (settings.kling_base_url or specs.get("base_url", "")).rstrip("/")
        self.t2v_id = settings.kling_t2v_model
        self.t2i_id = settings.kling_t2i_model
        self.i2v_id = settings.kling_i2v_model
        # All three validated on construction, not at the first request, so a typo
        # fails the run before any credits are spent -- including a typo in the i2v
        # model, which would otherwise surface only after the keyframe was paid for.
        self.t2v = self._spec(specs, "text_to_video", self.t2v_id)
        self.t2i = self._spec(specs, "text_to_image", self.t2i_id)
        self.i2v = self._spec(specs, "image_to_video", self.i2v_id)
        # Which spec the capability properties below describe. The two endpoints
        # have separate limits, and reporting the wrong one would have the caller
        # budget prompts and pack windows against an endpoint it never calls.
        self.render = self.i2v if settings.use_first_frame else self.t2v
        self.render_id = self.i2v_id if settings.use_first_frame else self.t2v_id
        # Signing needs the secret on every call, so it is checked once here
        # rather than producing a 401 twenty minutes into a batch.
        encode_jwt(settings.kling_access_key, settings.kling_secret_key, int(time.time()))

    @staticmethod
    def _spec(specs: dict, kind: str, model_name: str) -> dict:
        known = specs.get(kind, {})
        if model_name not in known:
            raise VideoProviderError(
                f"unknown {kind} model_name {model_name!r}. "
                f"Known: {sorted(known)}. Add it to configs/kling_models.yaml."
            )
        return known[model_name]

    # --- capabilities ------------------------------------------------------ #

    @property
    def min_clip_sec(self) -> float:
        return float(self.render.get("min_duration", 0))

    @property
    def supports_multi_shot(self) -> bool:
        return "multi_prompt" in (self.render.get("capabilities") or [])

    @property
    def supports_first_frame(self) -> bool:
        return "first_frame" in (self.i2v.get("capabilities") or [])

    @property
    def max_shot_window_sec(self) -> float:
        return float(self.render.get("max_shot_window_sec", 0))

    @property
    def max_shots_per_request(self) -> int:
        return int(self.render.get("max_shots_per_request", 0))

    @property
    def max_shot_prompt_chars(self) -> int:
        return int(self.render.get("max_shot_prompt_chars", 0))

    @property
    def max_concurrency(self) -> int:
        """From settings, not the spec file: Kling's limit is an account-tier
        property, so no per-model value could be correct."""
        return self.settings.kling_max_concurrency

    # --- transport --------------------------------------------------------- #

    def _headers(self) -> dict[str, str]:
        token = encode_jwt(
            self.settings.kling_access_key,
            self.settings.kling_secret_key,
            int(time.time()),
        )
        return {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json; charset=utf-8",
        }

    def _submit(self, path: str, payload: dict) -> str:
        """Create a task, return its id."""
        url = f"{self.base_url}{path}"
        try:
            response = httpx.post(url, json=payload, headers=self._headers(), timeout=60.0)
            response.raise_for_status()
            body = response.json()
        except httpx.HTTPStatusError as exc:
            # The body carries Kling's own error code and message, which is the
            # part worth reading; the status line alone says almost nothing.
            raise VideoProviderError(
                f"kling {path} returned {exc.response.status_code}: "
                f"{exc.response.text[:300]}"
            ) from exc
        except Exception as exc:
            raise VideoProviderError(f"kling {path} failed: {exc}") from exc

        if body.get("code") not in (0, None):
            raise VideoProviderError(
                f"kling {path} rejected the request: "
                f"code={body.get('code')} {body.get('message', '')}"
            )
        task_id = (body.get("data") or {}).get("task_id")
        if not task_id:
            raise VideoProviderError(f"no task_id in kling response: {str(body)[:300]}")
        return str(task_id)

    def _poll(self, path: str, task_id: str, output_key: str) -> str:
        """Block until the task finishes, return the first result URL.

        Polling rather than the `callback_url` the API also offers: a callback
        needs a public HTTP listener, which turns a CLI into a service.
        """
        url = f"{self.base_url}{path.rstrip('/')}/{task_id}"
        deadline = time.monotonic() + self.settings.kling_timeout_sec
        last_status = "submitted"
        while time.monotonic() < deadline:
            time.sleep(self.settings.kling_poll_interval_sec)
            try:
                response = httpx.get(url, headers=self._headers(), timeout=60.0)
                response.raise_for_status()
                data = (response.json().get("data") or {})
            except Exception as exc:
                raise VideoProviderError(f"kling poll {task_id} failed: {exc}") from exc

            last_status = data.get("task_status", last_status)
            if last_status == "succeed":
                return self._result_url(data, output_key)
            if last_status == "failed":
                raise VideoProviderError(
                    f"kling task {task_id} failed: {data.get('task_status_msg', '')}"
                )
        raise VideoProviderError(
            f"kling task {task_id} still {last_status} after "
            f"{self.settings.kling_timeout_sec:.0f}s. Raise STYLELOOM_KLING_TIMEOUT_SEC, "
            "or check the task in the Kling console -- it may still be running and "
            "will still be billed."
        )

    @staticmethod
    def _result_url(data: dict, output_key: str) -> str:
        items = (data.get("task_result") or {}).get(output_key) or []
        if items and isinstance(items[0], dict) and items[0].get("url"):
            return items[0]["url"]
        raise VideoProviderError(f"no {output_key} URL in kling result: {str(data)[:300]}")

    @staticmethod
    def _download(url: str, out_path: Path) -> Path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with httpx.stream("GET", url, timeout=300.0, follow_redirects=True) as r:
            r.raise_for_status()
            with out_path.open("wb") as fh:
                for chunk in r.iter_bytes():
                    fh.write(chunk)
        return out_path

    # --- payloads ---------------------------------------------------------- #
    #
    # Separated from the calls above so the shape -- the part that breaks
    # silently -- is testable without a key or a network.

    def _aspect_ratio(self, spec: dict) -> str:
        return aspect_ratio_for(
            self.settings.width,
            self.settings.height,
            spec.get("aspect_ratio_choices") or ["9:16", "16:9", "1:1"],
        )

    def _duration_string(self, spec: dict, duration: float) -> str:
        """Whole seconds for a single-cut request, always rounded UP.

        `render_shot` trims a long clip down and cannot extend a short one, so
        rounding to nearest would buy 4s for a 4.4s cut and leave the timeline
        quietly short. `round()` is also banker's: round(4.5) is 4.
        """
        wanted = max(duration, float(spec.get("min_duration", 1)))
        seconds = math.ceil(wanted - 1e-9)
        return str(min(seconds, int(spec.get("max_duration", seconds))))

    def shot_billed_duration(self, seconds: float) -> float:
        """One cut inside a multi-shot request, to whole seconds with a floor of 1.

        Nearest rather than up, unlike `_duration_string`: a multi-shot clip is
        never trimmed, so rounding up would stretch every video. floor(x + 0.5)
        rather than banker's round(), where round(2.5) is 2.
        """
        if self.render.get("multi_prompt_duration_type") == "integer_string":
            return float(max(1, math.floor(seconds + 0.5)))
        return seconds

    @staticmethod
    def encode_image(path: Path) -> str:
        """Raw Base64, with no data URI prefix.

        The prefix is the documented 400 on this API and the easiest mistake to
        make, since every browser-facing example of Base64 image data carries one.
        """
        return base64.b64encode(path.read_bytes()).decode("ascii")

    def _attach_first_frame(self, spec: dict, payload: dict, frame: Path) -> None:
        """Put the start frame where this endpoint expects it.

        Two shapes, and the difference is not cosmetic. image2video takes a flat
        Base64 string; omni-video takes a typed list where `type: first_frame` is
        what separates a start frame from a mere style reference. An untyped entry
        on omni is accepted and billed as a reference image, so the video does not
        start on the frame and nothing in the response says so.
        """
        param = spec.get("first_frame_param")
        if not param:
            raise VideoProviderError(
                f"{self.i2v_id} has no first_frame parameter in "
                "configs/kling_models.yaml, so it cannot take a start frame. "
                "Set STYLELOOM_USE_FIRST_FRAME=false, or pick a model that can."
            )
        encoded = self.encode_image(frame)
        if spec.get("first_frame_format") == "typed_list":
            payload[param] = [
                {
                    "type": spec.get("first_frame_type", "first_frame"),
                    "url": encoded,
                    "id": spec.get("first_frame_id", "image_1"),
                }
            ]
        else:
            payload[param] = encoded

    # --- payloads ---------------------------------------------------------- #
    #
    # Separated from the calls so the shape -- the part that breaks silently --
    # is testable without a key or a network.

    def _spec_for(self, first_frame: Path | None) -> tuple[dict, str]:
        """i2v when there is a frame to animate, t2v when there is not."""
        if first_frame is not None:
            return self.i2v, self.i2v_id
        return self.t2v, self.t2v_id

    def build_generate_payload(
        self, prompt: str, duration: float, first_frame: Path | None = None
    ) -> dict:
        spec, model_id = self._spec_for(first_frame)
        payload: dict = {
            "model_name": model_id,
            "mode": self.settings.kling_mode,
            "prompt": prompt,
            **spec.get("defaults", {}),
        }
        # Only when the endpoint has nothing to read it off. With a start frame the
        # ratio comes from the image, and stating it again can only disagree.
        if spec.get("sends_aspect_ratio", True):
            payload["aspect_ratio"] = self._aspect_ratio(spec)
        if first_frame is not None:
            self._attach_first_frame(spec, payload, first_frame)
        payload[spec["duration_param"]] = self._duration_string(spec, duration)
        limit = int(spec.get("max_prompt_chars", 0))
        if limit and len(prompt) > limit:
            raise VideoProviderError(
                f"prompt is {len(prompt)} characters, over {model_id}'s "
                f"{limit}-character limit."
            )
        return payload

    def build_image_payload(
        self, prompt: str, reference: Path | None = None
    ) -> dict:
        """A first frame. Optionally consistent with an anchor already generated.

        When `reference` is given the prompt is prefixed with the endpoint's
        reference token, because a reference image that is never addressed in the
        prompt is weighted as loose style guidance rather than as the subject to
        preserve -- which is precisely the identity drift this stage exists to stop.
        """
        spec = self.t2i
        if reference is not None:
            token = spec.get("reference_token", "")
            prompt = f"{token} {prompt}".strip() if token else prompt
        limit = int(spec.get("max_prompt_chars", 0))
        if limit and len(prompt) > limit:
            raise VideoProviderError(
                f"image prompt is {len(prompt)} characters, over {self.t2i_id}'s "
                f"{limit}-character limit."
            )
        payload: dict = {
            "model_name": self.t2i_id,
            "prompt": prompt,
            "aspect_ratio": self._aspect_ratio(spec),
            **spec.get("defaults", {}),
        }
        if reference is not None:
            param = spec.get("reference_param")
            if not param:
                raise VideoProviderError(
                    f"{self.t2i_id} has no reference_param in "
                    "configs/kling_models.yaml, so it cannot hold one anchor "
                    "across frames."
                )
            payload[param] = self.encode_image(reference)
        return payload

    def format_semicolon_shots(self, shots: list[MotionShot]) -> str:
        """The docs' prose form of a shot list: `shot n, m, words;` per cut.

        `n` is the 1-based shot number and `m` its duration in whole seconds. Kling
        documents this for hand-written Omni prompts, where there is one `prompt`
        field and no array to put a storyboard in.

        Not the default transport -- see `multi_shot_syntax` in
        configs/kling_models.yaml for why. Kept because it is the only multi-shot
        form available on a request that has no `multi_prompt` array, and because
        the failure mode when it is wrong is silent: prose that does not parse as a
        shot list becomes one long shot and the task still reports success.

        Semicolons inside a prompt are stripped rather than escaped. There is no
        documented escape, so a stray one would end the shot early and shift every
        cut after it.
        """
        parts = []
        for index, shot in enumerate(shots, start=1):
            seconds = int(self.shot_billed_duration(shot.duration))
            words = shot.prompt.replace(";", ",").strip()
            parts.append(f"shot {index}, {seconds}, {words}")
        return "; ".join(parts) + ";"

    def build_sequence_payload(
        self, shots: list[MotionShot], first_frame: Path | None = None
    ) -> dict:
        """Multi-shot storyboard, optionally anchored on a start frame.

        `multi_shot` and `shot_type` must both be set for `multi_prompt` to be
        read at all, and setting them makes the top-level `prompt` invalid.
        """
        spec, model_id = self._spec_for(first_frame)
        payload: dict = {
            "model_name": model_id,
            "mode": self.settings.kling_mode,
            **spec.get("defaults", {}),
            spec["multi_shot_param"]: "true",
            "shot_type": spec.get("shot_type", "customize"),
        }
        if spec.get("sends_aspect_ratio", True):
            payload["aspect_ratio"] = self._aspect_ratio(spec)
        if first_frame is not None:
            self._attach_first_frame(spec, payload, first_frame)

        if spec.get("multi_shot_syntax") == "semicolon_prompt":
            # The whole storyboard in the one text field, which is the point of
            # this syntax. `multi_prompt` is left out entirely rather than sent
            # alongside: the two would be two different shot lists in one request.
            payload["prompt"] = self.format_semicolon_shots(shots)
            return payload

        param = spec.get("multi_prompt_param")
        if not param:
            raise VideoProviderError(
                f"{model_id} has no multi_prompt parameter. "
                "Use render_mode=per_shot, or add the parameter name to "
                "configs/kling_models.yaml after checking the docs."
            )

        entries = []
        for index, shot in enumerate(shots, start=1):
            # Through shot_billed_duration, not a second copy of the rule: the
            # caller packs windows with that method, and a payload that rounded
            # differently would overflow the window the caller just verified.
            billed = self.shot_billed_duration(shot.duration)
            if spec.get("multi_prompt_duration_type") == "integer_string":
                value: str | float = str(int(billed))
            else:
                value = round(billed, 2)
            limit = int(spec.get("max_shot_prompt_chars", 0))
            if limit and len(shot.prompt) > limit:
                # Refused, not trimmed. The caller builds prompts to this budget
                # already; if one arrives over it, something upstream changed and
                # a silent trim would cut mid-clause -- handing the model half a
                # sentence and charging for the result.
                raise VideoProviderError(
                    f"storyboard entry {index} is {len(shot.prompt)} characters, "
                    f"over {model_id}'s {limit}-character limit for a shot "
                    "prompt. Shorten the style keywords, or render with "
                    "render_mode=per_shot, where the whole prompt goes in the "
                    f"top-level field ({spec.get('max_prompt_chars', '?')} chars)."
                )
            entry: dict = {"prompt": shot.prompt, "duration": value}
            if spec.get("multi_prompt_indexed"):
                entry = {"index": index, **entry}
            entries.append(entry)

        payload[param] = entries
        return payload

    # --- interface --------------------------------------------------------- #

    def generate_image(
        self, prompt: str, out_path: Path, reference: Path | None = None
    ) -> Path:
        path = self.t2i["path"]
        task_id = self._submit(path, self.build_image_payload(prompt, reference))
        return self._download(
            self._poll(path, task_id, self.t2i["output_key"]), out_path
        )

    def generate(
        self,
        prompt: str,
        duration: float,
        out_path: Path,
        first_frame: Path | None = None,
    ) -> Path:
        spec, _ = self._spec_for(first_frame)
        path = spec["path"]
        task_id = self._submit(
            path, self.build_generate_payload(prompt, duration, first_frame)
        )
        return self._download(
            self._poll(path, task_id, spec["output_key"]), out_path
        )

    def generate_sequence(
        self,
        shots: list[MotionShot],
        out_path: Path,
        first_frame: Path | None = None,
    ) -> Path:
        spec, _ = self._spec_for(first_frame)
        path = spec["path"]
        task_id = self._submit(
            path, self.build_sequence_payload(shots, first_frame)
        )
        return self._download(
            self._poll(path, task_id, spec["output_key"]), out_path
        )
