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

    Hand-rolled rather than pulling in PyJWT, and this is the one place in the
    codebase where that is the right call. HS256 is `hmac-sha256` over
    `header.payload` with both parts base64url-encoded -- twelve lines against a
    dependency that would otherwise be the only thing standing between a fresh
    clone and a working provider. The fal path needed `pip install fal-client`;
    this one needs nothing that httpx did not already bring.

    `nbf` is set five seconds in the past deliberately. It is not defensive
    padding -- the client's clock only has to be a second ahead of Kling's for a
    token stamped "not valid before now" to be rejected on arrival, and the
    resulting 401 says nothing about clocks.
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

    The official API has no width/height, only a ratio enum, so an output size
    that is not exactly 9:16 still has to become one of the listed strings.
    Nearest by ratio rather than a lookup table: a 720x1280 default and someone
    else's 1080x1920 both land on 9:16 without either being special-cased.
    """
    target = width / height
    return min(choices, key=lambda c: abs(target - _ratio(c)))


def _ratio(choice: str) -> float:
    w, _, h = choice.partition(":")
    return float(w) / float(h)


class KlingVideoProvider(BaseVideoProvider):
    """The official Open Platform, keyframe then motion.

    Kling's own image model generates the keyframe, so the whole pipeline runs on
    one account and one set of credentials.
    """

    name = "kling"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        specs = load_kling_specs(settings)
        self.base_url = (settings.kling_base_url or specs.get("base_url", "")).rstrip("/")
        t2v_id = settings.kling_t2v_model
        self.t2v_id = t2v_id
        # Validated on construction, not at the first request, so a typo fails the
        # run before any credits are spent.
        self.t2v = self._spec(specs, "text_to_video", self.t2v_id)
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
        return float(self.t2v.get("min_duration", 0))

    @property
    def supports_multi_shot(self) -> bool:
        return "multi_prompt" in (self.t2v.get("capabilities") or [])

    @property
    def max_shot_window_sec(self) -> float:
        return float(self.t2v.get("max_shot_window_sec", 0))

    @property
    def max_shots_per_request(self) -> int:
        return int(self.t2v.get("max_shots_per_request", 0))

    @property
    def max_shot_prompt_chars(self) -> int:
        return int(self.t2v.get("max_shot_prompt_chars", 0))

    @property
    def max_concurrency(self) -> int:
        """From settings, not from the spec file.

        fal published a per-endpoint concurrency limit, so the ceiling was spec
        data. Kling's limit is a property of the account tier instead -- the same
        endpoint allows more parallel tasks on a larger plan -- so it cannot be
        recorded per model and has to be told to us.
        """
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
        needs a public HTTP listener, which turns a CLI into a service. The
        interval is a setting because the right value depends on the clip -- a 3s
        cut and a 15s multi-shot request are minutes apart.
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

    def _aspect_ratio(self) -> str:
        return aspect_ratio_for(
            self.settings.width,
            self.settings.height,
            self.t2v.get("aspect_ratio_choices") or ["9:16", "16:9", "1:1"],
        )

    def _duration_string(self, duration: float) -> str:
        """Whole seconds for a single-cut request, always rounded UP.

        `duration` is only legal as an integer, and `render_shot` trims the result
        down to the cut length afterwards -- but it only trims when the clip is
        longer, because a short clip cannot be extended. Rounding to nearest would
        buy 4s for a 4.4s cut, and nothing downstream would notice: the clip is
        returned untrimmed and the timeline is quietly 0.4s short.

        `round()` would be doubly wrong here, being banker's rounding: round(4.5)
        is 4, so exactly-half cuts lose the most.
        """
        spec = self.t2v
        wanted = max(duration, float(spec.get("min_duration", 1)))
        seconds = math.ceil(wanted - 1e-9)
        return str(min(seconds, int(spec.get("max_duration", seconds))))

    def shot_billed_duration(self, seconds: float) -> float:
        """One cut inside a multi-shot request, to whole seconds with a floor of 1.

        Nearest rather than up, unlike `_duration_string`, and the difference is
        not an oversight. A multi-shot generation comes back as one clip that is
        never trimmed, so rounding down shortens a cut rather than producing an
        untrimmable file -- and nearest is what minimises total drift across the
        storyboard. Rounding up here would stretch every video instead.

        floor(x + 0.5) rather than round(), which is banker's: round(2.5) is 2,
        so a 2.5s cut would quietly lose half a second.
        """
        if self.t2v.get("multi_prompt_duration_type") == "integer_string":
            return float(max(1, math.floor(seconds + 0.5)))
        return seconds

    # --- payloads ---------------------------------------------------------- #
    #
    # Separated from the calls so the shape -- the part that breaks silently --
    # is testable without a key or a network.

    def build_generate_payload(self, prompt: str, duration: float) -> dict:
        spec = self.t2v
        payload: dict = {
            "model_name": self.t2v_id,
            "mode": self.settings.kling_mode,
            "prompt": prompt,
            "aspect_ratio": self._aspect_ratio(),
            **spec.get("defaults", {}),
        }
        payload[spec["duration_param"]] = self._duration_string(duration)
        limit = int(spec.get("max_prompt_chars", 0))
        if limit and len(prompt) > limit:
            raise VideoProviderError(
                f"prompt is {len(prompt)} characters, over {self.t2v_id}'s "
                f"{limit}-character limit."
            )
        return payload

    def build_sequence_payload(self, shots: list[MotionShot]) -> dict:
        """Multi-shot storyboard.

        Two rules from the official schema. `multi_shot` and `shot_type` must both
        be set for `multi_prompt` to be read at all, and setting them makes the
        top-level `prompt` invalid -- so it is never added rather than added and
        popped, since a competing top-level value is how a storyboard silently
        becomes one shot.
        """
        spec = self.t2v
        param = spec.get("multi_prompt_param")
        if not param:
            raise VideoProviderError(
                f"{self.t2v_id} has no multi_prompt parameter. "
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
                    f"over {self.t2v_id}'s {limit}-character limit for a shot "
                    "prompt. Shorten the style keywords, or render with "
                    "render_mode=per_shot, where the whole prompt goes in the "
                    f"top-level field ({spec.get('max_prompt_chars', '?')} chars)."
                )
            entry: dict = {"prompt": shot.prompt, "duration": value}
            if spec.get("multi_prompt_indexed"):
                entry = {"index": index, **entry}
            entries.append(entry)

        return {
            "model_name": self.t2v_id,
            "mode": self.settings.kling_mode,
            "aspect_ratio": self._aspect_ratio(),
            **spec.get("defaults", {}),
            spec["multi_shot_param"]: "true",
            "shot_type": spec.get("shot_type", "customize"),
            param: entries,
        }

    # --- interface --------------------------------------------------------- #

    def generate(self, prompt: str, duration: float, out_path: Path) -> Path:
        path = self.t2v["path"]
        task_id = self._submit(path, self.build_generate_payload(prompt, duration))
        return self._download(
            self._poll(path, task_id, self.t2v["output_key"]), out_path
        )

    def generate_sequence(self, shots: list[MotionShot], out_path: Path) -> Path:
        path = self.t2v["path"]
        task_id = self._submit(path, self.build_sequence_payload(shots))
        return self._download(
            self._poll(path, task_id, self.t2v["output_key"]), out_path
        )
