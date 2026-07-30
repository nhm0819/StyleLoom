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
from ..media import image_size
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


def _is_v3(spec: dict) -> bool:
    """Whether this endpoint speaks the 3.0 request and response shape.

    The two generations differ in the request envelope, in where the task id is
    read from, in which endpoint answers a poll, and in the string that means
    finished. None of those disagreements produce an error, so the flag is checked
    rather than inferred from the path.
    """
    return spec.get("api") == "v3"


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
        """Either transport counts. The legacy API carries a shot list in a
        `multi_prompt` array; the 3.0 API carries it as a semicolon shot list
        inside the prompt. Both put several cuts in one generation, which is the
        capability the caller is asking about."""
        caps = set(self.render.get("capabilities") or [])
        return bool(caps & {"multi_prompt", "semicolon_prompt"})

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

    def _submit(self, spec: dict, payload: dict) -> str:
        """Create a task, return its id."""
        path = spec["path"]
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
        # The 3.0 endpoints return `data.id`; the legacy ones `data.task_id`.
        # Reading the wrong key aborts a task that was created and will be billed.
        key = "id" if _is_v3(spec) else "task_id"
        task_id = (body.get("data") or {}).get(key)
        if not task_id:
            raise VideoProviderError(f"no {key} in kling response: {str(body)[:300]}")
        return str(task_id)

    def _poll(self, spec: dict, task_id: str) -> str:
        """Block until the task finishes, return the first result URL.

        Two query shapes, and every part of the disagreement is silent. The 3.0
        endpoints share one task endpoint -- GET /tasks?task_ids=... -- which
        answers with a *list*, calls the field `status`, and says `succeeded`. The
        legacy endpoints answer on GET {create path}/{id} with a single object
        whose field is `task_status` and whose terminal value is `succeed`. Polling
        with the wrong pair waits out the full timeout on a task that finished
        minutes earlier, and the generation is billed either way.

        Polling rather than the `callback_url` the API also offers: a callback
        needs a public HTTP listener, which turns a CLI into a service.
        """
        v3 = _is_v3(spec)
        if v3:
            url = f"{self.base_url}{spec['task_path']}"
            params: dict[str, str] | None = {"task_ids": task_id}
            status_key, message_key, done = "status", "message", "succeeded"
        else:
            url = f"{self.base_url}{spec['path'].rstrip('/')}/{task_id}"
            params = None
            status_key, message_key, done = "task_status", "task_status_msg", "succeed"

        deadline = time.monotonic() + self.settings.kling_timeout_sec
        last_status = "submitted"
        while time.monotonic() < deadline:
            time.sleep(self.settings.kling_poll_interval_sec)
            try:
                response = httpx.get(
                    url, params=params, headers=self._headers(), timeout=60.0
                )
                response.raise_for_status()
                task = self._task_record(spec, response.json(), task_id)
            except Exception as exc:
                raise VideoProviderError(f"kling poll {task_id} failed: {exc}") from exc

            last_status = task.get(status_key, last_status)
            if last_status == done:
                return self._result_url(spec, task)
            if last_status == "failed":
                raise VideoProviderError(
                    f"kling task {task_id} failed: {task.get(message_key, '')}"
                )
        raise VideoProviderError(
            f"kling task {task_id} still {last_status} after "
            f"{self.settings.kling_timeout_sec:.0f}s. Raise STYLELOOM_KLING_TIMEOUT_SEC, "
            "or check the task in the Kling console -- it may still be running and "
            "will still be billed."
        )

    @staticmethod
    def _task_record(spec: dict, body: dict, task_id: str) -> dict:
        """The one task in a poll response, whichever shape it arrived in.

        `/tasks` answers with a list even for a single id, and can answer with a
        short one just after submission, so a record that is not there yet is
        "keep waiting" rather than an error -- an empty dict has no terminal
        status and the loop simply polls again.
        """
        data = body.get("data")
        if not _is_v3(spec):
            return data or {}
        records = [r for r in (data or []) if isinstance(r, dict)]
        for record in records:
            if str(record.get("id")) == task_id:
                return record
        # One id was queried, so one unlabelled record is that id.
        return records[0] if len(records) == 1 else {}

    @staticmethod
    def _result_url(spec: dict, task: dict) -> str:
        if _is_v3(spec):
            # A typed `outputs` list rather than task_result.<key>, and `url` not
            # `watermark_url`: both are present on success and the watermarked one
            # is unusable as a deliverable.
            wanted = spec.get("output_type", "video")
            for item in task.get("outputs") or []:
                if item.get("type") == wanted and item.get("url"):
                    return item["url"]
            raise VideoProviderError(
                f"no {wanted} output in kling result: {str(task)[:300]}"
            )
        output_key = spec["output_key"]
        items = (task.get("task_result") or {}).get(output_key) or []
        if items and isinstance(items[0], dict) and items[0].get("url"):
            return items[0]["url"]
        raise VideoProviderError(f"no {output_key} URL in kling result: {str(task)[:300]}")

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

    def shot_billed_duration(self, seconds: float) -> float:
        """One cut inside a multi-shot request, to whole seconds with a floor of 1.

        Nearest rather than up: the cuts land inside one generation and are not
        trimmed individually, so rounding up would stretch every video. floor(x +
        0.5) rather than banker's round(), where round(2.5) is 2.
        """
        integral = (
            _is_v3(self.render)  # `m` in "shot n, m, words" is whole seconds
            or self.render.get("multi_prompt_duration_type") == "integer_string"
        )
        if integral:
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
        self.check_image(spec, frame, "first frame")
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

    # --- constraints ------------------------------------------------------- #
    #
    # Three checks, one constraint each, no overlap. They are separate because they
    # are enforced at different moments: the count and the per-shot text are
    # properties of the list the caller built, the durations are normalised while
    # the list is turned into a request, and the assembled prompt can only be
    # measured once it exists.
    #
    # All of them are cheap, and all of them guard against the same failure: this
    # endpoint accepts a malformed shot list as ordinary prose, renders it as one
    # long shot, reports `succeeded`, and bills it. There is nothing in the
    # response to check afterwards, so the checking happens before.

    def validate_shot_list(
        self, spec: dict, model_id: str, shots: list[MotionShot]
    ) -> None:
        """The count and the per-shot text budget.

        The 512-character budget is enforced on every shot, including a list of one
        where the endpoint would allow the whole 3072: `render` re-packs windows
        whenever the storyboard or the endpoint's limits change, so a prompt that is
        legal only while it happens to be alone in its window is a prompt that
        breaks on the next run. `storyboard` builds to the same number.
        """
        if not shots:
            raise VideoProviderError("a render request needs at least one shot.")
        cap = int(spec.get("max_shots_per_request", 0))
        if cap and len(shots) > cap:
            raise VideoProviderError(
                f"{len(shots)} shots in one request, over {model_id}'s limit of "
                f"{cap}. `render` packs windows to this limit, so a list this long "
                "means max_shots_per_request in configs/kling_models.yaml "
                "disagrees with the endpoint."
            )
        for index, shot in enumerate(shots, start=1):
            self._check_shot_prompt(spec, model_id, index, shot.prompt)

    @staticmethod
    def check_image(spec: dict, path: Path, role: str) -> None:
        """One image against the endpoint's documented limits.

        Checked before the file is base64'd rather than after the far end rejects
        it: encoding inflates the body by a third, so an oversized frame is paid for
        in upload time before the 400 arrives, and a frame outside the ratio range
        is rejected with nothing said about which of the two it was.
        """
        limits = spec.get("image_constraints") or {}
        if not limits:
            return
        formats = [str(f).lower() for f in (limits.get("formats") or [])]
        if formats and path.suffix.lower() not in formats:
            raise VideoProviderError(
                f"{role} {path.name} is {path.suffix or 'extensionless'}; this "
                f"endpoint accepts {', '.join(formats)}."
            )
        max_bytes = int(limits.get("max_bytes", 0))
        if max_bytes and path.stat().st_size > max_bytes:
            raise VideoProviderError(
                f"{role} {path.name} is {path.stat().st_size / 1e6:.1f}MB, over the "
                f"{max_bytes / 1e6:.0f}MB this endpoint accepts."
            )
        width, height = image_size(path)
        min_side = int(limits.get("min_side", 0))
        if min_side and min(width, height) < min_side:
            raise VideoProviderError(
                f"{role} {path.name} is {width}x{height}; both sides must be at "
                f"least {min_side}px."
            )
        max_ratio = float(limits.get("max_ratio", 0))
        if max_ratio and not 1 / max_ratio <= width / height <= max_ratio:
            raise VideoProviderError(
                f"{role} {path.name} is {width}x{height}, a ratio of "
                f"{width / height:.2f}:1. This endpoint accepts "
                f"1:{max_ratio:g} to {max_ratio:g}:1."
            )

    @staticmethod
    def _check_prompt_limit(spec: dict, model_id: str, prompt: str) -> None:
        limit = int(spec.get("max_prompt_chars", 0))
        if limit and len(prompt) > limit:
            raise VideoProviderError(
                f"prompt is {len(prompt)} characters, over {model_id}'s "
                f"{limit}-character limit."
            )

    @staticmethod
    def _check_shot_prompt(spec: dict, model_id: str, index: int, prompt: str) -> None:
        """The per-cut budget, which is far tighter than the whole-prompt one.

        Refused, not trimmed. The caller builds prompts to this budget already; if
        one arrives over it, something upstream changed and a silent trim would cut
        mid-clause -- handing the model half a sentence and charging for the result.
        """
        limit = int(spec.get("max_shot_prompt_chars", 0))
        if limit and len(prompt) > limit:
            raise VideoProviderError(
                f"storyboard entry {index} is {len(prompt)} characters, "
                f"over {model_id}'s {limit}-character limit for a shot "
                "prompt. Shorten the style keywords, or the `look.keywords` in "
                "the style -- `storyboard` budgets to this number and drops "
                "clauses to reach it, so an entry over it means the fixed part "
                "alone no longer fits."
            )

    def _build_v3_payload(
        self,
        spec: dict,
        model_id: str,
        text: str,
        duration: int,
        multi_shot: bool,
        first_frame: Path | None,
    ) -> dict:
        """The 3.0 request: a prompt, a `settings` object, and nothing flat.

        Nothing from the legacy shape survives. `model_name` is in the path, `mode`
        does not exist, `negative_prompt` does not exist, and `aspect_ratio` moved
        inside `settings`. Sending the old names is not an error: unknown keys are
        ignored, so the request succeeds and generates five seconds of defaults.

        The two 3.0 video endpoints do not agree on where the prompt goes.
        image-to-video takes a typed `contents` array, because it also has to carry
        the start frame and the type is what distinguishes a first frame from an
        Element reference. text-to-video has no image and keeps a top-level `prompt`
        string. Putting a `contents` array on text-to-video, or a bare `prompt` on
        image-to-video, is the silent-ignore failure this spec file exists to stop.
        """
        clause = spec.get("negative_clause", "")
        if clause and not multi_shot:
            # Negatives have no field of their own here; the prompt "can include
            # positive and negative descriptions". Single-cut requests only -- see
            # `negative_clause` in configs/kling_models.yaml for why a semicolon
            # shot list does not get one.
            text = f"{text} {clause}".strip()
        self._check_prompt_limit(spec, model_id, text)

        settings: dict = {
            **(spec.get("settings_defaults") or {}),
            # `multi_shot` defaults to TRUE on this API, so a single-cut request has
            # to say false out loud. Left off, the model may cut the one shot into
            # several -- and per_shot mode depends on one file being one cut.
            "multi_shot": multi_shot,
            spec.get("duration_param", "duration"): duration,
        }
        resolution = (spec.get("resolution_by_mode") or {}).get(self.settings.kling_mode)
        if resolution:
            settings["resolution"] = resolution
        # Only where the endpoint has nothing to read it off. With a start frame the
        # ratio comes from the image, and stating it again can only disagree.
        if spec.get("sends_aspect_ratio", True):
            settings["aspect_ratio"] = self._aspect_ratio(spec)

        payload: dict = {}
        if spec.get("prompt_format") == "contents":
            contents: list[dict] = [{"type": "prompt", "text": text}]
            if first_frame is not None:
                self.check_image(spec, first_frame, "first frame")
                contents.append(
                    {
                        "type": spec.get("first_frame_type", "first_frame"),
                        "url": self.encode_image(first_frame),
                    }
                )
            payload["contents"] = contents
        else:
            payload["prompt"] = text
        payload["settings"] = settings
        return payload

    def build_image_payload(
        self, prompt: str, reference: Path | None = None
    ) -> dict:
        """A first frame. Optionally consistent with an anchor already generated.

        `reference` goes in the `image` field as raw Base64. The image endpoint
        documents no token addressing, so nothing is prefixed to the prompt --
        `reference_token` stays supported in the spec file for an endpoint that
        does, and is unset for this one.
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
            self.check_image(spec, reference, "reference image")
            payload[param] = self.encode_image(reference)
        return payload

    def plan_shot_seconds(self, spec: dict, shots: list[MotionShot]) -> list[int]:
        """Whole seconds per cut, summing to a total the endpoint will accept.

        Three constraints bind at once in a semicolon shot list: every `m` is at
        least 1s, the sum of them must equal `settings.duration`, and that total
        must sit inside the duration enum. Rounding each cut on its own satisfies
        none of them -- a two-cut hook of 1.2s + 1.3s rounds to 1 + 1 = 2s, which
        is under the 3s floor, and the request is rejected whole.

        Any deficit goes on the last cut rather than being spread across all of
        them. Short-form pacing is front-loaded and the hook is first, so the
        closing cut is the least visible place to put a second the timeline did not
        ask for. `qc` measures the drift that results.
        """
        seconds = [int(self.shot_billed_duration(s.duration)) for s in shots]
        total = sum(seconds)
        deficit = int(spec.get("min_duration", 0)) - total
        if deficit > 0:
            seconds[-1] += deficit
        ceiling = int(spec.get("max_duration", 0))
        if ceiling and total > ceiling:
            # Refused rather than trimmed: the caller packs windows against this
            # same quantisation, so an overflow means a single cut is longer than a
            # whole request can be, and dropping the excess would silently delete
            # video that the storyboard asked for.
            raise VideoProviderError(
                f"the shot list totals {total}s, over the {ceiling}s a single "
                "request can hold. `render` packs windows to this ceiling, so a "
                "list this long means one cut is longer than a whole request -- "
                "shorten it in the storyboard."
            )
        return seconds

    def plan_shot_durations(self, durations: list[float]) -> list[float]:
        """What the render endpoint will actually run for each cut in one request.

        The interface method, against the spec the caller will reach. The private
        `plan_shot_seconds` takes a spec because payload building already knows
        which one it is holding.
        """
        shots = [MotionShot(prompt="", duration=d) for d in durations]
        return [float(s) for s in self.plan_shot_seconds(self.render, shots)]

    def format_semicolon_shots(
        self, shots: list[MotionShot], seconds: list[int] | None = None
    ) -> str:
        """The documented form of a 3.0 shot list: `shot n, m, words;` per cut.

        `n` is the 1-based shot number and `m` its duration in whole seconds. This
        is not an alternative to a `multi_prompt` array on the 3.0 API -- it is the
        only multi-shot form the API has, and the failure mode when it is malformed
        is silent: prose that does not parse as a shot list becomes one long shot
        and the task still reports success.

        Semicolons inside a prompt are replaced rather than escaped. There is no
        documented escape, so a stray one would end the shot early and shift every
        cut after it.
        """
        lengths = seconds if seconds is not None else [
            int(self.shot_billed_duration(s.duration)) for s in shots
        ]
        parts = []
        for index, (shot, length) in enumerate(
            zip(shots, lengths, strict=True), start=1
        ):
            words = shot.prompt.replace(";", ",").strip()
            parts.append(f"shot {index}, {length}, {words}")
        return "; ".join(parts) + ";"

    def build_sequence_payload(
        self, shots: list[MotionShot], first_frame: Path | None = None
    ) -> dict:
        """Multi-shot storyboard, optionally anchored on a start frame.

        On v3 the storyboard is the prompt: one text field holding
        `shot n, m, words;` per cut, with `settings.multi_shot` on and
        `settings.duration` equal to the sum of the shot durations. A list of one
        shot degenerates to a plain single-cut request, because that is what it is.

        On the legacy API it is an array instead, where `multi_shot` and
        `shot_type` must both be set for `multi_prompt` to be read at all, and
        setting them makes the top-level `prompt` invalid.
        """
        spec, model_id = self._spec_for(first_frame)
        if _is_v3(spec):
            self.validate_shot_list(spec, model_id, shots)
            seconds = self.plan_shot_seconds(spec, shots)
            # A list of one shot is just a prompt. Writing `shot 1, 3, words;` for it
            # spends the shot-list format on nothing, and it would forfeit the
            # negative clause, which only goes on requests that are not multi-shot.
            single = len(shots) == 1
            text = (
                shots[0].prompt
                if single
                else self.format_semicolon_shots(shots, seconds)
            )
            return self._build_v3_payload(
                spec,
                model_id,
                text,
                sum(seconds),
                multi_shot=not single,
                first_frame=first_frame,
            )
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
                f"{model_id} has no multi_prompt parameter, so it cannot carry "
                "a shot list. Pick one of the 3.0 entries, or add the parameter "
                "name to configs/kling_models.yaml after checking the docs."
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
            self._check_shot_prompt(spec, model_id, index, shot.prompt)
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
        task_id = self._submit(self.t2i, self.build_image_payload(prompt, reference))
        return self._download(self._poll(self.t2i, task_id), out_path)

    def generate_sequence(
        self,
        shots: list[MotionShot],
        out_path: Path,
        first_frame: Path | None = None,
    ) -> Path:
        spec, _ = self._spec_for(first_frame)
        task_id = self._submit(spec, self.build_sequence_payload(shots, first_frame))
        return self._download(self._poll(spec, task_id), out_path)
