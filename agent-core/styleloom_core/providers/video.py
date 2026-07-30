"""Video providers: two-stage `text -> keyframe -> motion clip`.

Why two stages instead of straight text-to-video: shot-to-shot consistency of
person and colour grade is what makes a short-form set read as one channel.
Locking the look at the image stage leaves the video model responsible only for
motion. Straight text-to-video re-rolls the look on every shot, and by shot six
the grade has drifted. See docs/TOOL_RATIONALE.md.

  * `mock` - ffmpeg only. Produces a real keyframe JPEG and a real MP4 with a
             slow push-in, so the whole harness runs with zero keys.
  * `kling` - KlingAI Open Platform, called directly. Lives in `kling.py`
              because its async task protocol and self-signed JWT have nothing
              in common with the offline renderer. Model names are NOT
              hardcoded; they are settings, and their payload differences are
              data in configs/kling_models.yaml.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

from ..config import Settings
from ..media import concat, gradient_keyframe, push_in
from .base import BaseVideoProvider, MotionShot


class MockVideoProvider(BaseVideoProvider):
    name = "mock"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def _still(self, prompt: str, out_path: Path) -> Path:
        """A gradient derived from the prompt.

        Internal, not part of the provider interface: this pipeline is text to
        video, and the still is only how the offline renderer fakes one. Hashing
        the prompt is what makes different cuts look different, which is what
        makes them detectable as cuts.
        """
        digest = hashlib.sha1(prompt.encode()).hexdigest()
        return gradient_keyframe(digest, out_path, self.settings)

    @property
    def supports_first_frame(self) -> bool:
        """True, and really implemented: `generate_image` writes a JPEG and both
        render methods start from it when given one.

        Not a stub, for the same reason `supports_multi_shot` is not: if the
        offline provider declared no first-frame support, the keyframe stage would
        never run in a test and the first time it ran would be against a paid API.

        What it cannot tell you is whether a real endpoint *honours* the frame.
        ffmpeg starts exactly where told; a video model is asked.
        """
        return True

    def generate_image(
        self, prompt: str, out_path: Path, reference: Path | None = None
    ) -> Path:
        """A still, derived from the prompt *and* from `reference` when given.

        Both inputs, not just the reference. Copying the anchor verbatim looks like
        a closer imitation of "stay consistent with it", and it breaks the pipeline:
        the opening frame of every cut becomes byte-identical, the concatenated
        video contains no detectable cuts, and QC reports a catastrophic pacing miss
        on a perfectly correct timeline. That is not a mock artifact. It is the real
        failure mode of an anchor applied too literally, and it is why each request's
        frame is *derived* from the anchor rather than being it.

        Mixing the reference into the digest is how the offline renderer shows that
        inheritance: same anchor with different prompts gives different frames, and
        the same prompt under a different anchor gives a different frame too.
        """
        seed = prompt
        if reference is not None and reference.is_file():
            seed = f"{hashlib.sha1(reference.read_bytes()).hexdigest()}:{prompt}"
        return self._still(seed, out_path)

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

    def shot_billed_duration(self, seconds: float) -> float:
        """Whole seconds, minimum one -- the same grid the real endpoints impose.

        ffmpeg renders any length, so offline this could be the identity. It is not,
        for the third time in this class and the same reason as the prompt limit and
        the shot cap above: a constraint the offline provider does not declare is a
        constraint nothing exercises until a paid run.

        This one matters more than the others because the storyboard now *plans*
        against it. A reference cut of 0.76s cannot be rendered, so the storyboard
        asks for 1s and the pacing moves away from the reference by 31%. That is a
        real cost of a real limit, and QC has to be able to see it -- with the
        identity here it would score a timeline no endpoint can deliver.
        """
        return float(max(1, math.floor(seconds + 0.5)))

    @property
    def max_shots_per_request(self) -> int:
        # Same reasoning again, and this one was missing: ffmpeg will concatenate
        # any number of cuts, so with no cap declared here a 14-cut storyboard
        # rendered offline as a single window and passed. The real endpoint accepts
        # six, so the first place that arrangement would have been rejected was a
        # paid request.
        return 6

    @property
    def max_shot_prompt_chars(self) -> int:
        # Same reasoning as the window above: ffmpeg does not care how long a
        # prompt is, but if the offline provider declared no limit then the
        # budgeting that keeps storyboard entries under Kling's 512 would never
        # run in a test, and the first time it ran would be against a paid API.
        return 512

    def generate_sequence(
        self,
        shots: list[MotionShot],
        out_path: Path,
        first_frame: Path | None = None,
    ) -> Path:
        """Build one clip whose shots are visually distinct.

        Each cut after the first derives its own frame from its own prompt. If they
        all looked alike the output would contain no detectable cuts at all, and the
        QC drift check would report a catastrophic miss on a perfectly correct
        timeline. `first_frame` therefore opens the clip and does not replace every
        cut in it -- which is also what the real endpoint does with it.
        """
        scratch = out_path.parent / f"{out_path.stem}_parts"
        # The planned durations, not the requested ones. A real endpoint quantises
        # the list and lifts a short one to its floor, so a mock that rendered the
        # exact request would hide every consequence of that arithmetic -- caption
        # placement and the total length both follow the plan, and offline is the
        # only place either can be checked.
        planned = self.plan_shot_durations([s.duration for s in shots])
        parts = []
        for i, (shot, seconds) in enumerate(zip(shots, planned, strict=True)):
            still = (
                first_frame
                if i == 0 and first_frame is not None
                else self._still(shot.prompt, scratch / f"{i:02d}.jpg")
            )
            parts.append(
                push_in(still, seconds, scratch / f"{i:02d}.mp4", self.settings)
            )
        return concat(parts, out_path)


# --------------------------------------------------------------------------- #


def build_video_provider(settings: Settings) -> BaseVideoProvider:
    if settings.resolved_video_provider() == "kling":
        # Imported here rather than at module scope so `mock` stays importable
        # even if the kling module grows a dependency later.
        from .kling import KlingVideoProvider

        return KlingVideoProvider(settings)
    return MockVideoProvider(settings)
