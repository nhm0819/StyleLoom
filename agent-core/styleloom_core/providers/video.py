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

    @property
    def max_shot_prompt_chars(self) -> int:
        # Same reasoning as the window above: ffmpeg does not care how long a
        # prompt is, but if the offline provider declared no limit then the
        # budgeting that keeps storyboard entries under Kling's 512 would never
        # run in a test, and the first time it ran would be against a paid API.
        return 512

    def generate(self, prompt: str, duration: float, out_path: Path) -> Path:
        scratch = out_path.parent / f"{out_path.stem}_still.jpg"
        return push_in(
            self._still(prompt, scratch), duration, out_path, self.settings
        )

    def generate_sequence(self, shots: list[MotionShot], out_path: Path) -> Path:
        """Build one clip whose shots are visually distinct.

        Each cut derives its own frame from its own prompt. If they all looked
        alike the output would contain no detectable cuts at all, and the QC drift
        check would report a catastrophic miss on a perfectly correct timeline.
        """
        scratch = out_path.parent / f"{out_path.stem}_parts"
        parts = [
            push_in(
                self._still(shot.prompt, scratch / f"{i:02d}.jpg"),
                shot.duration,
                scratch / f"{i:02d}.mp4",
                self.settings,
            )
            for i, shot in enumerate(shots)
        ]
        return concat(parts, out_path)


# --------------------------------------------------------------------------- #


def build_video_provider(settings: Settings) -> BaseVideoProvider:
    if settings.resolved_video_provider() == "kling":
        # Imported here rather than at module scope so `mock` stays importable
        # even if the kling module grows a dependency later.
        from .kling import KlingVideoProvider

        return KlingVideoProvider(settings)
    return MockVideoProvider(settings)
