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


def build_video_provider(settings: Settings) -> BaseVideoProvider:
    if settings.resolved_video_provider() == "kling":
        # Imported here rather than at module scope so `mock` stays importable
        # even if the kling module grows a dependency later.
        from .kling import KlingVideoProvider

        return KlingVideoProvider(settings)
    return MockVideoProvider(settings)
