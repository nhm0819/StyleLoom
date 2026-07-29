"""Provider interfaces.

Two-method interfaces, deliberately. The frontier video models turn over every
few months -- several of the current leaders did not exist a year ago. Hardcoding
one gives the harness a shelf life; keeping the choice in settings means someone
else can point this at whatever is best when they read it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

Task = str  # "ingest" | "outline" | "hook_candidates" | "style_synthesis"


@dataclass(frozen=True)
class MotionShot:
    """One cut inside a multi-shot request.

    Deliberately not `schema.Shot`: providers should not depend on the storyboard
    contract, only on prompt-and-duration pairs.
    """

    prompt: str
    duration: float


class BaseLLM:
    name = "base"

    def complete_json(
        self,
        task: Task,
        system: str,
        user: str,
        temperature: float = 0.7,
        images: list[bytes] | None = None,
    ) -> dict[str, Any]:
        raise NotImplementedError


class BaseVideoProvider:
    name = "base"

    @property
    def min_clip_sec(self) -> float:
        """Shortest clip this provider can produce.

        Real image-to-video endpoints have a floor (3-4s) far above a typical
        short-form shot, so callers request the floor and trim down.
        """
        return 0.0

    @property
    def max_concurrency(self) -> int:
        """Upper bound on parallel renders this provider tolerates."""
        return 32

    @property
    def supports_persona(self) -> bool:
        """Whether `persona_ref` is actually consumed.

        Declared rather than assumed so callers do not pay to generate a reference
        portrait that the endpoint will ignore.
        """
        return False

    @property
    def supports_multi_shot(self) -> bool:
        """Whether several cuts can be requested in one generation.

        This is the only escape from the per-shot duration floor: a request that
        carries its own per-cut durations bills the delivered length instead of
        rounding every 1.2s cut up to a 3-4s minimum.
        """
        return False

    @property
    def max_shot_window_sec(self) -> float:
        """Longest single multi-shot generation. Callers split across windows."""
        return 0.0

    @property
    def max_shots_per_request(self) -> int:
        """Most cuts one multi-shot generation accepts. 0 means no stated limit.

        Separate from `max_shot_window_sec` because the two limits bind
        independently: fourteen 0.76s cuts total under 11s and still exceed a
        six-shot cap.
        """
        return 0

    def shot_billed_duration(self, seconds: float) -> float:
        """How long one cut inside a multi-shot request actually runs.

        Declared because the caller has to do arithmetic with it. Endpoints
        quantise per-cut durations -- Kling's are integers with a floor of 1s --
        and a window packed against `max_shot_window_sec` using the durations we
        asked for will overflow the real limit once they are rounded. Four 3.6s
        cuts request 14.4s and deliver 16s, and the endpoint rejects the request
        rather than truncating it.

        Identity by default: a provider that honours the duration it is given has
        nothing to declare.
        """
        return seconds

    def animate_sequence(
        self,
        image_path: Path,
        shots: list[MotionShot],
        out_path: Path,
        persona_ref: Path | None = None,
    ) -> Path:
        """Render several cuts as one clip.

        Returns a single file containing every shot in order. The cuts land inside
        the returned video rather than at file boundaries, so the caller keeps the
        requested timeline and QC checks whether the model honoured it.
        """
        raise NotImplementedError

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
