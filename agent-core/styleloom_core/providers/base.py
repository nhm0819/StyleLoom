"""Provider interfaces.

Two-method interfaces, deliberately. The frontier video models turn over every
few months -- several of the current leaders did not exist a year ago. Hardcoding
one gives the harness a shelf life; keeping the choice in settings means someone
else can point this at whatever is best when they read it.

Text-to-video only. An earlier revision generated a keyframe and animated it,
which cost two calls and two round trips per cut and bought less than it looked
like: every cut generated its own keyframe from its own text, so the person in
shot 2 was not the person in shot 1. A multi-shot text-to-video request produces
every cut in the window from one generation, which is where cross-cut consistency
actually comes from.
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

        Real endpoints have a floor (3s on Kling v3) far above a typical
        short-form shot, so callers request the floor and trim down.
        """
        return 0.0

    @property
    def max_concurrency(self) -> int:
        """Upper bound on parallel renders this provider tolerates."""
        return 32

    @property
    def supports_multi_shot(self) -> bool:
        """Whether one request can carry several cuts and their durations."""
        return False

    @property
    def max_shot_window_sec(self) -> float:
        """Longest total duration one multi-shot request accepts."""
        return 0.0

    @property
    def max_shots_per_request(self) -> int:
        """Most cuts one multi-shot request accepts. 0 means no stated limit."""
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

    def generate(self, prompt: str, duration: float, out_path: Path) -> Path:
        """One cut, from text alone."""
        raise NotImplementedError

    def generate_sequence(self, shots: list[MotionShot], out_path: Path) -> Path:
        """Several cuts in one clip, each with its own prompt and duration.

        Only called when `supports_multi_shot`.
        """
        raise NotImplementedError
