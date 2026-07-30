"""Provider interfaces.

Two-method interfaces, deliberately. The frontier video models turn over every
few months -- several of the current leaders did not exist a year ago. Hardcoding
one gives the harness a shelf life; keeping the choice in settings means someone
else can point this at whatever is best when they read it.

Two generating methods, because consistency needs a stage that video generation
does not have on its own. One request holds identity *within* its own generation and
cannot hold it between requests; a first frame carried into every request holds it
across all of them. `generate_image` produces that frame and `generate_sequence`
accepts it.

There is no single-cut method. Every render request carries a shot list, and a list
of one shot is a request with one shot in it -- a second entry point for that case
would be a second place for the payload shape to drift.

`first_frame` is optional, so a provider that cannot take one still works and a run
configured without one still renders.
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

    @property
    def max_shot_prompt_chars(self) -> int:
        """Longest prompt one storyboard entry takes. Far tighter than the
        top-level prompt -- 512 against a few thousand on Kling. 0 means none."""
        return 0

    @property
    def supports_first_frame(self) -> bool:
        """Whether a generated still can be carried into a render as its opening
        frame, and whether `generate_image` is therefore worth calling."""
        return False

    def shot_billed_duration(self, seconds: float) -> float:
        """How long one cut inside a multi-shot request actually runs.

        Windows are packed against this, not against the requested durations:
        endpoints quantise, so four 3.6s cuts request 14.4s and deliver 16s, and
        the request is rejected whole rather than truncated.
        """
        return seconds

    def plan_shot_durations(self, durations: list[float]) -> list[float]:
        """What the endpoint will actually run for each cut in one request.

        Separate from `shot_billed_duration` because the binding constraint is on
        the list, not on any one cut: an endpoint whose request-level duration must
        equal the sum of its shots, and which has a floor of its own, cannot honour
        a list that quantises to less than that floor -- so a short list is lifted
        as a whole rather than cut by cut. Callers place captions against the result
        and trim against it, so it has to be the endpoint's arithmetic and not a
        second copy of it.
        """
        return [self.shot_billed_duration(d) for d in durations]

    def generate_image(
        self, prompt: str, out_path: Path, reference: Path | None = None
    ) -> Path:
        """One still.

        `reference` is another image the result should stay consistent with, which
        is the whole reason this method exists rather than each frame being an
        independent roll. Only called when `supports_first_frame`.
        """
        raise NotImplementedError

    def generate_sequence(
        self,
        shots: list[MotionShot],
        out_path: Path,
        first_frame: Path | None = None,
    ) -> Path:
        """Several cuts in one clip, each with its own prompt and duration.

        Only called when `supports_multi_shot`.
        """
        raise NotImplementedError
