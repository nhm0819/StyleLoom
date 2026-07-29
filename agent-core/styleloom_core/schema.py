"""StyleLoom data contracts.

Every tool is a `JSON in -> JSON out` function and every model here is persisted
to disk, so a run can be inspected, resumed, or graded stage by stage.

These models are the stable surface of `styleloom_core`. The CLI, and later the
API and worker, speak in these types and nothing else.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field


def utcnow() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


ShotSize = Literal["ECU", "CU", "MS", "WS", "EWS"]
InputKind = Literal["text", "image", "video"]


# --------------------------------------------------------------------------- #
# 1. Style DNA  (produced by tools.analyze, reusable asset)
# --------------------------------------------------------------------------- #


class Pacing(BaseModel):
    avg_shot_sec: float = Field(1.5, gt=0)
    shot_count: int = Field(10, gt=0)
    cut_style: str = "jump_cut"
    cut_times: list[float] = Field(default_factory=list)


class Camera(BaseModel):
    shot_size_dist: dict[str, float] = Field(
        default_factory=lambda: {"CU": 0.5, "MS": 0.35, "WS": 0.15}
    )
    moves: list[str] = Field(default_factory=lambda: ["handheld_micro_shake"])


class Look(BaseModel):
    grade: str = "neutral"
    saturation: float = Field(0.5, ge=0, le=1)
    contrast: float = Field(0.5, ge=0, le=1)
    warmth: float = Field(0.5, ge=0, le=1)
    keywords: list[str] = Field(default_factory=list)


class CaptionStyle(BaseModel):
    pos: Literal["top", "center", "center_lower", "bottom"] = "center_lower"
    # Only "pop_in" is honoured, as a 0.12s alpha ramp in media.burn_captions.
    anim: str = "pop_in"
    max_chars_per_line: int = 14
    color: str = "white"
    stroke_color: str = "black"


class AudioStyle(BaseModel):
    bpm: float = 120.0
    voice_tone: str = "neutral"


class HookStyle(BaseModel):
    """How the reference *formats* its first 3 seconds.

    Hook content is deliberately never stored here. Content is generated fresh
    per run by tools.hook; only the format is a property of the style.
    """

    window_sec: float = 3.0
    shot_size: ShotSize = "CU"
    cut_count: int = 2


class StyleSchema(BaseModel):
    style_id: str
    created_at: str = Field(default_factory=utcnow)
    source_refs: list[str] = Field(default_factory=list)
    total_duration: float = 30.0
    beat_pattern: list[str] = Field(
        default_factory=lambda: ["hook", "context", "turn", "payoff", "cta"]
    )
    pacing: Pacing = Field(default_factory=Pacing)
    camera: Camera = Field(default_factory=Camera)
    look: Look = Field(default_factory=Look)
    caption: CaptionStyle = Field(default_factory=CaptionStyle)
    audio: AudioStyle = Field(default_factory=AudioStyle)
    hook_style: HookStyle = Field(default_factory=HookStyle)
    notes: str = ""


# --------------------------------------------------------------------------- #
# 2. Run inputs + Brief
# --------------------------------------------------------------------------- #


class RunInputs(BaseModel):
    """What the caller supplied for this run.

    Distinct from `Brief`: these are raw, unnormalised inputs and side assets.
    They live on the session rather than being a tool artifact, because no tool
    produces them.
    """

    text: str = ""
    file_path: Path | None = None
    bgm: Path | None = None
    persona_ref: Path | None = None
    language: str = "ko"


class Brief(BaseModel):
    """Normalised user input. text / image / video all collapse to this."""

    input_kind: InputKind = "text"
    raw_text: str = ""
    source_path: str | None = None
    topic: str = ""
    audience: str = "general"
    key_message: str = ""
    facts: list[str] = Field(default_factory=list)
    language: str = "ko"


# --------------------------------------------------------------------------- #
# 3. Outline + Hook
# --------------------------------------------------------------------------- #


class CastChoice(BaseModel):
    id: str
    label: str = ""
    prompt: str = ""


class Casting(BaseModel):
    """Who is on screen and where.

    Sampled per run so three videos from one style are three different videos
    rather than the same footage with new captions.
    """

    creator: CastChoice
    setting: CastChoice
    creator_pool: list[str] = Field(default_factory=list)
    setting_pool: list[str] = Field(default_factory=list)
    # A generated portrait, reused as the reference image for every shot so the
    # presenter stays the same person across cuts. Only produced when the video
    # provider can actually consume a reference image.
    creator_ref: Path | None = None
    entropy_source: str = ""


class Beat(BaseModel):
    name: str
    intent: str
    content: str
    duration_sec: float = 3.0


class Outline(BaseModel):
    """Body beats only.

    The hook is generated afterwards so it can reference the payoff. That
    ordering is what makes the hook fit this specific video instead of being a
    generic attention grabber.
    """

    beats: list[Beat]
    payoff: str = ""


class HookCandidate(BaseModel):
    archetype: str
    text: str
    visual: str
    score: float = 0.0
    rationale: str = ""


class HookResult(BaseModel):
    """Full audit trail of the non-deterministic hook decision."""

    archetype_pool: list[str]
    archetype_sampled: str
    candidates: list[HookCandidate]
    selected: HookCandidate
    selection_method: str
    temperature: float
    entropy_source: str


# --------------------------------------------------------------------------- #
# 4. Storyboard
# --------------------------------------------------------------------------- #


class Shot(BaseModel):
    index: int
    role: str  # hook | context | turn | payoff | cta
    duration_sec: float
    shot_size: ShotSize
    camera_move: str
    action: str
    caption: str
    image_prompt: str  # -> t2i keyframe
    motion_prompt: str  # -> i2v motion


class Storyboard(BaseModel):
    style_id: str
    hook_text: str
    hook_archetype: str
    shots: list[Shot]

    @property
    def total_duration(self) -> float:
        return sum(s.duration_sec for s in self.shots)


class ClipSegment(BaseModel):
    """One rendered file and the shots it contains, in order.

    A segment exists because a clip is no longer always one shot. In `per_shot`
    mode every segment holds exactly one; in `multi_shot` mode one file carries
    several cuts and the boundaries are timestamps inside it. Downstream code
    reads segments and does not care which mode produced them.
    """

    path: Path
    shot_indices: list[int]
    # The storyboard's intent. In multi_shot mode the delivered cuts may differ,
    # which is what the QC drift check measures -- so the request has to be
    # recorded separately from what the endpoint was actually asked to run.
    requested_durations: list[float] = Field(default_factory=list)
    # What the endpoint was actually asked to run, after it quantised the request.
    # Kling's per-cut durations are whole seconds, so a 0.6s cut is asked for as
    # 1s and the cut inside the returned clip lands at 1s. Captions have to be
    # placed against these or they drift away from the cuts they belong to, a
    # little further with every shot.
    #
    # Empty means identical to `requested_durations`, which is the per_shot case:
    # each clip is trimmed to the exact cut length before it gets here.
    billed_durations: list[float] = Field(default_factory=list)

    @property
    def is_multi_shot(self) -> bool:
        return len(self.shot_indices) > 1

    @property
    def caption_durations(self) -> list[float]:
        """Where captions go: the timeline the endpoint was given, not the ideal."""
        return self.billed_durations or self.requested_durations


class RenderResult(BaseModel):
    """Output of the render tool. Paths, not bytes, so the artifact stays
    readable and a failed run shows exactly which shots survived."""

    mode: Literal["per_shot", "multi_shot"] = "per_shot"
    segments: list[ClipSegment] = Field(default_factory=list)
    errors: dict[int, str] = Field(default_factory=dict)

    @property
    def clips(self) -> list[Path]:
        return [seg.path for seg in self.segments]

    def cut_timeline(self) -> list[float]:
        """Requested cut times across the whole video, from its start.

        Boundaries between segments are cuts too, so this runs the durations
        end to end rather than per segment.
        """
        times: list[float] = []
        elapsed = 0.0
        for seg in self.segments:
            for duration in seg.requested_durations:
                elapsed += duration
                times.append(round(elapsed, 3))
        return times[:-1]  # the final boundary is the end of the video, not a cut


class AssembleResult(BaseModel):
    final_video: Path
    captioned_count: int = 0


# --------------------------------------------------------------------------- #
# 5. QC + Run record
# --------------------------------------------------------------------------- #


class QCCheck(BaseModel):
    name: str
    target: float
    actual: float
    tolerance: float
    passed: bool


class QCReport(BaseModel):
    style_id: str
    checks: list[QCCheck]
    score: float
    passed: bool


class RunStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


class RunRecord(BaseModel):
    run_id: str
    style_id: str
    status: RunStatus = RunStatus.QUEUED
    stage: str = "queued"
    created_at: str = Field(default_factory=utcnow)
    updated_at: str = Field(default_factory=utcnow)
    error: str | None = None
    artifacts: dict[str, str] = Field(default_factory=dict)
    hook_text: str | None = None
    hook_archetype: str | None = None
    qc_score: float | None = None

    def touch(self, **kw: Any) -> RunRecord:
        for k, v in kw.items():
            setattr(self, k, v)
        self.updated_at = utcnow()
        return self


class Choice(BaseModel):
    """One line of a style's decision history. Append-only; see memory.history.

    `kind` distinguishes hook archetypes from casting choices so a single log
    serves every pool that needs a recency penalty.
    """

    run_id: str
    kind: Literal["hook", "creator", "setting"]
    value: str
    detail: str = ""
    at: str = Field(default_factory=utcnow)
