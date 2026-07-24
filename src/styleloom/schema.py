"""StyleLoom data contracts.

Every pipeline stage is a pure `JSON in -> JSON out` function. These models are
the contract between stages, and each one is persisted to disk so a run can be
inspected, resumed, or graded stage by stage.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


ShotSize = Literal["ECU", "CU", "MS", "WS", "EWS"]


# --------------------------------------------------------------------------- #
# 1. Style DNA  (produced by pipeline.analyze, reusable asset)
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
    font: str = "bold_sans"
    pos: Literal["top", "center", "center_lower", "bottom"] = "center_lower"
    anim: str = "pop_in"
    max_chars_per_line: int = 14
    color: str = "white"
    stroke_color: str = "black"
    appear_on_beat: bool = True


class AudioStyle(BaseModel):
    bpm: float = 120.0
    sfx_on_cut: bool = True
    voice_tone: str = "neutral"


class HookStyle(BaseModel):
    """How the reference *formats* its first 3 seconds. Content is never stored
    here -- content is generated fresh per run by pipeline.hook."""

    window_sec: float = 3.0
    shot_size: ShotSize = "CU"
    cut_count: int = 2
    text_patterns: list[str] = Field(default_factory=lambda: ["question"])


class StyleSchema(BaseModel):
    style_id: str
    created_at: str = Field(default_factory=_now)
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
# 2. Brief  (produced by pipeline.ingest)
# --------------------------------------------------------------------------- #


class Brief(BaseModel):
    """Normalised user input. text / image / video all collapse to this."""

    input_kind: Literal["text", "image", "video"] = "text"
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


class Beat(BaseModel):
    name: str
    intent: str
    content: str
    duration_sec: float = 3.0


class Outline(BaseModel):
    """Body beats only. The hook is generated afterwards so it can reference the
    payoff -- this is what makes the hook fit the whole context."""

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
    created_at: str = Field(default_factory=_now)
    updated_at: str = Field(default_factory=_now)
    error: str | None = None
    artifacts: dict[str, str] = Field(default_factory=dict)
    hook_text: str | None = None
    hook_archetype: str | None = None
    qc_score: float | None = None

    def touch(self, **kw: Any) -> "RunRecord":
        for k, v in kw.items():
            setattr(self, k, v)
        self.updated_at = _now()
        return self
