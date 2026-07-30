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


class LookDetail(BaseModel):
    """The reference's visual vocabulary, split by which model can act on it.

    `look.keywords` was one flat list and that is why it went nowhere useful. It
    mixed things a still image can be told (`macro product texture`) with things
    only a video can express (`sfx-driven rhythm`) and things that actively fight
    the stage they were sent to -- `no talking head` in a prompt whose whole job is
    generating a portrait of the presenter.

    Split, each field can go to the stage that can use it. `keywords` stays as the
    fallback for styles extracted before this existed, and as somewhere for phrases
    that fit none of these.

    Fields are single phrases rather than lists: each one answers one question, and
    a phrase is easier to budget and to write into a prompt than a list to join.
    """

    # For image generation. These are what stop a frame from looking like stock.
    subject: str = ""    # what is on screen and how it is framed
    lighting: str = ""   # direction, hardness, colour of the light
    texture: str = ""    # surface qualities: skin, product, fabric, moisture
    palette: str = ""    # the actual colours, not a measurement of them
    lens: str = ""       # depth of field, focal feel, camera-as-device look

    # For video generation only. A still cannot express either.
    motion_feel: str = ""  # what drives the cutting and the movement

    def image_phrases(self) -> list[str]:
        """The fields a still-image prompt can act on, in prompt order."""
        return [p for p in (self.subject, self.lighting, self.texture,
                            self.palette, self.lens) if p.strip()]

    def video_phrases(self) -> list[str]:
        """What a shot prompt's *look* clause adds on the video side: nothing.

        `motion_feel` is the one video-side phrase, and `budget.motion_sentence`
        already puts it in the motion prompt. Returning it here too put it in both
        halves of the same request -- `continuation_video_prompt` is
        `continuation_prompt + motion_prompt`, so the phrase was sent twice and the
        512-character budget paid for it twice.
        """
        return []

    def any_set(self) -> bool:
        return bool(self.image_phrases() or self.video_phrases())


class Look(BaseModel):
    grade: str = "neutral"
    saturation: float = Field(0.5, ge=0, le=1)
    contrast: float = Field(0.5, ge=0, le=1)
    warmth: float = Field(0.5, ge=0, le=1)
    # Kept for styles extracted before `detail` existed, and as the fallback the
    # prompt builders use when `detail` is empty. Not removed: re-extracting a style
    # costs an LLM call and a hand-tuned style would lose its corrections.
    keywords: list[str] = Field(default_factory=list)
    detail: LookDetail = Field(default_factory=LookDetail)


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
    # What the pool said before this run specialised it against the brief. `id`
    # stays the pool entry's, because recency and the pool listing are keyed on it
    # -- so without this the drawn seed is unrecoverable from casting.json and
    # there is no way to see whether specialisation helped or drifted.
    seed_prompt: str = ""


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
    entropy_source: str = ""


class Beat(BaseModel):
    """One body beat.

    `content` and `caption` say the same thing to two different consumers under two
    very different budgets: `content` is the visual description that goes into a
    shot prompt (hundreds of characters), `caption` is what is burned on screen
    (tens). One string serving both is either a thin prompt or a truncated caption,
    so the generating stage is asked for each separately. `caption` empty means the
    model did not supply one and `storyboard` falls back to `content`.
    """

    name: str
    intent: str
    content: str
    caption: str = ""
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
    # Two fields though one string is sent: what the shot looks like, and what
    # moves in it. Fused, a storyboard is much harder to debug.
    scene_prompt: str
    motion_prompt: str
    # For an entry that is not first in its multi-shot request: identity and
    # location dropped, since the entry above it in the same generation gave them.
    continuation_prompt: str = ""

    @property
    def video_prompt(self) -> str:
        """Look first, then movement: the endpoint weights early tokens more."""
        return f"{self.scene_prompt} {self.motion_prompt}".strip()

    @property
    def continuation_video_prompt(self) -> str:
        """What a non-leading storyboard entry carries. Falls back to the full
        form, so a Shot built without one still renders."""
        if not self.continuation_prompt:
            return self.video_prompt
        return f"{self.continuation_prompt} {self.motion_prompt}".strip()


class Storyboard(BaseModel):
    style_id: str
    hook_text: str
    hook_archetype: str
    shots: list[Shot]

    @property
    def total_duration(self) -> float:
        return sum(s.duration_sec for s in self.shots)


class Keyframes(BaseModel):
    """The stills that open each render request.

    `anchor` is generated once per run from the cast creator and setting. Every
    frame in `frames` is then generated *with the anchor as a reference*, which is
    the whole design: one identity, many compositions. The previous revision of
    this project generated a frame per cut from that cut's own text and got neither.

    `frames` is keyed by the index of the shot that *leads* a request -- the first
    shot of each window. Render looks up its lead and passes the frame down; a
    missing key means that request falls back to text-to-video rather than failing.
    """

    anchor: Path | None = None
    frames: dict[int, Path] = Field(default_factory=dict)
    # Which model produced them, recorded because a run's cost and its look both
    # depend on it and neither is recoverable from the JPEG afterwards.
    model: str = ""

    def for_lead(self, shot_index: int) -> Path | None:
        return self.frames.get(shot_index)


class ClipSegment(BaseModel):
    """One rendered file and the shots it contains, in order.

    A segment exists because a clip is not one shot. One file carries a window of
    cuts and the boundaries are timestamps inside it, so everything downstream reads
    segments and shot indices rather than assuming a file per cut.
    """

    path: Path
    shot_indices: list[int]
    # The storyboard's intent, which the QC drift check measures against.
    requested_durations: list[float] = Field(default_factory=list)
    # What the endpoint was asked to run after quantisation. Kling's per-cut
    # durations are whole seconds, so a 0.6s cut lands at 1s and captions must be
    # placed against these. Empty means identical to the requested figures, which is
    # what a provider with no quantisation of its own reports.
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
    # The plan this run was actually given -- not `default_plan()` recomputed later,
    # since settings (use_first_frame, --no-qc) can change between when a run failed
    # and when someone resumes it. Empty means the run predates resume support.
    plan_steps: list[str] = Field(default_factory=list)

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
