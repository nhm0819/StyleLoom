"""Runtime configuration.

Env-driven so a different person can point the harness at their own models
without touching code.

Deliberately *not* a module-level singleton. The previous design instantiated
settings at import time, which meant tests had to set environment variables
before importing the package, and meant a server could never serve two
configurations. Callers construct `Settings` and hand it to `Context`.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from .errors import ConfigError

REPO_ROOT = Path(__file__).resolve().parents[2]

LLM_PROVIDERS = ("auto", "mock", "anthropic")
VIDEO_PROVIDERS = ("auto", "mock", "kling")
RENDER_MODES = ("per_shot", "multi_shot")
KLING_MODES = ("std", "pro")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="STYLELOOM_",
        env_file=".env",
        extra="ignore",
        # Required because the key fields carry a validation_alias. Without it,
        # pydantic accepts *only* the alias, so `Settings(kling_secret_key="x")` silently
        # yields "" -- no exception, just a wrong value. Every programmatic caller
        # and every test that injects a key directly depends on this.
        populate_by_name=True,
    )

    data_dir: Path = Path("data")
    archetypes_path: Path = Path("configs/archetypes.yaml")
    kling_models_path: Path = Path("configs/kling_models.yaml")
    casting_path: Path = Path("configs/casting.yaml")

    # --- LLM (planning, hook generation, style synthesis) -------------------
    # `auto` uses the real provider when its key is present and falls back to the
    # offline mock otherwise. This is the default because the most likely failure
    # for someone running this with their own keys is injecting the key and
    # getting placeholder output because a second variable was still on `mock`.
    llm_provider: str = "auto"  # auto | mock | anthropic
    llm_model: str = "claude-sonnet-5"

    # Accepts the conventional unprefixed name too. Anyone injecting credentials
    # into a container will set ANTHROPIC_API_KEY, not our namespaced version, and
    # silently reading an empty string would be the worst possible outcome.
    anthropic_api_key: str = Field(
        default="",
        validation_alias=AliasChoices(
            "STYLELOOM_ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY"
        ),
    )

    # --- Video (keyframe + image-to-video) ---------------------------------
    video_provider: str = "auto"  # auto | mock | kling
    # Two halves of one credential: the access key is public and travels in the
    # token, the secret key never leaves the process and signs it.
    kling_access_key: str = Field(
        default="",
        validation_alias=AliasChoices("STYLELOOM_KLING_ACCESS_KEY", "KLING_ACCESS_KEY"),
    )
    kling_secret_key: str = Field(
        default="",
        validation_alias=AliasChoices("STYLELOOM_KLING_SECRET_KEY", "KLING_SECRET_KEY"),
    )
    # api-singapore is the international account system. api-beijing.klingai.com
    # is the mainland one -- a separate account, not a region of the same one, so
    # switching hosts also means switching credentials.
    kling_base_url: str = "https://api-singapore.klingai.com"
    # `model_name` values, which must be keys in configs/kling_models.yaml -- the
    # provider validates both on construction rather than at the first request.
    #
    # One model, one call per cut: this is a text-to-video pipeline with no
    # keyframe stage. The Omni variant (kling-v3-omni) is the reference-driven
    # tier, but it lives on a different path and its advantage is elements, which
    # are not wired up yet.
    kling_t2v_model: str = "kling-v3"

    # The quality tier, which on the official API is a request field rather than
    # part of the endpoint. `std` is the default because the default is what an
    # evaluation run costs; `pro` is the same parameters at a higher rate.
    #
    # It is also a resolution choice: std renders 720p and pro 1080p. That pairs
    # with the 720x1280 output below, so the default combination generates at the
    # size it delivers. Raising width/height to 1080x1920 without also switching
    # to `pro` upscales a 720p render, which `styleloom doctor` flags.
    kling_mode: str = "std"  # std | pro

    # Generation is asynchronous, so these two bound a wait rather than a request.
    # The timeout is generous because it covers queue time as well as inference:
    # a 15s multi-shot job behind a busy queue is minutes, and giving up early
    # abandons a task that has already been billed.
    kling_timeout_sec: float = 900.0
    kling_poll_interval_sec: float = 5.0

    # Not spec data, unlike the fal registry it replaces: Kling's concurrency
    # ceiling is a property of the account tier, so the same endpoint allows more
    # parallel tasks on a larger plan and no per-model value could be correct.
    kling_max_concurrency: int = 1

    # --- Output ------------------------------------------------------------
    # 9:16 at 720p30, which is the short-form baseline every vertical platform
    # accepts without re-encoding. Portrait is the default rather than a flag
    # because this system exists to reproduce short-form references; landscape
    # still works (the Kling provider maps any size onto the nearest ratio the
    # API offers), it is just not what the defaults are tuned for.
    #
    # 30fps rather than 60: the reference set is 30, the video model delivers 30,
    # and a 60fps container holding 30fps content only doubles the file.
    width: int = 720
    height: int = 1280
    fps: int = 30

    # --- Hook non-determinism ----------------------------------------------
    hook_candidate_count: int = 5
    hook_temperature: float = 0.9
    hook_top_k: int = 3
    hook_softmax_temp: float = 0.8
    hook_recency_window: int = 4
    hook_recency_penalty: float = 0.35

    # Upper bound only. Providers with a lower documented limit clamp it down.
    max_concurrent_renders: int = 3

    # --- render strategy -----------------------------------------------------
    # per_shot   one generation per cut, trimmed to length with ffmpeg. Cuts land
    #            on file boundaries, so shot durations are exact by construction.
    #            Pays the endpoint's duration floor on every cut.
    # multi_shot one generation carrying several cuts and their durations. Removes
    #            the floor entirely -- roughly half the cost at short-form pacing --
    #            but the cuts are inside the model's output, so whether the
    #            requested timeline was honoured becomes a measurement rather than
    #            a guarantee. QC reports the drift.
    #
    # Defaults to per_shot: it is the verified path, and multi_shot trades a known
    # cost for pacing that depends on the endpoint. Opt in deliberately.
    render_mode: str = "per_shot"  # per_shot | multi_shot

    # --- provider resolution ------------------------------------------------

    def resolved_llm_provider(self) -> str:
        """Concrete provider name after resolving `auto`."""
        if self.llm_provider != "auto":
            return self.llm_provider
        return "anthropic" if self.anthropic_api_key else "mock"

    def resolved_video_provider(self) -> str:
        if self.video_provider != "auto":
            return self.video_provider
        return "kling" if self.kling_secret_key else "mock"

    def provider_summary(self) -> str:
        def fmt(requested: str, resolved: str) -> str:
            return resolved if requested == resolved else f"{resolved} (auto)"

        return (
            f"llm={fmt(self.llm_provider, self.resolved_llm_provider())} "
            f"video={fmt(self.video_provider, self.resolved_video_provider())}"
        )

    # --- derived paths ------------------------------------------------------

    @property
    def styles_dir(self) -> Path:
        return self.data_dir / "styles"

    @property
    def runs_dir(self) -> Path:
        return self.data_dir / "runs"

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    def ref_candidates(self, path: Path) -> list[Path]:
        """Where a reference video argument could live, in priority order.

        `uploads_dir` first so a bare filename works from any working directory --
        `data/uploads/` is the place references are meant to sit, and typing the
        full path to it every time is the friction this removes.

        The second candidate is the argument untouched, which is what makes an
        absolute path and a path relative to cwd keep working. Both are covered by
        the same two probes with no branch on the shape of the argument: joining an
        absolute path onto `uploads_dir` yields the absolute path itself, so for
        `/tmp/ref.mp4` the two candidates collapse to one.

        Deduplicated because the caller puts these in an error message, and
        "tried /tmp/ref.mp4, /tmp/ref.mp4" reads like a bug.
        """
        found: list[Path] = []
        for candidate in (self.uploads_dir / path, path):
            if candidate not in found:
                found.append(candidate)
        return found

    def resolve_ref(self, path: Path) -> Path | None:
        """First existing candidate, or None. Callers own the error message.

        Returns rather than raises because `extract_style` reports every missing
        reference at once -- raising here would surface them one run at a time.
        """
        return next((c for c in self.ref_candidates(path) if c.is_file()), None)

    def resolve_config(self, path: Path) -> Path:
        """Config files fall back to the ones bundled in the repo.

        An operator's own file wins; the bundled copy means a fresh clone runs
        without any setup.
        """
        if path.exists():
            return path
        bundled = REPO_ROOT / path
        if bundled.exists():
            return bundled
        raise ConfigError(f"config file not found: {path} (also tried {bundled})")

    def ensure_dirs(self) -> None:
        for d in (self.styles_dir, self.runs_dir, self.uploads_dir):
            d.mkdir(parents=True, exist_ok=True)

    def validate_runtime(self) -> None:
        """Fail before doing work, not at the first provider call."""
        if self.llm_provider not in LLM_PROVIDERS:
            raise ConfigError(
                f"unknown llm_provider: {self.llm_provider!r}. "
                f"Expected one of {LLM_PROVIDERS}."
            )
        if self.video_provider not in VIDEO_PROVIDERS:
            raise ConfigError(
                f"unknown video_provider: {self.video_provider!r}. "
                f"Expected one of {VIDEO_PROVIDERS}."
            )
        if self.render_mode not in RENDER_MODES:
            raise ConfigError(
                f"unknown render_mode: {self.render_mode!r}. "
                f"Expected one of {RENDER_MODES}."
            )
        # Checked here rather than left to the API: `mode` is the price tier, and
        # an unrecognised value is the kind of typo that is cheaper to catch now.
        if self.kling_mode not in KLING_MODES:
            raise ConfigError(
                f"unknown kling_mode: {self.kling_mode!r}. "
                f"Expected one of {KLING_MODES}."
            )
        if self.hook_candidate_count < 1:
            raise ConfigError("hook_candidate_count must be >= 1")
        if self.width < 16 or self.height < 16:
            raise ConfigError("width/height must be >= 16")
