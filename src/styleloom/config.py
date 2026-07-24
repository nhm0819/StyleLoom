"""Runtime configuration. Everything is env-driven so a different person can
point the harness at their own models without touching code."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="STYLELOOM_", env_file=".env", extra="ignore")

    data_dir: Path = Path("data")
    archetypes_path: Path = Path("configs/archetypes.yaml")
    fal_models_path: Path = Path("configs/fal_models.yaml")

    # --- LLM (planning, hook generation, style synthesis) -------------------
    llm_provider: str = "mock"  # mock | anthropic
    llm_model: str = "claude-sonnet-4-6"
    anthropic_api_key: str = ""

    # --- Video (keyframe + image-to-video) ---------------------------------
    video_provider: str = "mock"  # mock | fal
    fal_key: str = ""
    # Must be keys in configs/fal_models.yaml -- the provider validates on start.
    fal_t2i_model: str = "fal-ai/flux-2-flex"
    fal_i2v_model: str = "bytedance/seedance-2.0/fast/image-to-video"
    fal_timeout_sec: float = 600.0

    # --- Generation defaults ------------------------------------------------
    width: int = 720
    height: int = 1280
    fps: int = 30
    hook_candidate_count: int = 5
    hook_temperature: float = 0.9
    hook_top_k: int = 3
    hook_softmax_temp: float = 0.8
    max_concurrent_renders: int = 3

    @property
    def styles_dir(self) -> Path:
        return self.data_dir / "styles"

    @property
    def runs_dir(self) -> Path:
        return self.data_dir / "runs"

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    def ensure_dirs(self) -> None:
        for d in (self.styles_dir, self.runs_dir, self.uploads_dir):
            d.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_dirs()
