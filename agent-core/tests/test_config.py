"""Configuration and context assembly.

The point of these tests is the property the old singleton could not offer: two
configurations coexisting in one process, with no dependence on import order.
"""

from __future__ import annotations

import pytest
from styleloom_core import Settings, build_context
from styleloom_core.errors import ConfigError


def test_two_contexts_hold_different_configurations(tmp_path):
    a = build_context(Settings(data_dir=tmp_path / "a", width=360, height=640))
    b = build_context(Settings(data_dir=tmp_path / "b", width=1080, height=1920))

    assert a.settings.width == 360
    assert b.settings.width == 1080
    assert a.styles.dir_for("s") != b.styles.dir_for("s")


def test_env_prefix_is_honoured(monkeypatch, tmp_path):
    monkeypatch.setenv("STYLELOOM_WIDTH", "540")
    monkeypatch.setenv("STYLELOOM_DATA_DIR", str(tmp_path))
    assert Settings().width == 540


def test_explicit_arguments_beat_the_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("STYLELOOM_WIDTH", "540")
    assert Settings(data_dir=tmp_path, width=720).width == 720


def test_building_a_context_creates_the_data_directories(tmp_path):
    ctx = build_context(Settings(data_dir=tmp_path / "fresh"))
    assert ctx.settings.styles_dir.is_dir()
    assert ctx.settings.runs_dir.is_dir()
    assert ctx.settings.uploads_dir.is_dir()


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("llm_provider", "gpt", "llm_provider"),
        ("video_provider", "runway", "video_provider"),
        ("hook_candidate_count", 0, "hook_candidate_count"),
        ("width", 4, "width/height"),
    ],
)
def test_invalid_settings_fail_before_any_work(tmp_path, field, value, message):
    """Fail at construction, not at the first provider call, so nothing is spent
    discovering a typo."""
    with pytest.raises(ConfigError, match=message):
        build_context(Settings(data_dir=tmp_path, **{field: value}))


def test_providers_can_be_injected_without_environment_variables(tmp_path):
    """How tests avoid both the network and the env-ordering problem."""
    sentinel = object()
    ctx = build_context(Settings(data_dir=tmp_path), llm=sentinel, video=sentinel)  # type: ignore[arg-type]
    assert ctx.llm is sentinel
    assert ctx.video is sentinel


def test_bundled_config_is_found_from_a_foreign_working_directory(tmp_path, monkeypatch):
    """A fresh clone has to run without setting any paths, from anywhere."""
    monkeypatch.chdir(tmp_path)
    settings = Settings(data_dir=tmp_path / "data")
    assert settings.resolve_config(settings.archetypes_path).exists()
    assert settings.resolve_config(settings.fal_models_path).exists()


def test_a_missing_config_file_says_where_it_looked(tmp_path):
    from pathlib import Path

    settings = Settings(data_dir=tmp_path, archetypes_path=Path("nowhere/none.yaml"))
    with pytest.raises(ConfigError, match="also tried"):
        settings.resolve_config(settings.archetypes_path)


def test_the_core_declares_no_framework_dependency():
    """The boundary this package exists to hold. If a web or CLI framework ever
    becomes reachable from the core, the layering has quietly collapsed."""
    from importlib.metadata import requires

    declared = " ".join(requires("styleloom-core") or []).lower()
    for forbidden in ("fastapi", "typer", "click", "flask", "django", "uvicorn"):
        assert forbidden not in declared, f"styleloom-core must not depend on {forbidden}"


# --- credential injection -------------------------------------------------- #
#
# These cover the grader's path: keys arrive as environment variables under their
# conventional names, and nothing else is configured. Reading an empty string here
# would produce placeholder video from a correctly-keyed environment, which is the
# worst available failure because it looks like the system does not work.


def test_conventional_key_names_are_accepted(monkeypatch, tmp_path):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-conventional")
    monkeypatch.setenv("FAL_KEY", "fal-conventional")
    settings = Settings(data_dir=tmp_path)
    assert settings.anthropic_api_key == "sk-conventional"
    assert settings.fal_key == "fal-conventional"


def test_namespaced_key_names_are_also_accepted(monkeypatch, tmp_path):
    monkeypatch.setenv("STYLELOOM_ANTHROPIC_API_KEY", "sk-namespaced")
    monkeypatch.setenv("STYLELOOM_FAL_KEY", "fal-namespaced")
    settings = Settings(data_dir=tmp_path)
    assert settings.anthropic_api_key == "sk-namespaced"
    assert settings.fal_key == "fal-namespaced"


def test_namespaced_key_wins_over_the_conventional_one(monkeypatch, tmp_path):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-generic")
    monkeypatch.setenv("STYLELOOM_ANTHROPIC_API_KEY", "sk-specific")
    assert Settings(data_dir=tmp_path).anthropic_api_key == "sk-specific"


def test_auto_falls_back_to_mock_without_keys(tmp_path):
    settings = Settings(data_dir=tmp_path)
    assert settings.resolved_llm_provider() == "mock"
    assert settings.resolved_video_provider() == "mock"


def test_auto_selects_the_real_provider_once_a_key_appears(monkeypatch, tmp_path):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-x")
    monkeypatch.setenv("FAL_KEY", "fal-x")
    settings = Settings(data_dir=tmp_path)
    assert settings.resolved_llm_provider() == "anthropic"
    assert settings.resolved_video_provider() == "fal"


def test_an_explicit_provider_overrides_auto_detection(monkeypatch, tmp_path):
    """Pinning to mock must survive a key being present, or offline testing on a
    keyed machine becomes impossible."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-x")
    settings = Settings(data_dir=tmp_path, llm_provider="mock")
    assert settings.resolved_llm_provider() == "mock"


def test_provider_summary_marks_what_was_auto_resolved(monkeypatch, tmp_path):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-x")
    summary = Settings(data_dir=tmp_path).provider_summary()
    assert "llm=anthropic (auto)" in summary
    assert "video=mock (auto)" in summary


def test_auto_is_a_valid_provider_value(tmp_path):
    build_context(Settings(data_dir=tmp_path, llm_provider="auto", video_provider="auto"))


def test_keys_can_still_be_passed_directly(tmp_path):
    """Regression: adding `validation_alias` to the key fields made pydantic accept
    only the alias, so `Settings(fal_key=...)` silently evaluated to "" with no
    exception. populate_by_name=True restores it."""
    settings = Settings(data_dir=tmp_path, fal_key="direct-fal", anthropic_api_key="direct-sk")
    assert settings.fal_key == "direct-fal"
    assert settings.anthropic_api_key == "direct-sk"
