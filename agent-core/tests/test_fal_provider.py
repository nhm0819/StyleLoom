"""fal endpoint payloads.

These assert the payload *shape* per endpoint, which is the part that breaks
silently. Seedance takes `image_url`, Kling takes `start_image_url`; `duration` is
a string on both; Kling infers aspect ratio from the start image, so sending one
is wrong; and Kling's per-user concurrency is 1, so exceeding it fails a run.

No network and no key: the SDK is stubbed and only payload construction is
exercised, which is why `build_animate_payload` is separated from the call.
"""

from __future__ import annotations

import sys
import types

import pytest
from styleloom_core import Settings
from styleloom_core.errors import VideoProviderError
from styleloom_core.providers.video import FalVideoProvider, load_fal_specs

SEEDANCE = "bytedance/seedance-2.0/fast/image-to-video"
KLING_PRO = "fal-ai/kling-video/v3/pro/image-to-video"
VEO2 = "fal-ai/veo2/image-to-video"
FLUX = "fal-ai/flux-2-flex"


@pytest.fixture
def stub_fal(monkeypatch):
    """A stand-in for the fal_client module."""
    module = types.ModuleType("fal_client")
    module.subscribe = lambda *a, **kw: {}  # type: ignore[attr-defined]
    module.upload_file = lambda path: f"https://cdn.test/{path.name}"  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "fal_client", module)
    return module


def make_provider(stub_fal, i2v: str = SEEDANCE, **overrides) -> FalVideoProvider:
    fields: dict = {
        "video_provider": "fal",
        "fal_key": "test-key",
        "fal_i2v_model": i2v,
        "fal_t2i_model": FLUX,
        "width": 720,
        "height": 1280,
    }
    fields.update(overrides)
    return FalVideoProvider(Settings(**fields))


# --- spec file integrity --------------------------------------------------- #


def test_every_i2v_spec_declares_the_fields_the_provider_reads():
    specs = load_fal_specs(Settings())
    for model_id, spec in specs["image_to_video"].items():
        for key in ("image_param", "duration_param", "min_duration", "output_key"):
            assert key in spec, f"{model_id} is missing {key}"


def test_unknown_endpoint_fails_at_construction_not_at_first_request(stub_fal):
    """A typo must not be discovered after money has been spent on keyframes."""
    with pytest.raises(VideoProviderError, match="unknown image_to_video"):
        make_provider(stub_fal, i2v="bytedance/seedance-9.9/nope")


def test_missing_key_is_reported_clearly(stub_fal, monkeypatch):
    monkeypatch.delenv("STYLELOOM_FAL_KEY", raising=False)
    with pytest.raises(VideoProviderError, match="FAL_KEY"):
        FalVideoProvider(Settings(video_provider="fal", fal_key=""))


# --- Seedance -------------------------------------------------------------- #


def test_seedance_uses_image_url_and_a_string_duration(stub_fal):
    provider = make_provider(stub_fal, i2v=SEEDANCE)
    payload = provider.build_animate_payload("https://cdn.test/k.jpg", "push in", 1.2, None)

    assert payload["image_url"] == "https://cdn.test/k.jpg"
    assert "start_image_url" not in payload
    assert payload["duration"] == "4", "must be a string, and clamped up to the 4s floor"
    assert payload["aspect_ratio"] == "9:16"
    assert payload["resolution"] == "720p"
    assert payload["generate_audio"] is False


def test_seedance_floor_is_four_seconds(stub_fal):
    provider = make_provider(stub_fal, i2v=SEEDANCE)
    assert provider.min_clip_sec == 4.0
    for requested in (0.5, 1.2, 3.9):
        payload = provider.build_animate_payload("u", "p", requested, None)
        assert payload["duration"] == "4"


def test_seedance_warns_rather_than_silently_dropping_a_persona(stub_fal):
    """The reference image cannot be honoured on this endpoint. Proceeding
    silently would look like persona consistency was working."""
    provider = make_provider(stub_fal, i2v=SEEDANCE)
    with pytest.warns(UserWarning, match="persona ignored"):
        payload = provider.build_animate_payload("u", "p", 4.0, "https://cdn.test/me.jpg")
    assert "elements" not in payload


# --- Kling ----------------------------------------------------------------- #


def test_kling_uses_start_image_url_and_omits_aspect_ratio(stub_fal):
    """Kling infers aspect ratio from the start image, so sending the parameter is
    wrong rather than merely redundant."""
    provider = make_provider(stub_fal, i2v=KLING_PRO)
    payload = provider.build_animate_payload("https://cdn.test/k.jpg", "push in", 1.2, None)

    assert payload["start_image_url"] == "https://cdn.test/k.jpg"
    assert "image_url" not in payload
    assert "aspect_ratio" not in payload
    assert payload["duration"] == "3", "Kling's floor is 3s, not Seedance's 4s"


def test_kling_carries_a_persona_through_elements(stub_fal):
    provider = make_provider(stub_fal, i2v=KLING_PRO)
    payload = provider.build_animate_payload("u", "walk forward", 3.0, "https://cdn.test/me.jpg")

    assert payload["elements"] == [{"frontal_image_url": "https://cdn.test/me.jpg"}]
    assert payload["prompt"].startswith("@Element1")


def test_kling_concurrency_limit_clamps_our_setting(stub_fal):
    """Regression: max_concurrent_renders could exceed Kling's per-user limit of 1
    and fail the whole run."""
    provider = make_provider(stub_fal, i2v=KLING_PRO, max_concurrent_renders=8)
    assert provider.max_concurrency == 1
    effective = max(min(provider.settings.max_concurrent_renders, provider.max_concurrency), 1)
    assert effective == 1


def test_seedance_concurrency_falls_back_to_the_default(stub_fal):
    provider = make_provider(stub_fal, i2v=SEEDANCE)
    assert provider.max_concurrency > 1


# --- keyframes ------------------------------------------------------------- #


def test_flux_takes_an_image_size_object(stub_fal):
    provider = make_provider(stub_fal)
    payload = provider.build_keyframe_payload("a red door")

    assert payload["image_size"] == {"width": 720, "height": 1280}
    assert payload["prompt"] == "a red door"
    assert payload["output_format"] == "jpeg"


def test_landscape_settings_flip_the_aspect_ratio(stub_fal):
    provider = make_provider(stub_fal, i2v=SEEDANCE, width=1280, height=720)
    payload = provider.build_animate_payload("u", "p", 4.0, None)
    assert payload["aspect_ratio"] == "16:9"


# --- Veo 2 ----------------------------------------------------------------- #


def test_veo2_duration_is_a_suffixed_enum_not_a_bare_number(stub_fal):
    """Veo 2's `duration` is "5s"|"6s"|"7s"|"8s". Sending "5" is rejected, and
    that is exactly the class of silent breakage the spec file exists to stop."""
    provider = make_provider(stub_fal, i2v=VEO2)
    payload = provider.build_animate_payload("https://cdn.test/k.jpg", "push in", 1.2, None)

    assert payload["image_url"] == "https://cdn.test/k.jpg"
    assert payload["duration"] == "5s", "5s floor, and the 's' suffix is required"
    assert "start_image_url" not in payload
    assert "aspect_ratio" not in payload, "aspect is inferred from the input image"
    assert "resolution" not in payload


def test_veo2_clamps_to_its_eight_second_ceiling(stub_fal):
    provider = make_provider(stub_fal, i2v=VEO2)
    assert provider.min_clip_sec == 5.0
    assert provider.build_animate_payload("u", "p", 20.0, None)["duration"] == "8s"


def test_veo2_has_no_persona_or_multi_shot_path(stub_fal):
    provider = make_provider(stub_fal, i2v=VEO2)
    assert provider.supports_persona is False
    assert provider.supports_multi_shot is False
    with pytest.warns(UserWarning, match="persona ignored"):
        payload = provider.build_animate_payload("u", "p", 5.0, "https://cdn.test/me.jpg")
    assert "elements" not in payload
