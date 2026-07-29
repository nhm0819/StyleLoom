"""KlingAI Open Platform: token correctness and payload shape.

The payload assertions are here for the same reason the fal ones were -- a wrong
parameter name is accepted and ignored rather than rejected, so nothing fails
until someone watches the output and notices the creator changed face. The
official API has more of these traps than the wrapper did: `multi_shot` and
`shot_type` have to accompany `multi_prompt` or the storyboard is silently one
shot, and a Base64 image with a data URI prefix is a 400.

The JWT assertions are here for a different reason: it is hand-rolled, so it gets
checked against an independently computed signature rather than against itself.

No network and no credentials. Payload construction is separated from the calls,
which is what makes that possible.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json

import pytest
from styleloom_core import Settings
from styleloom_core.errors import VideoProviderError
from styleloom_core.providers.base import MotionShot
from styleloom_core.providers.kling import (
    KlingVideoProvider,
    aspect_ratio_for,
    encode_jwt,
    load_kling_specs,
)

V3 = "kling-v3"
OMNI = "kling-v3-omni"


def make_provider(**overrides) -> KlingVideoProvider:
    fields: dict = {
        "video_provider": "kling",
        "kling_access_key": "test-access",
        "kling_secret_key": "test-secret",
        "width": 720,
        "height": 1280,
    }
    fields.update(overrides)
    return KlingVideoProvider(Settings(**fields))


# --- spec file integrity --------------------------------------------------- #


def test_every_i2v_spec_declares_the_fields_the_provider_reads():
    specs = load_kling_specs(Settings())
    for model_name, spec in specs["image_to_video"].items():
        for key in ("path", "image_param", "duration_param", "min_duration", "output_key"):
            assert key in spec, f"{model_name} is missing {key}"


def test_every_t2i_spec_declares_the_fields_the_provider_reads():
    specs = load_kling_specs(Settings())
    for model_name, spec in specs["text_to_image"].items():
        for key in ("path", "output_key"):
            assert key in spec, f"{model_name} is missing {key}"


def test_an_unknown_model_name_fails_on_construction():
    """Before any credits are spent, not at the first request."""
    with pytest.raises(VideoProviderError, match="unknown image_to_video"):
        make_provider(kling_i2v_model="kling-v9-imaginary")


def test_missing_credentials_fail_on_construction():
    with pytest.raises(VideoProviderError, match="KLING_ACCESS_KEY"):
        make_provider(kling_secret_key="")


# --- JWT ------------------------------------------------------------------- #


def test_jwt_signature_matches_an_independently_computed_one():
    """Recomputed from the spec rather than compared to a stored string, so this
    fails if the encoding drifts rather than if the token merely changes."""
    token = encode_jwt("ak", "sk", issued_at=1_700_000_000)
    header_b64, payload_b64, signature_b64 = token.split(".")

    expected = hmac.new(
        b"sk", f"{header_b64}.{payload_b64}".encode("ascii"), hashlib.sha256
    ).digest()
    assert signature_b64 == base64.urlsafe_b64encode(expected).rstrip(b"=").decode()


def test_jwt_carries_the_claims_kling_checks():
    token = encode_jwt("my-access-key", "sk", issued_at=1_700_000_000)
    payload = json.loads(_unpad(token.split(".")[1]))
    assert payload["iss"] == "my-access-key"
    assert payload["exp"] == 1_700_000_000 + 1800
    # Backdated on purpose: a client clock one second ahead of Kling's would
    # otherwise have its own token rejected as not-yet-valid.
    assert payload["nbf"] == 1_700_000_000 - 5


def test_jwt_header_declares_hs256():
    header = json.loads(_unpad(encode_jwt("ak", "sk", 0).split(".")[0]))
    assert header == {"alg": "HS256", "typ": "JWT"}


def test_jwt_is_unpadded_base64url():
    """`=` padding and `+/` are both illegal in a JWT segment."""
    for segment in encode_jwt("a" * 7, "s" * 11, 1_700_000_001).split("."):
        assert "=" not in segment
        assert "+" not in segment and "/" not in segment


def _unpad(segment: str) -> bytes:
    return base64.urlsafe_b64decode(segment + "=" * (-len(segment) % 4))


# --- keyframe payload ------------------------------------------------------ #


def test_keyframe_asks_for_a_vertical_ratio_not_a_pixel_size():
    """The official image API has no width/height, only a ratio enum."""
    payload = make_provider().build_keyframe_payload("a face")
    assert payload["aspect_ratio"] == "9:16"
    assert "width" not in payload and "height" not in payload
    assert payload["model_name"] == V3


def test_landscape_settings_pick_a_landscape_ratio():
    payload = make_provider(width=1920, height=1080).build_keyframe_payload("x")
    assert payload["aspect_ratio"] == "16:9"


def test_an_unusual_size_snaps_to_the_nearest_legal_ratio():
    """1080x1350 is 4:5, which Kling does not offer; 3:4 is the closest."""
    assert aspect_ratio_for(1080, 1350, ["9:16", "16:9", "1:1", "3:4"]) == "3:4"


# --- animate payload ------------------------------------------------------- #


def test_animate_uses_the_official_parameter_names():
    payload = make_provider().build_animate_payload("BASE64", "she smiles", 1.0, None)
    # `image`, not fal's `start_image_url`.
    assert payload["image"] == "BASE64"
    assert "start_image_url" not in payload
    assert payload["model_name"] == V3
    assert payload["mode"] == "std"


def test_duration_is_a_bare_string_at_or_above_the_floor():
    """A 0.76s cut cannot be bought; the endpoint floor is what gets billed."""
    payload = make_provider().build_animate_payload("B64", "p", 0.76, None)
    assert payload["duration"] == "3"
    assert isinstance(payload["duration"], str)


def test_duration_is_capped_at_the_endpoint_maximum():
    payload = make_provider().build_animate_payload("B64", "p", 99.0, None)
    assert payload["duration"] == "15"


def test_audio_is_off_by_default():
    """Captions are burned in and the reference set has no dialogue, so audio is
    a surcharge for something the output discards."""
    assert make_provider().build_animate_payload("B64", "p", 3.0, None)["sound"] == "off"


def test_a_persona_is_refused_rather_than_silently_dropped():
    """The failure this prevents is invisible otherwise: the video renders fine
    and the creator's face changes between cuts."""
    with pytest.raises(VideoProviderError, match="silently dropped"):
        make_provider().build_animate_payload("B64", "p", 3.0, "PERSONA_B64")


def test_supports_persona_is_false_and_says_so():
    """Declared, so the caller does not pay to generate a reference portrait
    that the request would then drop."""
    assert make_provider().supports_persona is False


# --- multi-shot payload ---------------------------------------------------- #


def test_sequence_sets_the_two_flags_that_make_multi_prompt_readable():
    """`multi_prompt` alone is ignored: without `multi_shot` and `shot_type` the
    request is a single shot and nothing says so."""
    payload = make_provider().build_sequence_payload(
        "B64", [MotionShot("a", 1.0), MotionShot("b", 2.0)], None
    )
    assert payload["multi_shot"] == "true"
    assert payload["shot_type"] == "customize"
    assert len(payload["multi_prompt"]) == 2


def test_sequence_drops_the_top_level_prompt():
    """The schema makes `prompt` invalid once multi_shot is on. Leaving it in is
    how a storyboard silently collapses to one shot."""
    assert "prompt" not in make_provider().build_sequence_payload(
        "B64", [MotionShot("a", 1.0)], None
    )


def test_shot_entries_are_indexed_from_one():
    """Kling orders the storyboard by `index`, not by array position."""
    payload = make_provider().build_sequence_payload(
        "B64", [MotionShot("a", 1.0), MotionShot("b", 1.0), MotionShot("c", 1.0)], None
    )
    assert [e["index"] for e in payload["multi_prompt"]] == [1, 2, 3]


def test_sub_second_cuts_are_billed_and_delivered_at_one_second():
    """Per-shot duration is an integer with a floor of 1, so a 0.4s cut has no
    legal representation. qc reports the drift rather than this hiding it."""
    payload = make_provider().build_sequence_payload(
        "B64", [MotionShot("a", 0.4)], None
    )
    assert payload["multi_prompt"][0]["duration"] == "1"


def test_half_seconds_round_up_not_to_even():
    """floor(x + 0.5), not round(): Python's round(2.5) is 2, which would quietly
    lose half a second off every 2.5s cut."""
    payload = make_provider().build_sequence_payload(
        "B64", [MotionShot("a", 2.5)], None
    )
    assert payload["multi_prompt"][0]["duration"] == "3"


def test_the_omni_model_posts_to_a_different_path():
    """The trap fal's naming hid: on fal both were one call shape under
    kling-video/{v3,o3}. Officially they are separate endpoints."""
    v3 = make_provider(kling_i2v_model=V3)
    omni = make_provider(kling_i2v_model=OMNI)
    assert v3.i2v["path"] == "/v1/videos/image2video"
    assert omni.i2v["path"] == "/v1/videos/omni-video/"


def test_storyboard_cap_is_six_shots():
    """Verified from the docs, and the windowing arithmetic depends on it: 14
    sub-second cuts fit in one 15s window but still need three requests."""
    assert make_provider().max_shots_per_request == 6


def test_concurrency_comes_from_settings_not_the_spec():
    """Kling's ceiling is an account-tier property, so no per-model value in the
    spec file could be correct for every reader."""
    assert make_provider(kling_max_concurrency=4).max_concurrency == 4


# --- duration quantisation ------------------------------------------------- #
#
# Two different rounding rules, and the difference is load-bearing. Regression:
# both modes used round() to nearest, which is banker's rounding on top of being
# the wrong direction for per_shot.


@pytest.mark.parametrize("wanted", [3.4, 4.4, 4.5, 6.49, 10.01])
def test_per_shot_never_buys_a_clip_shorter_than_the_cut(wanted):
    """`render_shot` trims a long clip down and cannot extend a short one, so
    rounding down leaves the timeline quietly short with nothing to notice it."""
    sent = float(make_provider().build_animate_payload("B64", "p", wanted, None)["duration"])
    assert sent >= wanted


def test_per_shot_rounds_up_rather_than_to_nearest():
    """4.4s asked for as 4s was a real bug: the clip came back 0.4s short and
    passed straight through, because the trim only fires when the clip is long."""
    payload = make_provider().build_animate_payload("B64", "p", 4.4, None)
    assert payload["duration"] == "5"


def test_multi_shot_rounds_to_nearest_not_up():
    """The opposite rule, on purpose. A multi-shot clip is never trimmed, so
    rounding up would stretch every video; nearest minimises total drift."""
    payload = make_provider().build_sequence_payload(
        "B64", [MotionShot("a", 4.4), MotionShot("b", 4.6)], None
    )
    assert [e["duration"] for e in payload["multi_prompt"]] == ["4", "5"]


def test_the_payload_and_the_planner_agree_on_shot_length():
    """`split_windows` packs against `shot_billed_duration`. If the payload
    rounded differently the request would overflow the window just verified."""
    provider = make_provider()
    shots = [MotionShot("a", 0.4), MotionShot("b", 2.5), MotionShot("c", 4.4)]
    payload = provider.build_sequence_payload("B64", shots, None)
    assert [float(e["duration"]) for e in payload["multi_prompt"]] == [
        provider.shot_billed_duration(s.duration) for s in shots
    ]
