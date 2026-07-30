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
from pathlib import Path

import cv2
import numpy as np
import pytest
from styleloom_core import Settings
from styleloom_core.errors import MediaError, VideoProviderError
from styleloom_core.providers.base import MotionShot
from styleloom_core.providers.kling import (
    KlingVideoProvider,
    aspect_ratio_for,
    load_kling_specs,
)

V3 = "kling-v3"
OMNI = "kling-v3-omni"
# The 3.0 API entry. A path segment rather than a `model_name`, unlike the two above.
V3_API = "kling-3.0"


def make_provider(**overrides) -> KlingVideoProvider:
    fields: dict = {
        "video_provider": "kling",
        "kling_api_key": "test-api-key",
        "width": 720,
        "height": 1280,
    }
    fields.update(overrides)
    return KlingVideoProvider(Settings(**fields))


def write_image(path: Path, width: int = 720, height: int = 1280) -> Path:
    """A real decodable still.

    Byte strings that merely start with a JPEG magic number used to be enough here.
    They are not any more: the provider checks a frame's dimensions and ratio before
    base64'ing it, and 720x1280 is what the image endpoint returns for the 9:16 1k
    request this pipeline makes.
    """
    cv2.imwrite(str(path), np.full((height, width, 3), 128, dtype=np.uint8))
    return path


def one_cut(provider, prompt: str = "p", duration: float = 3.0, frame=None) -> dict:
    """A request carrying a single cut.

    `build_generate_payload` used to be a second entry point for this. It went with
    `per_shot`: every render request now carries a shot list, and a list of one shot
    is a request with one shot in it.
    """
    return provider.build_sequence_payload([MotionShot(prompt, duration)], frame)


def legacy_provider(**overrides) -> KlingVideoProvider:
    """Pinned to the /v1/videos/* entries.

    The legacy endpoints are still in the spec file for an account pinned to them,
    so their payload shape is still asserted -- but it is no longer what a default
    provider builds, and a test that wants the flat request has to ask for it.
    """
    return make_provider(kling_t2v_model=V3, kling_i2v_model=V3, **overrides)


# --- spec file integrity --------------------------------------------------- #


def test_every_t2v_spec_declares_the_fields_the_provider_reads():
    specs = load_kling_specs(Settings())
    for model_name, spec in specs["text_to_video"].items():
        # On v3 the result is a typed `outputs` list, so `output_key` names nothing.
        required = (
            ("path", "task_path", "output_type", "duration_param", "min_duration")
            if spec.get("api") == "v3"
            else ("path", "duration_param", "min_duration", "output_key")
        )
        for key in required:
            assert key in spec, f"{model_name} is missing {key}"


def test_an_unknown_model_name_fails_on_construction():
    """Before any credits are spent, not at the first request."""
    with pytest.raises(VideoProviderError, match="unknown text_to_video"):
        make_provider(kling_t2v_model="kling-v9-imaginary")


def test_a_missing_api_key_fails_on_construction():
    with pytest.raises(VideoProviderError, match="KLING_API_KEY"):
        make_provider(kling_api_key="")


# --- authentication -------------------------------------------------------- #


def test_the_api_key_is_sent_verbatim_as_a_bearer_token():
    """No signing and no expiry.

    The older scheme took an access key and a secret key and had the client build an
    HS256 JWT per request -- `iss` the access key, signed with the secret, 30-minute
    `exp`, `nbf` backdated for clock skew. Those tests, and the hand-rolled signer
    they covered, are gone with it: the current API Key *is* the bearer token, and it
    is the only credential form that covers models past 3.0.
    """
    headers = make_provider(kling_api_key="ak-live-123")._headers()
    assert headers["Authorization"] == "Bearer ak-live-123"
    # A JWT would have two dots in it. This must not be signed or wrapped.
    assert headers["Authorization"].count(".") == 0


# --- keyframe payload ------------------------------------------------------ #


def test_a_vertical_ratio_is_asked_for_rather_than_a_pixel_size():
    """The API has no width/height, only a ratio enum. On 3.0 it lives inside
    `settings`; at the top level it is ignored."""
    payload = one_cut(make_provider(), "a face")
    assert payload["settings"]["aspect_ratio"] == "9:16"
    assert "width" not in payload and "height" not in payload
    assert "aspect_ratio" not in payload


def test_landscape_settings_pick_a_landscape_ratio():
    payload = one_cut(make_provider(width=1920, height=1080), "x")
    assert payload["settings"]["aspect_ratio"] == "16:9"


def test_an_unusual_size_snaps_to_the_nearest_legal_ratio():
    """1080x1350 is 4:5, which Kling does not offer; 3:4 is the closest."""
    assert aspect_ratio_for(1080, 1350, ["9:16", "16:9", "1:1", "3:4"]) == "3:4"


# --- animate payload ------------------------------------------------------- #


def test_text_to_video_keeps_a_top_level_prompt_string():
    """The two 3.0 video endpoints disagree on where the prompt goes: image-to-video
    takes a typed `contents` array because it also carries the start frame, and
    text-to-video, having no image, keeps a `prompt` string. A `contents` array sent
    here is ignored and the task fails on a missing prompt."""
    payload = one_cut(make_provider(), "she smiles", 1.0)
    assert payload["prompt"].startswith("she smiles")
    assert "contents" not in payload
    assert set(payload) == {"prompt", "settings"}
    for gone in ("model_name", "mode", "duration", "aspect_ratio", "negative_prompt"):
        assert gone not in payload


def test_duration_is_capped_at_the_endpoint_maximum():
    with pytest.raises(VideoProviderError, match="over the 15s"):
        one_cut(make_provider(), "p", 99.0)


def test_the_audio_setting_reaches_the_request():
    """`STYLELOOM_AUDIO` used to be unread: the spec file's value was the only thing
    that ever reached the request, so changing the setting did nothing."""
    assert one_cut(make_provider(audio="native"))["settings"]["audio"] == "native"
    assert one_cut(make_provider(audio="off"))["settings"]["audio"] == "off"


def test_the_setting_overrides_the_endpoint_default():
    """The spec file states what the endpoint defaults to; the run decides."""
    provider = make_provider(audio="off")
    assert (provider.i2v.get("settings_defaults") or {}).get("audio") == "native"
    assert one_cut(provider)["settings"]["audio"] == "off"


def test_the_legacy_endpoint_keeps_its_own_audio_field_name():
    """`sound`, not `audio` -- and it is a spec-file default, not driven by the
    setting, because that field is not the same field."""
    assert one_cut(legacy_provider())["sound"] == "off"




# --- multi-shot payload ---------------------------------------------------- #


def test_sequence_sets_the_two_flags_that_make_multi_prompt_readable():
    """`multi_prompt` alone is ignored: without `multi_shot` and `shot_type` the
    request is a single shot and nothing says so."""
    payload = legacy_provider().build_sequence_payload(
        [MotionShot("a", 1.0), MotionShot("b", 2.0)]
    )
    assert payload["multi_shot"] == "true"
    assert payload["shot_type"] == "customize"
    assert len(payload["multi_prompt"]) == 2


def test_sequence_drops_the_top_level_prompt():
    """The schema makes `prompt` invalid once multi_shot is on. Leaving it in is
    how a storyboard silently collapses to one shot."""
    assert "prompt" not in legacy_provider().build_sequence_payload([MotionShot("a", 1.0)])


def test_shot_entries_are_indexed_from_one():
    """Kling orders the storyboard by `index`, not by array position."""
    payload = legacy_provider().build_sequence_payload(
        [MotionShot("a", 1.0), MotionShot("b", 1.0), MotionShot("c", 1.0)]
    )
    assert [e["index"] for e in payload["multi_prompt"]] == [1, 2, 3]


def test_sub_second_cuts_are_billed_and_delivered_at_one_second():
    """Per-shot duration is an integer with a floor of 1, so a 0.4s cut has no
    legal representation. qc reports the drift rather than this hiding it."""
    payload = legacy_provider().build_sequence_payload([MotionShot("a", 0.4)])
    assert payload["multi_prompt"][0]["duration"] == "1"


def test_half_seconds_round_up_not_to_even():
    """floor(x + 0.5), not round(): Python's round(2.5) is 2, which would quietly
    lose half a second off every 2.5s cut."""
    payload = legacy_provider().build_sequence_payload([MotionShot("a", 2.5)])
    assert payload["multi_prompt"][0]["duration"] == "3"


def test_the_omni_model_posts_to_a_different_path():
    """The trap fal's naming hid: on fal both were one call shape under
    kling-video/{v3,o3}. Officially they are separate endpoints."""
    v3 = make_provider(kling_t2v_model=V3)
    omni = make_provider(kling_t2v_model=OMNI)
    assert v3.t2v["path"] == "/v1/videos/text2video"
    assert omni.t2v["path"] == "/v1/videos/omni-video/"


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


def test_multi_shot_rounds_to_nearest_not_up():
    """The opposite rule, on purpose. A multi-shot clip is never trimmed, so
    rounding up would stretch every video; nearest minimises total drift."""
    payload = legacy_provider().build_sequence_payload(
        [MotionShot("a", 4.4), MotionShot("b", 4.6)]
    )
    assert [e["duration"] for e in payload["multi_prompt"]] == ["4", "5"]


def test_the_payload_and_the_planner_agree_on_shot_length():
    """`split_windows` packs against `shot_billed_duration`. If the payload
    rounded differently the request would overflow the window just verified."""
    provider = legacy_provider()
    shots = [MotionShot("a", 0.4), MotionShot("b", 2.5), MotionShot("c", 4.4)]
    payload = provider.build_sequence_payload(shots)
    assert [float(e["duration"]) for e in payload["multi_prompt"]] == [
        provider.shot_billed_duration(s.duration) for s in shots
    ]


@pytest.mark.parametrize("model_name", [V3, OMNI])
def test_every_model_enforces_the_prompt_limits(model_name):
    """The omni entry was missing both limits and the negative prompt, so
    switching to it silently turned off caption suppression and the budget."""
    provider = make_provider(kling_t2v_model=model_name, kling_i2v_model=V3)
    assert provider.max_shot_prompt_chars == 512
    assert one_cut(provider)["negative_prompt"]
    with pytest.raises(VideoProviderError, match="over"):
        provider.build_sequence_payload([MotionShot("x" * 600, 1.0)])


# --- first frame: image payload -------------------------------------------- #
#
# The keyframe stage is back after having been removed once. It is a different
# design -- one anchor per run, reused as the reference for every request's opening
# frame -- and these assertions are on the part that made the old one pointless:
# whether the anchor actually reaches the request.


def test_every_new_spec_declares_the_fields_the_provider_reads():
    specs = load_kling_specs(Settings())
    for spec in specs["text_to_image"].values():
        for key in ("path", "output_key", "reference_param"):
            assert key in spec
    for model_name, spec in specs["image_to_video"].items():
        # Two API generations, two sets of required keys. On v3 the start frame is
        # a typed entry in `contents` and the result is a typed `outputs` list, so
        # `first_frame_param` and `output_key` have nothing to name.
        required = (
            ("path", "task_path", "output_type", "duration_param", "min_duration")
            if spec.get("api") == "v3"
            else ("path", "first_frame_param", "duration_param", "output_key")
        )
        for key in required:
            assert key in spec, f"{model_name} is missing {key}"


def test_an_unknown_i2v_model_fails_on_construction():
    """Before the keyframe is paid for, not after."""
    with pytest.raises(VideoProviderError, match="unknown image_to_video"):
        make_provider(kling_i2v_model="kling-v9-imaginary")


def test_an_image_payload_without_a_reference_sends_no_reference_field():
    payload = make_provider().build_image_payload("a face")
    assert payload["model_name"] == V3
    assert payload["aspect_ratio"] == "9:16"
    assert "image" not in payload


def test_a_reference_is_raw_base64_with_no_data_uri_prefix(tmp_path):
    """The prefix is a documented 400, and every browser-facing base64 example
    carries one, so this is the easiest mistake in the whole integration."""
    ref = write_image(tmp_path / "anchor.jpg")
    payload = make_provider().build_image_payload("same person, wider", ref)
    assert payload["image"] == base64.b64encode(ref.read_bytes()).decode()
    assert not payload["image"].startswith("data:")


def test_a_reference_is_not_prefixed_with_an_undocumented_token(tmp_path):
    """`<<<image_1>>>` was prepended here, and /v1/images/generations documents no
    token addressing of any kind -- the reference is the `image` field. So the
    prefix addressed nothing and put twelve literal characters at the front of the
    prompt, where the model weights most heavily."""
    ref = write_image(tmp_path / "anchor.jpg")
    payload = make_provider().build_image_payload("wider angle", ref)
    assert payload["prompt"] == "wider angle"
    assert payload["image"]


# --- first frame: video payload -------------------------------------------- #


def _frame(tmp_path):
    return write_image(tmp_path / "lead.jpg")


def test_omni_carries_the_first_frame_as_a_typed_list_entry(tmp_path):
    """`type` is what separates a start frame from a style reference on omni. An
    untyped entry is accepted and billed, the video does not start on the frame,
    and nothing in the response says so."""
    frame = _frame(tmp_path)
    provider = make_provider(kling_i2v_model=OMNI)
    payload = one_cut(provider, "she smiles", 3.0, frame)
    assert payload["model_name"] == OMNI
    entry = payload["image_list"][0]
    assert entry["type"] == "first_frame"
    assert entry["id"] == "image_1"
    assert entry["url"] == base64.b64encode(frame.read_bytes()).decode()


def test_the_non_omni_endpoint_takes_a_flat_base64_field(tmp_path):
    """Two shapes for the same concept. A flat string sent to omni, or a list sent
    to image2video, is the silent-ignore failure this spec file exists to prevent."""
    provider = make_provider(kling_i2v_model=V3)
    payload = one_cut(provider, "p", 3.0, _frame(tmp_path))
    assert isinstance(payload["image"], str)
    assert "image_list" not in payload


def test_a_request_with_a_start_frame_states_no_aspect_ratio(tmp_path):
    """The endpoint reads the ratio off the image, and has no `aspect_ratio` field
    at all. Sending one can only disagree with the frame that was just uploaded.

    Checked inside `settings` as well as at the top level: this test passed while
    the ratio was being sent one level down, because 3.0 moved the field there.
    """
    payload = one_cut(make_provider(), "p", 3.0, _frame(tmp_path))
    assert "aspect_ratio" not in payload
    assert "aspect_ratio" not in payload["settings"]


def test_no_start_frame_still_states_the_ratio():
    """The other half of the rule: text-to-video has nothing to read it off, and
    `aspect_ratio` is a field on that endpoint only."""
    payload = one_cut(make_provider())
    assert "aspect_ratio" in payload["settings"]


def test_a_storyboard_can_be_anchored_and_still_be_a_storyboard(tmp_path):
    """Both mechanisms at once, which is the whole point: the shot list holds
    identity within one generation and the frame holds it between them."""
    payload = make_provider().build_sequence_payload(
        [MotionShot("a", 1.0), MotionShot("b", 2.0)], _frame(tmp_path)
    )
    assert payload["settings"]["multi_shot"] is True
    types = [c["type"] for c in payload["contents"]]
    assert types == ["prompt", "first_frame"]
    assert payload["contents"][0]["text"] == "shot 1, 1, a; shot 2, 2, b;"


def test_the_render_spec_follows_the_first_frame_setting():
    """Capability properties have to describe the endpoint that will actually be
    called, or the caller budgets prompts and packs windows against another one."""
    assert make_provider().render_id == V3_API
    assert make_provider().render["path"] == "/image-to-video/kling-3.0"
    assert make_provider(use_first_frame=False).render["path"] == "/text-to-video/kling-3.0"


# --- semicolon shot syntax ------------------------------------------------- #


def test_semicolon_shots_follow_the_documented_shape():
    """`shot n, m, words;` -- n the 1-based number, m whole seconds."""
    text = make_provider().format_semicolon_shots(
        [MotionShot("wide establishing", 3.0), MotionShot("close up", 4.4)]
    )
    assert text == "shot 1, 3, wide establishing; shot 2, 4, close up;"


def test_a_semicolon_inside_a_prompt_is_neutralised():
    """There is no documented escape, so a stray one would end the shot early and
    shift every cut after it."""
    text = make_provider().format_semicolon_shots([MotionShot("a; b", 1.0)])
    assert text == "shot 1, 1, a, b;"


def test_a_v3_sequence_carries_the_shot_list_in_the_prompt_and_no_array(tmp_path):
    """On the 3.0 API the semicolon list is not an opt-in alternative to
    `multi_prompt` -- it is the only multi-shot form the API has."""
    payload = make_provider().build_sequence_payload(
        [MotionShot("a", 1.0), MotionShot("b", 2.0)], _frame(tmp_path)
    )
    assert "multi_prompt" not in payload
    assert payload["contents"][0]["text"] == "shot 1, 1, a; shot 2, 2, b;"


# --- the 3.0 request envelope ---------------------------------------------- #
#
# The whole reason these assertions exist: the legacy request is not rejected by
# the 3.0 endpoint, it is *ignored*. Unknown keys are dropped, `contents` is
# missing, and the task fails on a missing prompt or succeeds on five seconds of
# defaults. Nothing in the response names the fields that went nowhere.


def test_the_v3_request_is_contents_and_settings_and_nothing_flat(tmp_path):
    payload = one_cut(make_provider(), "she smiles", 6.0, _frame(tmp_path))
    assert set(payload) == {"contents", "settings"}
    for gone in ("model_name", "mode", "prompt", "image", "duration", "aspect_ratio"):
        assert gone not in payload, f"{gone} is a legacy field, silently ignored on 3.0"


def test_the_v3_prompt_and_frame_are_typed_entries_in_contents(tmp_path):
    frame = _frame(tmp_path)
    contents = one_cut(make_provider(), "she smiles", 3.0, frame)["contents"]
    assert contents[0]["type"] == "prompt"
    assert contents[0]["text"].startswith("she smiles")
    assert contents[1]["type"] == "first_frame"
    assert contents[1]["url"] == base64.b64encode(frame.read_bytes()).decode()
    assert not contents[1]["url"].startswith("data:")


def test_v3_duration_is_an_int_inside_settings(tmp_path):
    """A bare string was right for the legacy `duration` field and is the wrong
    type here, where the schema is an int enum."""
    settings = one_cut(make_provider(), "p", 6.0, _frame(tmp_path))["settings"]
    assert settings["duration"] == 6
    assert isinstance(settings["duration"], int)


def test_a_single_cut_v3_request_turns_multi_shot_off_explicitly(tmp_path):
    """`multi_shot` defaults to TRUE on this API. Left unsaid, the model may cut
    one shot into several -- and per_shot mode trims one file as one cut."""
    settings = one_cut(make_provider(), "p", 3.0, _frame(tmp_path))["settings"]
    assert settings["multi_shot"] is False


def test_v3_uses_settings_audio_and_sends_no_negative_prompt(tmp_path):
    """`sound` and `negative_prompt` are legacy names. The audio switch is
    `settings.audio`; the negative field does not exist at all."""
    payload = one_cut(make_provider(audio="off"), "p", 3.0, _frame(tmp_path))
    assert payload["settings"]["audio"] == "off"
    assert "sound" not in payload["settings"]
    assert "negative_prompt" not in payload["settings"]


def test_the_quality_tier_becomes_a_resolution_because_mode_does_not_exist(tmp_path):
    """`mode` is not a field on 3.0, so std/pro would have silently done nothing.
    It means what it always meant: std renders 720p, pro 1080p."""
    frame = _frame(tmp_path)
    std = one_cut(make_provider(), "p", 3.0, frame)
    pro = one_cut(make_provider(kling_mode="pro"), "p", 3.0, frame)
    assert std["settings"]["resolution"] == "720p"
    assert pro["settings"]["resolution"] == "1080p"


def test_the_v3_total_duration_equals_the_sum_of_the_shot_durations(tmp_path):
    """The documented constraint on a shot list, and the one most likely to be
    missed: `settings.duration` was not sent at all, so a 12s storyboard would
    have been rendered as 5s of default."""
    shots = [MotionShot("a", 3.0), MotionShot("b", 4.0), MotionShot("c", 2.0)]
    payload = make_provider().build_sequence_payload(shots, _frame(tmp_path))
    assert payload["contents"][0]["text"].startswith("shot 1, 3, a; shot 2, 4, b;")
    assert payload["settings"]["duration"] == 9


def test_a_short_shot_list_is_lifted_to_the_duration_floor(tmp_path):
    """Two sub-second cuts quantise to 1 + 1 = 2s, under the 3s floor, and the
    endpoint rejects the request whole. The deficit lands on the closing cut."""
    payload = make_provider().build_sequence_payload(
        [MotionShot("a", 1.2), MotionShot("b", 1.3)], _frame(tmp_path)
    )
    assert payload["settings"]["duration"] == 3
    assert payload["contents"][0]["text"] == "shot 1, 1, a; shot 2, 2, b;"


def test_a_shot_list_longer_than_one_request_is_refused_not_truncated(tmp_path):
    """Silently dropping the overflow would delete video the storyboard asked for."""
    with pytest.raises(VideoProviderError, match="over the 15s"):
        make_provider().build_sequence_payload(
            [MotionShot("a", 20.0)], _frame(tmp_path)
        )


def test_the_negative_clause_goes_in_the_prompt_on_single_cut_requests(tmp_path):
    """There is no negative_prompt field on 3.0 -- the prompt itself carries
    negative descriptions, so the clause has to be inside the text or nowhere."""
    payload = one_cut(make_provider(), "she smiles", 3.0, _frame(tmp_path))
    assert "captions" in payload["contents"][0]["text"]


def test_the_negative_clause_stays_out_of_a_semicolon_shot_list(tmp_path):
    """The documented form is `shot n, m, words;` with no place for a global clause.
    Prose before the list or after the final semicolon may parse as a malformed
    shot, which collapses the storyboard to one long cut and still reports success.
    Multi-shot relies on the start frame instead, which `keyframe` generates asking
    for no text, no captions and no watermark."""
    text = make_provider().build_sequence_payload(
        [MotionShot("a", 2.0), MotionShot("b", 2.0)], _frame(tmp_path)
    )["contents"][0]["text"]
    assert text == "shot 1, 2, a; shot 2, 2, b;"
    assert text.endswith(";")


def test_text_to_video_carries_a_shot_list_the_same_way():
    """One shot-list syntax across both 3.0 video endpoints; only the field the
    prompt sits in differs."""
    payload = make_provider().build_sequence_payload(
        [MotionShot("wide", 2.0), MotionShot("close", 3.0)]
    )
    assert payload["prompt"] == "shot 1, 2, wide; shot 2, 3, close;"
    assert payload["settings"]["duration"] == 5
    assert payload["settings"]["multi_shot"] is True


# --- constraints ----------------------------------------------------------- #
#
# Checked before anything is spent, because there is nothing to check afterwards:
# a malformed shot list is read as prose, rendered as one long shot, reported as
# `succeeded` and billed.


def test_more_shots_than_one_request_holds_is_refused(tmp_path):
    """1-6 shots per request. `render` packs to the cap, so a longer list means the
    cap in the spec file disagrees with the endpoint."""
    shots = [MotionShot(f"cut {i}", 2.0) for i in range(7)]
    with pytest.raises(VideoProviderError, match="limit of 6"):
        make_provider().build_sequence_payload(shots, _frame(tmp_path))


def test_an_empty_shot_list_is_refused(tmp_path):
    with pytest.raises(VideoProviderError, match="at least one shot"):
        make_provider().build_sequence_payload([], _frame(tmp_path))


def test_the_512_budget_applies_even_to_a_lone_shot(tmp_path):
    """The endpoint would allow the whole 3072 for a single cut, and the budget is
    still 512: `render` re-packs windows whenever the storyboard or the endpoint's
    limits change, so a prompt that is legal only while it happens to be alone in
    its window breaks on the next run."""
    with pytest.raises(VideoProviderError, match="512-character limit"):
        one_cut(make_provider(), "x" * 600, 3.0, _frame(tmp_path))


def test_a_frame_below_the_minimum_side_is_refused(tmp_path):
    frame = write_image(tmp_path / "small.jpg", 200, 400)
    with pytest.raises(VideoProviderError, match="at least 300px"):
        one_cut(make_provider(), "p", 3.0, frame)


def test_a_frame_outside_the_ratio_range_is_refused(tmp_path):
    """1:2.5 to 2.5:1. A 9:16 vertical frame is 0.56 and legal; 300x1200 is not."""
    frame = write_image(tmp_path / "tall.jpg", 300, 1200)
    with pytest.raises(VideoProviderError, match="1:2.5 to 2.5:1"):
        one_cut(make_provider(), "p", 3.0, frame)


def test_the_frame_this_pipeline_actually_produces_passes(tmp_path):
    """720x1280 is what the image endpoint returns for a 9:16 1k request."""
    frame = write_image(tmp_path / "ok.jpg", 720, 1280)
    assert one_cut(make_provider(), "p", 3.0, frame)["contents"][1]["url"]


def test_an_unreadable_frame_is_caught_before_it_is_uploaded(tmp_path):
    """`_download` writes whatever the response contained, so a truncated file is a
    real possibility -- and base64 inflates the body by a third before the far end
    gets a chance to reject it."""
    frame = tmp_path / "truncated.jpg"
    frame.write_bytes(b"\xff\xd8\xff\xe0")
    with pytest.raises(MediaError, match="not a readable image"):
        one_cut(make_provider(), "p", 3.0, frame)


def test_a_frame_in_an_unsupported_format_is_refused(tmp_path):
    frame = write_image(tmp_path / "frame.png", 720, 1280).rename(
        tmp_path / "frame.webp"
    )
    with pytest.raises(VideoProviderError, match="accepts"):
        one_cut(make_provider(), "p", 3.0, frame)


# --- the 3.0 task endpoint ------------------------------------------------- #


def test_v3_reads_the_task_id_from_data_id_not_data_task_id():
    """`data.task_id` is the legacy key. Missing it aborts a task that was
    created and will be billed regardless."""
    provider = make_provider()
    body = {"code": 0, "data": {"id": "893605946402811985", "status": "submitted"}}
    assert provider._task_record(provider.i2v, {"data": [body["data"]]}, "893605946402811985")


def test_v3_polling_finds_the_task_in_a_list_response():
    """/tasks answers with a list even for one id."""
    provider = make_provider()
    body = {"data": [{"id": "1", "status": "processing"}, {"id": "2", "status": "succeeded"}]}
    assert provider._task_record(provider.i2v, body, "2")["status"] == "succeeded"


def test_v3_polling_treats_a_missing_record_as_not_yet_rather_than_an_error():
    """The list can be empty for a moment after submission; an empty dict has no
    terminal status, so the loop simply polls again."""
    provider = make_provider()
    assert provider._task_record(provider.i2v, {"data": []}, "7") == {}


def test_v3_takes_the_url_from_the_typed_outputs_list():
    """`task_result.videos[]` is the legacy shape. On 3.0 the results are a typed
    list, and `watermark_url` sits beside `url` on the same entry."""
    provider = make_provider()
    task = {
        "outputs": [
            {"type": "image", "url": "https://example.com/frame.png"},
            {
                "type": "video",
                "url": "https://example.com/clip.mp4",
                "watermark_url": "https://example.com/wm.mp4",
            },
        ]
    }
    assert provider._result_url(provider.i2v, task) == "https://example.com/clip.mp4"


def test_the_legacy_response_shape_still_reads():
    provider = make_provider(kling_i2v_model=OMNI)
    task = {"task_result": {"videos": [{"url": "https://example.com/legacy.mp4"}]}}
    assert provider._result_url(provider.i2v, task) == "https://example.com/legacy.mp4"
