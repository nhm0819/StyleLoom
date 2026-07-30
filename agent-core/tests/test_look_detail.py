"""Where the reference's visual vocabulary ends up.

The output looked generic for a measurable reason: `style.json` carried seven
descriptive phrases about the reference and none of them reached a model. The
keyframe prompts used a helper that emitted only the grade name and two measured
numbers, and the video prompt's compression dropped the phrases first -- so the
surviving content was `saturation 0.35, contrast 0.44`, which no image or video
model interprets.

These tests pin down the fix in the direction the constraint actually runs: the
image endpoint allows 2500 characters and the video endpoint 512 per shot, so the
description belongs in the image prompt and the video inherits it from the frame.
"""

from __future__ import annotations

from styleloom_core.budget import (
    BODY_MOTION,
    descriptive_phrases,
    look_tokens,
    motion_sentence,
    style_tokens,
)
from styleloom_core.providers.kling import load_kling_specs
from styleloom_core.schema import LookDetail, Shot, StyleSchema
from styleloom_core.tools import casting as casting_tool
from styleloom_core.tools.keyframe import anchor_prompt, frame_prompt

DETAIL = LookDetail(
    subject="hands and product only, no face in frame",
    lighting="soft warm key light, no hard shadow",
    texture="hydrogel sheen and visible moisture",
    palette="pale pink and white surfaces",
    lens="macro close focus, shallow depth of field",
    motion_feel="cuts fall on the action and the sound effect",
)


def _styled(style: StyleSchema, **look) -> StyleSchema:
    return style.model_copy(update={"look": style.look.model_copy(update=look)})


def _shot() -> Shot:
    return Shot(
        index=0, role="apply", duration_sec=1.0, shot_size="ECU",
        camera_move="static_locked_off_macro", action="손이 미스트를 분사한다",
        caption="", scene_prompt="x", motion_prompt="y",
    )


# --- the split -------------------------------------------------------------- #


def test_image_and_video_get_different_phrases():
    """`sfx-driven rhythm` in an image prompt is noise, and `shallow depth of field`
    in a motion prompt is already settled by the frame. One flat list could not tell
    them apart, which is why it went nowhere useful."""
    assert DETAIL.motion_feel not in DETAIL.image_phrases()
    assert DETAIL.subject not in DETAIL.video_phrases()
    assert len(DETAIL.image_phrases()) == 5
    assert DETAIL.video_phrases() == [DETAIL.motion_feel]


def test_a_style_without_detail_falls_back_to_its_keywords(style):
    """Re-extracting costs an LLM call and would discard hand corrections, so a
    style saved before the split still has to work."""
    old = _styled(style, detail=LookDetail(), keywords=["macro product texture", "x"])
    assert descriptive_phrases(old, for_image=True) == ["macro product texture", "x"]
    assert descriptive_phrases(old, for_image=False) == ["macro product texture", "x"]


def test_detail_wins_over_keywords_once_it_is_set(style):
    both = _styled(style, detail=DETAIL, keywords=["stale keyword"])
    assert "stale keyword" not in descriptive_phrases(both, for_image=True)


# --- measurements out of prompts -------------------------------------------- #


def test_no_prompt_carries_the_measured_colour_figures(style, ctx, brief):
    """QC reads saturation and contrast from style.json. Putting them in a prompt
    spent characters on numbers no model reads -- and in a 512-character budget they
    were the *last* thing dropped, displacing the phrases that describe the shot."""
    styled = _styled(style, detail=DETAIL)
    session = _session(ctx, styled, brief)
    cast = casting_tool.casting(ctx, session)

    # The formatted figures, not the words: a grade name is allowed to be called
    # `warm_high_contrast`, and that is language rather than a measurement.
    numbers = (
        f"{styled.look.saturation:.2f}",
        f"{styled.look.contrast:.2f}",
    )
    for text in (
        look_tokens(styled, for_image=True),
        look_tokens(styled, for_image=False),
        style_tokens(styled, cast),
        anchor_prompt(styled, cast),
        frame_prompt(styled, _shot(), brief),
    ):
        for number in numbers:
            assert number not in text, f"{number!r} leaked into: {text[:120]}"
    # The grade *name* is language, so it stays.
    assert styled.look.grade in look_tokens(styled, for_image=True)


def _session(ctx, style, brief):
    from styleloom_core.schema import RunInputs, RunRecord
    from styleloom_core.session import RunSession

    s = RunSession(
        record=RunRecord(run_id="t", style_id=style.style_id),
        inputs=RunInputs(text="미스트 제품 광고"), store=ctx.runs,
    )
    s.artifacts.update({"style": style, "brief": brief})
    return s


# --- the image prompt is where the description goes ------------------------- #


def test_the_frame_prompt_carries_every_image_dimension(ctx, style, brief):
    styled = _styled(style, detail=DETAIL)
    prompt = frame_prompt(styled, _shot(), brief)
    for phrase in DETAIL.image_phrases():
        assert phrase in prompt, f"missing from the frame prompt: {phrase}"


def test_the_frame_prompt_says_what_the_run_is_about(ctx, style, brief):
    """The storyboard leaves the topic out of its 512-character shot prompts on
    purpose. Here it is free, and without it the still drifts toward a generic
    version of the beat."""
    prompt = frame_prompt(_styled(style, detail=DETAIL), _shot(), brief)
    assert brief.topic in prompt
    # And it is still optional, for callers that have no brief to hand.
    assert brief.topic not in frame_prompt(style, _shot())


def test_the_anchor_leaves_out_what_would_fight_a_reference_portrait(ctx, style, brief):
    """`hands and product only, no face in frame` and `shallow depth of field` are
    correct about the reference and wrong for this one frame, whose whole job is a
    sharp, unambiguous portrait every later frame descends from."""
    prompt = anchor_prompt(_styled(style, detail=DETAIL), casting_tool.casting(
        ctx, _session(ctx, _styled(style, detail=DETAIL), brief)))
    assert DETAIL.lighting in prompt and DETAIL.palette in prompt
    assert DETAIL.subject not in prompt
    assert DETAIL.lens not in prompt


def test_the_image_prompt_uses_far_more_room_than_the_video_prompt_has(ctx, style, brief):
    """The whole direction of the fix. The image endpoint allows 2500 characters and
    was using ~210; the video endpoint allows 512 per shot and cannot be raised. So
    the description is established in the frame, not in the shot prompt."""
    specs = load_kling_specs(ctx.settings)
    image_limit = specs["text_to_image"]["kling-v3"]["max_prompt_chars"]
    video_limit = specs["image_to_video"]["kling-3.0"]["max_shot_prompt_chars"]

    assert image_limit > video_limit * 4, "the asymmetry is the reason for the design"

    styled = _styled(style, detail=DETAIL)
    cast = casting_tool.casting(ctx, _session(ctx, styled, brief))
    prompt = frame_prompt(styled, _shot(), brief)

    # Every image dimension is in the frame prompt, and it fits with room to spare.
    assert all(p in prompt for p in DETAIL.image_phrases())
    assert len(prompt) <= image_limit

    # And the shot prompt carries none of them: the frame it opens on already does,
    # and 512 characters are better spent on the beat and on how the shot moves.
    tokens = style_tokens(styled, cast)
    assert not any(p in tokens for p in DETAIL.image_phrases())


# --- what the shot prompt keeps under pressure ------------------------------ #


def test_motion_rides_in_the_motion_sentence_not_the_scene_tokens(ctx, style, brief):
    """A still cannot express cutting rhythm, so `motion_feel` has to reach the video
    prompt -- but not by competing with the presenter for the same 512 characters.

    It replaces the sentence that used to be sent instead: every video this system
    produced said "Natural continuous motion, no cut inside the shot", which is the
    filler that made the output read as generic. Similar length, actual content.
    """
    styled = _styled(style, detail=DETAIL)
    cast = casting_tool.casting(ctx, _session(ctx, styled, brief))

    assert DETAIL.motion_feel in motion_sentence(styled)
    assert DETAIL.motion_feel not in style_tokens(styled, cast)
    # The instruction half of the generic sentence is kept; the flavour is not.
    assert "No cut inside the shot" in motion_sentence(styled)
    assert "Natural continuous motion" not in motion_sentence(styled)
    # A style with no motion_feel still gets a usable sentence.
    assert motion_sentence(_styled(style, detail=LookDetail())) == BODY_MOTION


def test_compression_trims_clause_tails_before_dropping_a_whole_token(ctx, style, brief):
    """A location prompt runs past 110 characters: a head plus qualifiers. Trimming
    the tail keeps the place; dropping the token loses it entirely, and at 512
    characters that difference decides whether the grade survives at all."""
    styled = _styled(style, detail=DETAIL)
    cast = casting_tool.casting(ctx, _session(ctx, styled, brief))

    full = style_tokens(styled, cast)
    # A budget just under the full length: something must give, but nothing vanishes.
    squeezed = style_tokens(styled, cast, budget=len(full) - 20)

    assert len(squeezed) <= len(full) - 20 or len(squeezed) < len(full)
    assert cast.creator.prompt.split(",")[0] in squeezed
    assert cast.setting.prompt.split(",")[0].removeprefix("in ") in squeezed
    assert styled.look.grade in squeezed
