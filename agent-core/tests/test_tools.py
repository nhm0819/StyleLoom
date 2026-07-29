"""Per-tool behaviour that is easy to break and hard to notice."""

from __future__ import annotations

import pytest
from styleloom_core.errors import ToolError
from styleloom_core.media import wrap_caption
from styleloom_core.schema import Beat, HookCandidate, HookResult, RunInputs, RunRecord
from styleloom_core.session import RunSession
from styleloom_core.tools import casting as casting_tool
from styleloom_core.tools import ingest as ingest_tool
from styleloom_core.tools import outline as outline_tool
from styleloom_core.tools import storyboard as storyboard_tool


def make_session(ctx, style, **inputs):
    session = RunSession(
        record=RunRecord(run_id="t1", style_id=style.style_id),
        inputs=RunInputs(**inputs),
        store=ctx.runs,
    )
    session.artifacts["style"] = style
    return session


# --- outline --------------------------------------------------------------- #


def test_outline_total_respects_the_reference_duration(ctx, style):
    """Regression: the outline passed the model's raw beat durations straight
    through, so a 7s reference produced a 19s video.

    Asserted against the reference rather than against the floor constant. The
    floor was 6.0s, which on a short-form reference is not a safety net but an
    override: a 7.25s style with a 3s hook needs a 4.25s body and got 6s, so the
    output ran 24% long and this test passed anyway because it was checking the
    override rather than the style.
    """
    session = make_session(ctx, style, text="주제")
    session.artifacts["brief"] = ingest_tool.ingest(ctx, session)
    result = outline_tool.outline(ctx, session)

    wanted = style.total_duration - style.hook_style.window_sec
    assert wanted > outline_tool.MIN_BODY_BUDGET, (
        "fixture no longer exercises the real path -- the floor is binding"
    )
    assert sum(b.duration_sec for b in result.beats) == pytest.approx(wanted, abs=0.6)


def test_the_body_budget_floor_stays_below_short_form_references(ctx, style):
    """The floor exists to stop a degenerate zero-length body, not to lengthen a
    real reference. Short-form runs 7-15s, so anything near 6s overrules most of
    them."""
    assert outline_tool.MIN_BODY_BUDGET <= 2.0


def test_outline_never_writes_the_hook(ctx, style):
    session = make_session(ctx, style, text="주제")
    session.artifacts["brief"] = ingest_tool.ingest(ctx, session)
    result = outline_tool.outline(ctx, session)
    assert "hook" not in [b.name for b in result.beats]
    assert result.payoff


def test_fit_to_budget_keeps_proportions():
    beats = [
        Beat(name="a", intent="", content="", duration_sec=1.0),
        Beat(name="b", intent="", content="", duration_sec=3.0),
    ]
    outline_tool.fit_to_budget(beats, 8.0)
    assert sum(b.duration_sec for b in beats) == pytest.approx(8.0, abs=0.1)
    assert beats[1].duration_sec == pytest.approx(3 * beats[0].duration_sec, rel=0.05)


def test_fit_to_budget_enforces_a_floor():
    beats = [Beat(name=n, intent="", content="", duration_sec=1.0) for n in "abcdefghij"]
    outline_tool.fit_to_budget(beats, 1.0)
    assert all(b.duration_sec >= 0.5 for b in beats)


# --- ingest ---------------------------------------------------------------- #


@pytest.mark.parametrize(
    "name,kind",
    [("a.mp4", "video"), ("a.MOV", "video"), ("a.jpg", "image"), ("a.png", "image")],
)
def test_ingest_classifies_by_suffix(name, kind):
    from pathlib import Path

    assert ingest_tool.classify(Path(name).suffix) == kind


def test_ingest_of_plain_text(ctx, style):
    session = make_session(ctx, style, text="회사에서 아무도 안 알려주는 엑셀 단축키")
    brief = ingest_tool.ingest(ctx, session)
    assert brief.input_kind == "text"
    assert brief.topic
    assert brief.language == "ko"


# --- storyboard ------------------------------------------------------------ #


def _hook_result() -> HookResult:
    selected = HookCandidate(
        archetype="question", text="훅 문장", visual="스냅 줌 인", score=0.8
    )
    return HookResult(
        archetype_pool=["question"],
        archetype_sampled="question",
        candidates=[selected],
        selected=selected,
        selection_method="test",
        temperature=0.9,
        entropy_source="test",
    )


def test_storyboard_opens_with_the_hook_window(ctx, style, brief, outline):
    session = make_session(ctx, style, text="주제")
    session.artifacts.update({"brief": brief, "outline": outline, "hook": _hook_result()})
    session.artifacts["casting"] = casting_tool.casting(ctx, session)
    board = storyboard_tool.storyboard(ctx, session)

    hook_shots = [s for s in board.shots if s.role == "hook"]
    assert len(hook_shots) == style.hook_style.cut_count
    assert sum(s.duration_sec for s in hook_shots) == pytest.approx(
        style.hook_style.window_sec, abs=0.1
    )
    # The caption lands once and holds, rather than re-animating on every cut.
    assert sum(1 for s in hook_shots if s.caption) == 1
    assert hook_shots[0].caption == "훅 문장"


def test_every_shot_prompt_carries_the_style_tokens(ctx, style, brief, outline):
    """Image and video models have no memory across calls, so a look described
    only in shot 1 has drifted by shot 6."""
    session = make_session(ctx, style, text="주제")
    session.artifacts.update({"brief": brief, "outline": outline, "hook": _hook_result()})
    session.artifacts["casting"] = casting_tool.casting(ctx, session)
    board = storyboard_tool.storyboard(ctx, session)

    assert len(board.shots) >= 3
    for shot in board.shots:
        # The grade rides in both forms: a continuation entry inherits the person
        # from the entry above it, but an unstated grade drifts.
        assert style.look.grade in shot.scene_prompt
        assert style.look.grade in shot.continuation_prompt
        # No framing token. `aspect_ratio` is an explicit request field under
        # text-to-video, so spelling "vertical 9:16" into a 512-character entry
        # spends characters restating what the API already knows.
        assert "vertical 9:16 framing" not in shot.scene_prompt
        # No burnt-in-caption instruction either: negative_prompt is one field per
        # request, against 47 characters in every one of six entries.
        assert "No text or captions" not in shot.motion_prompt
        assert "no text" not in shot.video_prompt.lower()


def test_a_continuation_entry_drops_identity_but_keeps_the_grade(
    ctx, style, brief, outline
):
    """Every entry in a multi-shot request is read by one generation, so the
    presenter and the room only need saying once. Repeating them cost 260 of the
    512 characters an entry gets."""
    session = make_session(ctx, style, text="주제")
    session.artifacts.update({"brief": brief, "outline": outline, "hook": _hook_result()})
    cast = casting_tool.casting(ctx, session)
    session.artifacts["casting"] = cast
    board = storyboard_tool.storyboard(ctx, session)

    for shot in board.shots:
        assert cast.creator.prompt not in shot.continuation_prompt
        assert "Same subject and location" in shot.continuation_prompt
        assert len(shot.continuation_video_prompt) < len(shot.video_prompt)


def test_the_camera_move_is_stated_once_per_entry(ctx, style, brief, outline):
    """Regression: the move sat in the scene line and again in the motion clause,
    so an entry read 'CU shot, static_phone_mount. ... static_phone_mount.'"""
    session = make_session(ctx, style, text="주제")
    session.artifacts.update({"brief": brief, "outline": outline, "hook": _hook_result()})
    session.artifacts["casting"] = casting_tool.casting(ctx, session)
    board = storyboard_tool.storyboard(ctx, session)

    for shot in board.shots:
        assert shot.video_prompt.count(shot.camera_move) == 1
        assert shot.continuation_video_prompt.count(shot.camera_move) == 1


def test_shot_indices_are_contiguous(ctx, style, brief, outline):
    session = make_session(ctx, style, text="주제")
    session.artifacts.update({"brief": brief, "outline": outline, "hook": _hook_result()})
    session.artifacts["casting"] = casting_tool.casting(ctx, session)
    board = storyboard_tool.storyboard(ctx, session)
    assert [s.index for s in board.shots] == list(range(len(board.shots)))


# --- session --------------------------------------------------------------- #


def test_missing_artifact_names_what_is_available(ctx, style):
    session = make_session(ctx, style)
    with pytest.raises(ToolError, match="plan"):
        session.get("outline", type(style))


def test_wrong_artifact_type_is_caught(ctx, style, brief):
    session = make_session(ctx, style)
    session.artifacts["outline"] = brief
    with pytest.raises(ToolError, match="expected"):
        session.get("outline", type(style))


# --- captions -------------------------------------------------------------- #


@pytest.mark.parametrize("width", [8, 14, 40])
def test_caption_wrap_respects_width(width):
    text = "회사에서 아무도 안 알려주는 엑셀 단축키 모음"
    for line in wrap_caption(text, width).splitlines():
        assert len(line) <= width or " " not in line


def test_caption_wrap_caps_line_count():
    long_text = " ".join(["단어"] * 60)
    assert len(wrap_caption(long_text, 6).splitlines()) <= 3


def test_caption_wrap_of_empty_text():
    assert wrap_caption("", 10) == ""


def test_adjacent_shots_do_not_ask_for_identical_images(ctx, style, brief, outline):
    """Regression: the camera move lived only in the motion prompt, so every cut
    inside a beat requested byte-identical imagery. A two-cut hook was one frame
    shown twice -- an invisible cut, which QC's drift check then could not see."""
    session = make_session(ctx, style, text="주제")
    session.artifacts.update({"brief": brief, "outline": outline, "hook": _hook_result()})
    session.artifacts["casting"] = casting_tool.casting(ctx, session)
    board = storyboard_tool.storyboard(ctx, session)

    for previous, current in zip(board.shots, board.shots[1:], strict=False):
        assert current.scene_prompt != previous.scene_prompt


def test_distinct_frames_survives_a_single_move_style(ctx, style, brief, outline):
    """Camera moves cycle by index, so a style with only one move cannot rely on
    them to separate neighbouring cuts."""
    style.camera.moves = ["handheld_micro_shake"]
    session = make_session(ctx, style, text="주제")
    session.artifacts.update({"brief": brief, "outline": outline, "hook": _hook_result()})
    session.artifacts["casting"] = casting_tool.casting(ctx, session)
    board = storyboard_tool.storyboard(ctx, session)

    for previous, current in zip(board.shots, board.shots[1:], strict=False):
        assert current.scene_prompt != previous.scene_prompt


# --- video ingest ----------------------------------------------------------- #
#
# Regression: a video input reached the model as a filename and a duration and
# nothing else, so `--file clip.mp4` produced the same brief as `--file
# anything.mp4`. The vision path existed and was only wired to stills.


class RecordingLLM:
    """Captures what ingest actually sends, which is the thing under test.

    The mock LLM ignores `images` -- correctly, it has no eyes -- so asserting on
    the returned Brief cannot distinguish frames being sent from frames being
    dropped. The call itself has to be inspected.
    """

    def __init__(self, inner):
        self.inner = inner
        self.calls: list[dict] = []

    def complete_json(self, task, system, user, temperature=0.7, images=None):
        self.calls.append({"task": task, "system": system, "user": user,
                           "images": images})
        return self.inner.complete_json(task, system, user, temperature, images)


def test_video_input_sends_frames_to_the_model(ctx, style, reference_video):
    ctx.llm = RecordingLLM(ctx.llm)
    session = make_session(ctx, style, file_path=reference_video, text="이 영상 소개해줘")

    brief = ingest_tool.ingest(ctx, session)

    assert brief.input_kind == "video"
    sent = ctx.llm.calls[0]["images"]
    assert sent is not None, "video input reached the model with no frames"
    assert len(sent) == ingest_tool.VIDEO_FRAME_SAMPLES
    # JPEG magic. The API is told media_type image/jpeg, so anything else is a
    # 400 that only shows up with a real key.
    assert all(frame.startswith(b"\xff\xd8\xff") for frame in sent)


def test_the_prompt_says_the_stills_are_video_frames(ctx, style, reference_video):
    """Without it the model sees three unrelated photos and averages them."""
    ctx.llm = RecordingLLM(ctx.llm)
    session = make_session(ctx, style, file_path=reference_video)

    ingest_tool.ingest(ctx, session)

    user = ctx.llm.calls[0]["user"]
    assert "frames from this video" in user
    assert "beginning, middle and end" in user


def test_frames_are_small_enough_to_send(ctx, style, reference_video):
    """Nothing between here and the API resizes, so oversized frames are paid for
    on every ingest call."""
    ctx.llm = RecordingLLM(ctx.llm)
    ingest_tool.ingest(ctx, make_session(ctx, style, file_path=reference_video))

    for frame in ctx.llm.calls[0]["images"]:
        assert len(frame) < 400_000


def test_an_undecodable_video_degrades_instead_of_failing(ctx, style, tmp_path):
    """A run that cannot read frames should still produce a brief from the text."""
    fake = tmp_path / "broken.mp4"
    fake.write_bytes(b"not a video")
    ctx.llm = RecordingLLM(ctx.llm)
    session = make_session(ctx, style, file_path=fake, text="주제")

    brief = ingest_tool.ingest(ctx, session)

    assert brief.input_kind == "video"
    assert ctx.llm.calls[0]["images"] is None
    assert "no stills are attached" in ctx.llm.calls[0]["user"]


# --- prompt length budget --------------------------------------------------- #
#
# Regression: a storyboard entry is capped at 512 characters while the top-level
# prompt allows thousands, and every prompt this system builds was 590-700. Every
# multi_shot request would have been rejected -- on the paid API, since the
# offline provider declared no limit and so never exercised the budget.


def test_multi_shot_prompts_fit_the_storyboard_entry_limit(ctx, style, brief, outline):
    session = make_session(ctx, style, text="겨울철 건조한 피부 관리 루틴 총정리")
    session.artifacts.update({"brief": brief, "outline": outline, "hook": _hook_result()})
    session.artifacts["casting"] = casting_tool.casting(ctx, session)
    ctx.settings.render_mode = "multi_shot"

    board = storyboard_tool.storyboard(ctx, session)

    limit = ctx.video.max_shot_prompt_chars
    assert limit > 0, "the offline provider must declare the limit or this never runs"
    for shot in board.shots:
        assert len(shot.video_prompt) <= limit, f"shot {shot.index} is over by "\
            f"{len(shot.video_prompt) - limit}"


def test_per_shot_prompts_are_not_squeezed(ctx, style, brief, outline):
    """The budget is a multi-shot constraint. Applying it everywhere would throw
    away style context the top-level field has room for.

    Asserted as "the full token string survives intact" rather than by comparing
    two builds: the hook is sampled, so two storyboards of the same style differ
    in length for reasons that have nothing to do with the budget.
    """
    session = make_session(ctx, style, text="겨울철 건조한 피부 관리 루틴 총정리")
    session.artifacts.update({"brief": brief, "outline": outline, "hook": _hook_result()})
    cast = casting_tool.casting(ctx, session)
    session.artifacts["casting"] = cast
    ctx.settings.render_mode = "per_shot"

    board = storyboard_tool.storyboard(ctx, session)

    full = storyboard_tool.style_tokens(style, cast)
    assert all(full in shot.scene_prompt for shot in board.shots)


def test_the_budget_drops_whole_clauses_from_the_tail(ctx, style, brief):
    """Identity and grade are what QC measures, so they outrank the descriptive
    keywords. Truncating mid-token would hand the model half a sentence."""
    session = make_session(ctx, style, text="주제")
    session.artifacts["brief"] = brief
    cast = casting_tool.casting(ctx, session)

    full = storyboard_tool.style_tokens(style, cast)
    squeezed = storyboard_tool.style_tokens(style, cast, budget=200)

    assert len(squeezed) <= 200
    assert full.startswith(squeezed.split(",")[0])
    assert cast.creator.prompt in squeezed          # identity survives
    assert not squeezed.endswith(",")               # no half-clause
    assert all(part.strip() in full for part in squeezed.split(","))
