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
    through, so a 7s reference produced a 19s video."""
    session = make_session(ctx, style, text="주제")
    session.artifacts["brief"] = ingest_tool.ingest(ctx, session)
    result = outline_tool.outline(ctx, session)

    body_budget = max(style.total_duration - style.hook_style.window_sec, 6.0)
    assert sum(b.duration_sec for b in result.beats) == pytest.approx(body_budget, abs=0.6)


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
        assert style.look.grade in shot.image_prompt
        assert "9:16" in shot.image_prompt
        assert "No text overlay" in shot.motion_prompt


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
        assert current.image_prompt != previous.image_prompt


def test_distinct_frames_survives_a_single_move_style(ctx, style, brief, outline):
    """Camera moves cycle by index, so a style with only one move cannot rely on
    them to separate neighbouring cuts."""
    style.camera.moves = ["handheld_micro_shake"]
    session = make_session(ctx, style, text="주제")
    session.artifacts.update({"brief": brief, "outline": outline, "hook": _hook_result()})
    session.artifacts["casting"] = casting_tool.casting(ctx, session)
    board = storyboard_tool.storyboard(ctx, session)

    for previous, current in zip(board.shots, board.shots[1:], strict=False):
        assert current.image_prompt != previous.image_prompt
