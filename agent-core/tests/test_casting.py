"""Casting: the creator and background variation the assignment asks for.

What has to hold: the choices are made by the system rather than supplied by the
user, they can differ across reruns of the same input, they reach every shot
prompt, and a reference portrait is only generated where it can actually be used.
"""

from __future__ import annotations

import collections

import pytest
from styleloom_core.errors import ConfigError
from styleloom_core.sampling import load_pool, new_rng, sample_with_recency
from styleloom_core.schema import (
    HookCandidate,
    HookResult,
    RunInputs,
    RunRecord,
)
from styleloom_core.session import RunSession
from styleloom_core.tools import casting as C
from styleloom_core.tools import storyboard as storyboard_tool

RUNS = 24


def make_session(ctx, style, brief, **inputs):
    session = RunSession(
        record=RunRecord(run_id="cast_test", style_id=style.style_id),
        inputs=RunInputs(**inputs),
        store=ctx.runs,
    )
    session.artifacts["style"] = style
    session.artifacts["brief"] = brief
    return session


# --- pools ------------------------------------------------------------------ #


def test_both_pools_load_and_are_usable(ctx):
    path = ctx.settings.resolve_config(ctx.settings.casting_path)
    creators = load_pool(path, C.CREATOR_KEY)
    settings_pool = load_pool(path, C.SETTING_KEY)

    assert len(creators) >= 3 and len(settings_pool) >= 3
    for entry in [*creators, *settings_pool]:
        assert entry.get("id") and entry.get("prompt")


def test_a_pool_without_ids_is_rejected(ctx, tmp_path):
    bad = tmp_path / "casting.yaml"
    bad.write_text("creators:\n  - weight: 1.0\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="`id`"):
        load_pool(bad, C.CREATOR_KEY)


def test_a_missing_pool_key_is_rejected(ctx, tmp_path):
    bad = tmp_path / "casting.yaml"
    bad.write_text("creators:\n  - id: x\n    prompt: y\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="no entries"):
        load_pool(bad, C.SETTING_KEY)


# --- the choices are the system's, and they vary ---------------------------- #


def test_casting_is_not_deterministic_across_reruns(ctx, style, brief):
    """Same input, many runs. Creator and location both have to move, or the three
    submitted videos are one video with different captions."""
    session = make_session(ctx, style, brief)
    creators, settings_ = set(), set()
    for _ in range(RUNS):
        result = C.casting(ctx, session)
        creators.add(result.creator.id)
        settings_.add(result.setting.id)

    assert len(creators) > 1
    assert len(settings_) > 1


def test_casting_ignores_the_user_input(ctx, style, brief):
    """The creator is cast by the system. Nothing the user typed selects it."""
    session = make_session(ctx, style, brief, text="20대 남성 캐주얼로 해줘")
    ids = {C.casting(ctx, session).creator.id for _ in range(12)}
    assert len(ids) > 1, "input text must not pin the casting"


def test_casting_records_the_pools_it_drew_from(ctx, style, brief):
    result = C.casting(ctx, make_session(ctx, style, brief))
    assert result.creator.id in result.creator_pool
    assert result.setting.id in result.setting_pool
    assert "SystemRandom" in result.entropy_source


def test_recency_penalty_applies_to_creators(ctx):
    path = ctx.settings.resolve_config(ctx.settings.casting_path)
    pool = load_pool(path, C.CREATOR_KEY)
    first = pool[0]["id"]
    rng = new_rng()
    n = 4000

    def share(recent: list[str]) -> float:
        counts = collections.Counter(
            sample_with_recency(pool, recent, rng, penalty=0.35)["id"] for _ in range(n)
        )
        return counts[first] / n

    assert share([first]) < share([]) / 2


def test_casting_writes_both_kinds_to_history(ctx, style, brief):
    C.casting(ctx, make_session(ctx, style, brief))
    assert len(ctx.history.recent_values(style.style_id, "creator")) == 1
    assert len(ctx.history.recent_values(style.style_id, "setting")) == 1


def test_history_is_recorded_before_rendering(ctx, style, brief):
    """A run that dies during render still consumed its casting, so the next run in
    a batch must see it. Otherwise a failing batch repeats the same cast."""
    session = make_session(ctx, style, brief)
    result = C.casting(ctx, session)
    recorded = ctx.history.recent_values(style.style_id, "creator")
    assert recorded == [result.creator.id]


# --- casting reaches the prompts -------------------------------------------- #


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


def test_every_shot_prompt_carries_the_creator_and_the_setting(ctx, style, brief, outline):
    """Same reasoning as the style tokens: models have no memory across calls, so a
    presenter named once is a different person by the third cut."""
    session = make_session(ctx, style, brief)
    cast = C.casting(ctx, session)
    session.artifacts.update({"outline": outline, "hook": _hook_result(), "casting": cast})

    board = storyboard_tool.storyboard(ctx, session)
    assert len(board.shots) >= 3

    creator_head = cast.creator.prompt.split(",")[0]
    setting_head = cast.setting.prompt.split(",")[0]
    for shot in board.shots:
        assert creator_head in shot.image_prompt
        assert setting_head in shot.image_prompt


def test_casting_tokens_lead_the_prompt(ctx, style, brief):
    """Identity is what viewers notice breaking, so it must not sit at the tail of a
    long prompt where image models weight it least."""
    session = make_session(ctx, style, brief)
    cast = C.casting(ctx, session)
    tokens = storyboard_tool.style_tokens(style, cast)
    assert tokens.startswith(cast.creator.prompt.split(",")[0])
    assert tokens.index(cast.setting.prompt.split(",")[0]) < tokens.index("colour grade")


# --- the reference portrait -------------------------------------------------- #


def test_no_portrait_is_generated_for_a_provider_that_ignores_it(ctx, style, brief):
    """The offline renderer cannot honour an identity reference, so paying for a
    portrait would produce a file nothing consumes."""
    assert ctx.video.supports_persona is False
    result = C.casting(ctx, make_session(ctx, style, brief))
    assert result.creator_ref is None


def test_a_portrait_is_generated_when_the_provider_can_use_it(ctx, style, brief, monkeypatch):
    monkeypatch.setattr(type(ctx.video), "supports_persona", property(lambda self: True))
    result = C.casting(ctx, make_session(ctx, style, brief))
    assert result.creator_ref is not None
    assert result.creator_ref.exists()


def test_the_portrait_prompt_excludes_the_scene(ctx, style, brief):
    """It exists to fix identity. Baking in a background would fight the per-shot
    setting prompt instead of supporting it."""
    cast = C.casting(ctx, make_session(ctx, style, brief))
    prompt = C.creator_portrait_prompt(cast.creator, style)
    assert "neutral seamless background" in prompt
    assert cast.setting.prompt.split(",")[0] not in prompt


def test_an_explicit_persona_suppresses_the_generated_portrait(
    ctx, style, brief, tmp_path, monkeypatch
):
    """If a caller hands us a specific person, a cast stand-in is not what they
    asked for."""
    monkeypatch.setattr(type(ctx.video), "supports_persona", property(lambda self: True))
    mine = tmp_path / "me.jpg"
    mine.write_bytes(b"not-a-real-jpeg")
    result = C.casting(ctx, make_session(ctx, style, brief, persona_ref=mine))
    assert result.creator_ref is None


def test_a_failed_portrait_degrades_instead_of_failing_the_run(ctx, style, brief, monkeypatch):
    monkeypatch.setattr(type(ctx.video), "supports_persona", property(lambda self: True))

    def boom(*a, **kw):
        raise RuntimeError("t2i unavailable")

    monkeypatch.setattr(ctx.video, "keyframe", boom)
    result = C.casting(ctx, make_session(ctx, style, brief))
    assert result.creator_ref is None
    assert result.creator.id  # the cast itself still happened
