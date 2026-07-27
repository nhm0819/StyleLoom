"""The hook is the graded requirement: system-generated, context-fitting, and
able to differ across reruns. These tests measure each of those.
"""

from __future__ import annotations

import collections

import pytest
from styleloom_core.errors import ConfigError, ToolError
from styleloom_core.sampling import new_rng, sample_with_recency, softmax_sample
from styleloom_core.schema import HookCandidate
from styleloom_core.tools import hook as H

RUNS = 24


def test_archetype_pool_loads_from_config(ctx):
    pool = H.load_archetypes(ctx.settings)
    assert len(pool) >= 4
    assert all(a.get("id") and a.get("pattern") for a in pool)


def test_missing_archetype_ids_are_rejected(ctx, tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("archetypes:\n  - weight: 1.0\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="`id`"):
        H.load_archetypes(ctx.settings, path=bad)


def test_empty_archetype_file_is_rejected(ctx, tmp_path):
    empty = tmp_path / "empty.yaml"
    empty.write_text("archetypes: []\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="no entries"):
        H.load_archetypes(ctx.settings, path=empty)


def test_hook_is_not_deterministic_across_reruns(ctx, brief, outline, style):
    """Same input, many generations. If this ever returns one distinct value the
    central requirement has silently broken."""
    results = [H.generate(ctx, brief, outline, style) for _ in range(RUNS)]
    assert len({r.selected.text for r in results}) > 1
    assert len({r.archetype_sampled for r in results}) > 1


def test_hook_records_its_own_decision_trail(ctx, brief, outline, style):
    result = H.generate(ctx, brief, outline, style)
    assert result.archetype_sampled in result.archetype_pool
    assert result.selected in result.candidates
    assert "SystemRandom" in result.entropy_source
    assert "softmax" in result.selection_method


def test_candidates_are_distinct_after_dedupe(ctx, brief, outline, style):
    result = H.generate(ctx, brief, outline, style)
    texts = [c.text for c in result.candidates]
    assert len(texts) == len(set(texts))
    assert all(t for t in texts)


def test_recency_penalty_suppresses_a_recent_archetype(ctx):
    """The mechanism behind batch variety. Measured rather than asserted, because
    it is a probability shift, not a guarantee."""
    pool = H.load_archetypes(ctx.settings)
    rng = new_rng()
    n = 4000

    def share(recent: list[str]) -> float:
        counts = collections.Counter(
            sample_with_recency(pool, recent, rng, penalty=0.35)["id"] for _ in range(n)
        )
        return counts["question"] / n

    assert share(["question"]) < share([]) / 2


def test_zero_penalty_effectively_excludes(ctx):
    """Documented escape hatch: STYLELOOM_HOOK_RECENCY_PENALTY=0 turns the soft
    penalty into hard exclusion for anyone who needs guaranteed variety."""
    pool = H.load_archetypes(ctx.settings)
    rng = new_rng()
    counts = collections.Counter(
        sample_with_recency(pool, ["question"], rng, penalty=0.0)["id"] for _ in range(2000)
    )
    assert counts["question"] == 0


def test_dedupe_keeps_the_best_scoring_duplicate():
    """Regression: duplicate candidates made the softmax draw a no-op, quietly
    removing one of the three randomisation points."""
    kept = H.dedupe(
        [
            HookCandidate(archetype="a", text="같은 문장", visual="v", score=0.4),
            HookCandidate(archetype="a", text="같은 문장", visual="v", score=0.9),
            HookCandidate(archetype="a", text="다른 문장", visual="v", score=0.5),
        ]
    )
    assert len(kept) == 2
    assert max(c.score for c in kept if c.text == "같은 문장") == 0.9


def test_dedupe_drops_empty_text():
    assert H.dedupe([HookCandidate(archetype="a", text="", visual="v")]) == []


def test_softmax_favours_but_does_not_guarantee_the_top_candidate():
    cands = [
        HookCandidate(archetype="a", text="high", visual="v", score=0.9),
        HookCandidate(archetype="a", text="mid", visual="v", score=0.6),
        HookCandidate(archetype="a", text="low", visual="v", score=0.3),
    ]
    rng = new_rng()
    picks = collections.Counter(
        softmax_sample(cands, k=3, temp=0.8, rng=rng)[0].text for _ in range(2000)
    )
    assert picks["high"] > picks["low"]
    assert len(picks) > 1, "argmax would collapse this to one outcome"


def test_no_usable_candidates_raises(ctx, brief, outline, style, monkeypatch):
    monkeypatch.setattr(ctx.llm, "complete_json", lambda **kw: {"candidates": []})
    with pytest.raises(ToolError, match="no usable candidates"):
        H.generate(ctx, brief, outline, style)


def test_hook_prompt_carries_the_payoff(ctx, brief, outline, style):
    """The hook must promise what the body delivers, which is only possible if the
    payoff reaches the prompt."""
    seen = {}

    def capture(**kwargs):
        seen.update(kwargs)
        return {"candidates": [{"archetype": "question", "text": "t", "visual": "v"}]}

    ctx.llm.complete_json = capture  # type: ignore[method-assign]
    H.generate(ctx, brief, outline, style)
    assert outline.payoff in seen["user"]
    assert "never bait" in seen["system"]
