"""Stage 4 - Hook generation. The graded core of the system.

Requirement: the hook must be produced by the system (not typed by the user),
must fit the whole video's context, and must be able to differ across reruns of
the same input.

Relying on LLM `temperature` alone would satisfy that only by accident. So
variation is engineered at three independent points:

  1. Archetype sampling  - weighted draw from configs/archetypes.yaml, with a
                           recency penalty read from previous runs of the same
                           style. Different archetype => structurally different
                           hook, not a paraphrase.
  2. Candidate generation - N candidates, temperature 0.9, no fixed seed.
  3. Stochastic selection - top-k softmax sample instead of argmax, so the best
                           candidate is favoured but not guaranteed.

Every one of those decisions is written to `hook.json` so the non-determinism is
auditable rather than asserted.
"""

from __future__ import annotations

import math
import secrets
from pathlib import Path

import yaml

from ..config import settings
from ..providers.llm import BaseLLM
from ..schema import Brief, HookCandidate, HookResult, Outline, StyleSchema

BUNDLED_ARCHETYPES = Path(__file__).resolve().parents[3] / "configs" / "archetypes.yaml"

SCORE_WEIGHTS = {"context_fit": 0.5, "style_fit": 0.3, "novelty": 0.2}


def archetypes_path() -> Path:
    """STYLELOOM_ARCHETYPES_PATH wins, so a different person can drop in their
    own archetype set without touching the repo."""
    return settings.archetypes_path if settings.archetypes_path.exists() else BUNDLED_ARCHETYPES


def load_archetypes(path: Path | None = None) -> list[dict]:
    src = path or archetypes_path()
    data = yaml.safe_load(src.read_text(encoding="utf-8"))
    items = data.get("archetypes", [])
    if not items:
        raise ValueError(f"no archetypes defined in {src}")
    return items


def sample_archetype(
    archetypes: list[dict],
    recent: list[str],
    rng: secrets.SystemRandom,
    penalty: float = 0.35,
) -> dict:
    """Weighted draw. An archetype used in the last N runs is down-weighted so a
    batch of three videos does not open the same way three times."""
    weights = []
    for a in archetypes:
        w = float(a.get("weight", 1.0))
        if a["id"] in recent:
            w *= penalty
        weights.append(max(w, 1e-6))

    total = sum(weights)
    pick = rng.random() * total
    acc = 0.0
    for archetype, w in zip(archetypes, weights):
        acc += w
        if pick <= acc:
            return archetype
    return archetypes[-1]


def _softmax_sample(
    candidates: list[HookCandidate], k: int, temp: float, rng: secrets.SystemRandom
) -> tuple[HookCandidate, str]:
    ranked = sorted(candidates, key=lambda c: c.score, reverse=True)[: max(k, 1)]
    if len(ranked) == 1:
        return ranked[0], f"top_k_softmax(k={k}, T={temp}) [single candidate]"
    exps = [math.exp(c.score / max(temp, 1e-6)) for c in ranked]
    total = sum(exps)
    pick = rng.random() * total
    acc = 0.0
    for cand, e in zip(ranked, exps):
        acc += e
        if pick <= acc:
            return cand, f"top_k_softmax(k={k}, T={temp})"
    return ranked[-1], f"top_k_softmax(k={k}, T={temp})"


def generate_hook(
    llm: BaseLLM,
    brief: Brief,
    outline: Outline,
    style: StyleSchema,
    recent_archetypes: list[str] | None = None,
    candidate_count: int | None = None,
) -> HookResult:
    rng = secrets.SystemRandom()  # OS entropy: not seedable, never repeats a run
    archetypes = load_archetypes()
    n = candidate_count or settings.hook_candidate_count

    chosen = sample_archetype(archetypes, recent_archetypes or [], rng)

    body_summary = " / ".join(f"[{b.name}] {b.content}" for b in outline.beats)
    parsed = llm.complete_json(
        task="hook_candidates",
        system=(
            f"You write the first {style.hook_style.window_sec:.0f} seconds of short-form "
            f"videos, in {brief.language}. The hook must promise exactly what the body "
            "delivers -- never bait something the video does not pay off.\n"
            "Return JSON: candidates = list of "
            "{archetype, text, visual, context_fit, style_fit, novelty, rationale}. "
            "context_fit / style_fit / novelty are floats in [0,1]. "
            "`text` is the on-screen caption, max 16 characters per line. "
            "`visual` describes what is on screen, in one sentence."
        ),
        user=(
            f"ARCHETYPE: {chosen['id']}\n"
            f"ARCHETYPE_RULE: {chosen.get('pattern', '')}\n"
            f"ARCHETYPE_EXAMPLES: {chosen.get('fewshot', [])}\n"
            f"CANDIDATE_COUNT: {n}\n\n"
            f"TOPIC: {brief.topic}\n"
            f"KEY_MESSAGE: {brief.key_message}\n"
            f"AUDIENCE: {brief.audience}\n"
            f"BODY_BEATS: {body_summary}\n"
            f"PAYOFF_THE_HOOK_MUST_PROMISE: {outline.payoff}\n\n"
            f"STYLE_SHOT_SIZE: {style.hook_style.shot_size}\n"
            f"STYLE_CUTS_IN_WINDOW: {style.hook_style.cut_count}\n"
            f"STYLE_CAMERA_MOVES: {style.camera.moves}\n"
            f"STYLE_TONE: {style.audio.voice_tone}\n"
        ),
        temperature=settings.hook_temperature,
    )

    candidates: list[HookCandidate] = []
    for c in parsed.get("candidates", []):
        score = sum(float(c.get(k, 0.5)) * w for k, w in SCORE_WEIGHTS.items())
        candidates.append(
            HookCandidate(
                archetype=str(c.get("archetype", chosen["id"])),
                text=str(c.get("text", "")).strip(),
                visual=str(c.get("visual", "")).strip(),
                score=round(score, 4),
                rationale=str(c.get("rationale", "")),
            )
        )
    # Duplicates make the softmax draw meaningless -- keep the best of each text.
    best_by_text: dict[str, HookCandidate] = {}
    for c in candidates:
        if not c.text:
            continue
        if c.text not in best_by_text or c.score > best_by_text[c.text].score:
            best_by_text[c.text] = c
    candidates = list(best_by_text.values())
    if not candidates:
        raise ValueError("hook generation produced no usable candidates")

    selected, method = _softmax_sample(
        candidates, settings.hook_top_k, settings.hook_softmax_temp, rng
    )

    return HookResult(
        archetype_pool=[a["id"] for a in archetypes],
        archetype_sampled=chosen["id"],
        candidates=candidates,
        selected=selected,
        selection_method=method,
        temperature=settings.hook_temperature,
        entropy_source="secrets.SystemRandom (OS entropy, unseeded)",
    )
