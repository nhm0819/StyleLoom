"""Weighted sampling with a recency penalty, and stochastic top-k selection.

Both the hook archetype and the casting choices (creator, background) are "draw
from a pool, but avoid what this style used recently". That shared logic lives
here rather than in either tool, because tools must not import each other.

Entropy is always `secrets.SystemRandom`: OS-provided and unseedable, so a run can
never accidentally repeat a previous one.
"""

from __future__ import annotations

import math
import secrets
from pathlib import Path
from typing import Any, Protocol

import yaml

from .errors import ConfigError

Pool = list[dict[str, Any]]


def new_rng() -> secrets.SystemRandom:
    return secrets.SystemRandom()


ENTROPY_SOURCE = "secrets.SystemRandom (OS entropy, unseeded)"


def load_pool(path: Path, key: str) -> Pool:
    """Read one named list of `{id, weight, ...}` entries out of a YAML file."""
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    items = data.get(key) or []
    if not items:
        raise ConfigError(f"{path}: no entries under {key!r}")
    missing = [i for i, item in enumerate(items) if not item.get("id")]
    if missing:
        raise ConfigError(f"{path}: every entry under {key!r} needs an `id`")
    return items


def sample_with_recency(
    pool: Pool,
    recent: list[str],
    rng: secrets.SystemRandom,
    penalty: float = 0.35,
) -> dict[str, Any]:
    """Weighted draw, down-weighting anything used recently.

    The penalty is a multiplier rather than an exclusion, so a small pool still
    works and the result stays a probability shift rather than a rotation. Setting
    it to 0 makes the exclusion hard, which is the documented escape hatch for
    callers who need guaranteed variety.
    """
    weights: list[float] = []
    for item in pool:
        w = float(item.get("weight", 1.0))
        if item["id"] in recent:
            w *= penalty
        weights.append(max(w, 1e-6))

    total = sum(weights)
    pick = rng.random() * total
    acc = 0.0
    for item, w in zip(pool, weights, strict=True):
        acc += w
        if pick <= acc:
            return item
    return pool[-1]


class Scored(Protocol):
    score: float


def softmax_sample(
    candidates: list[Any],
    k: int,
    temp: float,
    rng: secrets.SystemRandom,
) -> tuple[Any, str]:
    """Sample from the top k by softmax over `.score`, rather than taking argmax.

    Argmax would make the candidate generation step decorative: five candidates
    and a deterministic pick is one candidate with extra cost.
    """
    ranked = sorted(candidates, key=lambda c: c.score, reverse=True)[: max(k, 1)]
    label = f"top_k_softmax(k={k}, T={temp})"
    if len(ranked) == 1:
        return ranked[0], f"{label} [single candidate]"

    exps = [math.exp(c.score / max(temp, 1e-6)) for c in ranked]
    total = sum(exps)
    pick = rng.random() * total
    acc = 0.0
    for cand, e in zip(ranked, exps, strict=True):
        acc += e
        if pick <= acc:
            return cand, label
    return ranked[-1], label
