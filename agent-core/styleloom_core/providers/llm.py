"""LLM providers.

  * `mock`      - no network, no key. Produces *randomised* structured output so
                  the pipeline, including the hook non-determinism proof, is
                  runnable and gradeable offline.
  * `anthropic` - real Messages API call, vision-capable for style analysis.

`task` is passed explicitly rather than sniffed from the prompt, so the mock can
dispatch without string matching on prose.
"""

from __future__ import annotations

import base64
import json
import random
import re
from typing import Any

import httpx

from ..config import Settings
from ..errors import LLMError
from .base import BaseLLM, Task

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"


def extract_json(text: str) -> dict[str, Any]:
    """Strip markdown fences and parse.

    "Return only JSON" is a request, not a guarantee, so fenced output and stray
    prose are both handled before giving up.
    """
    cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start == -1 or end <= start:
            raise LLMError(f"no JSON object in LLM response: {text[:200]}") from None
        try:
            return json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError as exc:
            raise LLMError(f"unparseable JSON from LLM: {exc}") from exc


class AnthropicLLM(BaseLLM):
    name = "anthropic"

    def __init__(self, api_key: str, model: str, timeout: float = 120.0) -> None:
        if not api_key:
            raise LLMError(
                "STYLELOOM_ANTHROPIC_API_KEY is required for llm_provider=anthropic"
            )
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def complete_json(
        self,
        task: Task,
        system: str,
        user: str,
        temperature: float = 0.7,
        images: list[bytes] | None = None,
    ) -> dict[str, Any]:
        content: list[dict[str, Any]] = []
        for img in images or []:
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": base64.b64encode(img).decode(),
                    },
                }
            )
        content.append({"type": "text", "text": user})

        try:
            resp = httpx.post(
                ANTHROPIC_URL,
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": self.model,
                    "max_tokens": 4000,
                    # "temperature": temperature,
                    "system": system
                    + "\n\nRespond with a single JSON object and nothing else.",
                    "messages": [{"role": "user", "content": content}],
                },
                timeout=self.timeout,
            )
        except httpx.HTTPError as exc:
            raise LLMError(f"anthropic request failed: {exc}") from exc

        if resp.status_code >= 400:
            raise LLMError(f"anthropic {resp.status_code}: {resp.text[:300]}")
        blocks = resp.json().get("content", [])
        text = "\n".join(b.get("text", "") for b in blocks if b.get("type") == "text")
        return extract_json(text)


# --------------------------------------------------------------------------- #

# One fragment set per archetype in configs/archetypes.yaml. Kept in sync with it
# so the mock's `archetype` label actually matches the shape of the text it
# returns -- otherwise the offline hook variety looks real but is not structural.
_HOOK_FRAGMENTS = {
    "question": [
        "아직도 {t} 이렇게 하세요?",
        "왜 아무도 {t} 얘긴 안 할까요?",
        "{t}, 진짜 맞을까요?",
    ],
    "reversal": [
        "{t}, 결론부터 말하면 반대였습니다",
        "이게 {t}의 마지막 모습입니다",
        "{t}를 3일 만에 접었습니다",
    ],
    "shock_number": [
        "{t}에 쓴 돈 470만원",
        "{t} 하나로 조회수 92만",
        "{t}, 실패율 87%",
    ],
    "empathy": [
        "{t} 앞에서 멈칫한 적 있죠",
        "저만 {t} 어려운 거 아니죠?",
        "{t} 때문에 새벽 3시",
    ],
    "visual_impact": [
        "(무자막) {t}의 첫 프레임",
        "말 없이 {t}부터 보여드립니다",
    ],
    "contradiction": [
        "{t}, 하면 할수록 손해입니다",
        "{t}는 다들 반대로 알고 계십니다",
    ],
}

_MOCK_VISUALS = [
    "급격한 스냅 줌 인",
    "손 클로즈업에서 얼굴로 틸트업",
    "정지 프레임에서 갑작스런 컷",
]


class MockLLM(BaseLLM):
    """Deterministic in structure, random in content.

    Uses a fresh `random.Random()` seeded from OS entropy, so repeated calls
    differ and the hook non-determinism is real offline rather than simulated.
    """

    name = "mock"

    def __init__(self) -> None:
        self.rng = random.Random()

    def complete_json(
        self,
        task: Task,
        system: str,
        user: str,
        temperature: float = 0.7,
        images: list[bytes] | None = None,
    ) -> dict[str, Any]:
        topic = self._topic(user)
        if task == "ingest":
            return {
                "topic": topic,
                "audience": "숏폼 시청자",
                "key_message": f"{topic}의 핵심을 30초 안에 전달한다",
                "facts": [f"{topic} 관련 포인트 {i}" for i in (1, 2, 3)],
            }
        if task == "outline":
            names = self._beat_names(user)
            per = self._per_beat(user)
            # Written to the budgets the prompt states, and padded up to them. A mock
            # that always answered well inside the limits would never exercise them:
            # the budget arithmetic, the caption/content split and the warning that
            # fires when a prompt has to be squeezed would all be untested offline,
            # and the first real test would be a paid run.
            content_max = self._limit(user, "CONTENT_MAX_CHARS", 200)
            caption_max = self._limit(user, "CAPTION_MAX_CHARS", 42)
            return {
                "payoff": f"{topic}에서 실제로 통한 방법",
                "beats": [
                    {
                        "name": n,
                        "intent": f"{n} 단계 목적",
                        "content": self._to_length(f"{topic} - {n} 구간 내용", content_max),
                        "caption": self._to_length(f"{topic} {n}", caption_max),
                        "duration_sec": per,
                    }
                    for n in names
                ],
            }
        if task == "hook_candidates":
            arch = self._archetype(user)
            pool = _HOOK_FRAGMENTS.get(arch, _HOOK_FRAGMENTS["question"])
            n = self._candidate_count(user)
            return {
                "candidates": [
                    {
                        "archetype": arch,
                        "text": self._to_length(
                            self.rng.choice(pool).format(t=topic),
                            self._limit(user, "TEXT_MAX_CHARS", 42),
                        ),
                        "visual": self._to_length(
                            self.rng.choice(_MOCK_VISUALS),
                            self._limit(user, "VISUAL_MAX_CHARS", 200),
                        ),
                        "context_fit": round(self.rng.uniform(0.55, 0.98), 3),
                        "style_fit": round(self.rng.uniform(0.55, 0.98), 3),
                        "novelty": round(self.rng.uniform(0.35, 0.99), 3),
                        "rationale": f"{arch} 아키타입으로 {topic} 맥락 진입",
                    }
                    for _ in range(n)
                ]
            }
        if task == "style_synthesis":
            return {
                "grade": self.rng.choice(
                    ["warm_high_contrast", "cool_desaturated", "neutral_clean"]
                ),
                "cut_style": self.rng.choice(["jump_cut", "match_cut", "hard_cut"]),
                "moves": ["handheld_micro_shake", "snap_zoom_in"],
                "voice_tone": "fast_energetic",
                "keywords": ["vertical 9:16", "social-native", "high clarity"],
                "notes": "mock synthesis - metrics only, no VLM captioning",
            }
        return {}

    @staticmethod
    def _topic(user: str) -> str:
        m = re.search(r"TOPIC:\s*(.+)", user)
        raw = m.group(1) if m else (user.strip().splitlines()[0] if user.strip() else "주제")
        # Real models return a short noun phrase; keep the mock comparable.
        return " ".join(raw.strip().split()[:3])[:18]

    @staticmethod
    def _archetype(user: str) -> str:
        m = re.search(r"ARCHETYPE:\s*([\w-]+)", user)
        return m.group(1) if m else "question"

    @staticmethod
    def _beat_names(user: str) -> list[str]:
        m = re.search(r"BEAT_NAMES:\s*\[(.*?)\]", user)
        if not m:
            return ["context", "turn", "payoff", "cta"]
        return [x.strip().strip("'\"") for x in m.group(1).split(",") if x.strip()]

    @staticmethod
    def _per_beat(user: str) -> float:
        m = re.search(r"SECONDS_PER_BEAT:\s*([\d.]+)", user)
        return float(m.group(1)) if m else 4.0

    @staticmethod
    def _limit(user: str, key: str, default: int) -> int:
        """A budget the caller stated in the prompt."""
        m = re.search(rf"{key}:\s*(\d+)", user)
        return int(m.group(1)) if m else default

    @staticmethod
    def _to_length(text: str, limit: int) -> str:
        """Obey a stated budget, on a word boundary.

        Only shortens. An earlier version padded up to the limit so that offline runs
        exercised the budget arithmetic -- but `text` and `caption` are burned on
        screen, and filling them meant repeating words in the finished video. The
        arithmetic is exercised by tests that hand the stages over-long text on
        purpose; the mock's job is to answer plausibly, and a real model told a limit
        writes short, not to the byte.
        """
        if limit <= 0 or len(text) <= limit:
            return text
        words, out = text.split(), ""
        for word in words:
            candidate = f"{out} {word}".strip()
            if len(candidate) > limit:
                break
            out = candidate
        return out or text[:limit]

    @staticmethod
    def _candidate_count(user: str) -> int:
        m = re.search(r"CANDIDATE_COUNT:\s*(\d+)", user)
        return int(m.group(1)) if m else 5


def build_llm(settings: Settings) -> BaseLLM:
    if settings.resolved_llm_provider() == "anthropic":
        return AnthropicLLM(settings.anthropic_api_key, settings.llm_model)
    return MockLLM()
