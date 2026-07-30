"""Test fixtures.

Note what is absent: the previous suite had to set `STYLELOOM_DATA_DIR` in the
environment *before importing the package*, because settings were a module-level
singleton bound at import time. Settings are now constructed and injected, so
fixtures are ordinary fixtures and tests can hold different configurations at
once.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from styleloom_core import Settings, build_context, extract_style
from styleloom_core.context import Context
from styleloom_core.events import ListSink
from styleloom_core.schema import (
    Beat,
    Brief,
    Casting,
    Outline,
    RunInputs,
    RunRecord,
    StyleSchema,
)
from styleloom_core.session import RunSession
from styleloom_core.tools import casting as casting_tool

# Small frames keep the ffmpeg work in these tests to roughly a second.
TEST_W, TEST_H, TEST_FPS = 240, 426, 24
REF_SEGMENT_SEC = 1.2
REF_COLOURS = ["red", "blue", "green", "orange", "purple", "teal"]


@pytest.fixture(autouse=True)
def _no_real_credentials(monkeypatch):
    """Keep the suite hermetic.

    Provider selection defaults to `auto`, so a real key in the developer's shell
    would otherwise silently point these tests at a paid API.
    """
    for var in (
        "ANTHROPIC_API_KEY",
        "STYLELOOM_ANTHROPIC_API_KEY",
        "FAL_KEY",
        "STYLELOOM_FAL_KEY",
    ):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        data_dir=tmp_path / "data",
        width=TEST_W,
        height=TEST_H,
        fps=TEST_FPS,
        llm_provider="mock",
        video_provider="mock",
    )


@pytest.fixture
def sink() -> ListSink:
    return ListSink()


@pytest.fixture
def ctx(settings: Settings, sink: ListSink) -> Context:
    return build_context(settings, events=sink)


@pytest.fixture(scope="session")
def ref_spec() -> dict:
    """What the reference fixture is built to contain, so tests can assert that
    extraction recovers it rather than hardcoding numbers twice."""
    return {
        "segment_sec": REF_SEGMENT_SEC,
        "shot_count": len(REF_COLOURS),
        "total_sec": len(REF_COLOURS) * REF_SEGMENT_SEC,
    }


@pytest.fixture(scope="session")
def reference_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Six hard cuts at 1.2s -- a plausible short-form pacing signature.

    Session-scoped: it is read-only and the ffmpeg encode is the slowest thing in
    the suite.
    """
    d = tmp_path_factory.mktemp("ref")
    segments = []
    for i, colour in enumerate(REF_COLOURS):
        seg = d / f"seg{i}.mp4"
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-f", "lavfi",
                "-i", f"color=c={colour}:s={TEST_W}x{TEST_H}:d={REF_SEGMENT_SEC}:r={TEST_FPS}",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", str(seg),
            ],
            check=True,
        )
        segments.append(seg)

    listing = d / "segments.txt"
    listing.write_text("\n".join(f"file '{s}'" for s in segments), encoding="utf-8")
    out = d / "reference.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "concat", "-safe", "0", "-i", str(listing),
            "-c", "copy", str(out),
        ],
        check=True,
    )
    return out


@pytest.fixture
def style(ctx: Context, reference_video: Path) -> StyleSchema:
    """A style extracted from the fixture, saved to the store."""
    extracted = extract_style(ctx, "fixture", [reference_video])
    ctx.styles.save(extracted)
    return extracted


@pytest.fixture
def brief() -> Brief:
    return Brief(
        topic="테스트 주제",
        audience="숏폼 시청자",
        key_message="핵심 메시지",
        facts=["사실 1", "사실 2"],
        language="ko",
    )


@pytest.fixture
def casting(ctx: Context, style: StyleSchema, brief: Brief) -> Casting:
    """A drawn creator and setting.

    `outline` and `hook` both size their prompts against the style tokens, and those
    carry the drawn creator and setting descriptions -- so how much room a generated
    sentence gets depends on this. Which is also why casting runs before both.
    """
    session = RunSession(
        record=RunRecord(run_id="fixture", style_id=style.style_id),
        inputs=RunInputs(text="테스트"),
        store=ctx.runs,
    )
    session.artifacts.update({"style": style, "brief": brief})
    return casting_tool.casting(ctx, session)


@pytest.fixture
def outline() -> Outline:
    return Outline(
        beats=[
            Beat(name="context", intent="상황", content="배경 설명", duration_sec=2.0),
            Beat(name="payoff", intent="결론", content="해결 방법", duration_sec=2.0),
        ],
        payoff="실제로 통한 방법",
    )
