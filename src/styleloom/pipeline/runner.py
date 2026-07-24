"""Pipeline orchestrator.

`ingest -> outline -> hook -> storyboard -> render -> post -> qc`, writing one
JSON artifact per stage into `data/runs/<run_id>/`. A run is therefore fully
inspectable after the fact, and a failed run shows exactly which stage broke.
"""

from __future__ import annotations

from pathlib import Path

from .. import storage
from ..providers.llm import get_llm
from ..providers.video import get_video_provider
from ..schema import RunRecord, RunStatus
from . import hook as hook_stage
from . import plan, post, prompt, qc, render

RECENT_WINDOW = 4


def recent_archetypes(style_id: str, limit: int = RECENT_WINDOW) -> list[str]:
    """Feeds the hook module's recency penalty."""
    picks = []
    for run in storage.list_runs():
        if run.style_id == style_id and run.hook_archetype:
            picks.append(run.hook_archetype)
        if len(picks) >= limit:
            break
    return picks


def execute(
    run_id: str,
    style_id: str,
    text: str = "",
    file_path: Path | None = None,
    bgm: Path | None = None,
    persona_ref: Path | None = None,
    language: str = "ko",
) -> RunRecord:
    run = storage.load_run(run_id)
    out_dir = storage.run_dir(run_id)

    def stage(name: str) -> None:
        storage.save_run(run.touch(status=RunStatus.RUNNING, stage=name))

    try:
        style = storage.load_style(style_id)
        llm = get_llm()
        video = get_video_provider()

        stage("ingest")
        brief = plan.ingest(llm, text=text, file_path=file_path, language=language)
        storage.save_artifact(run_id, "brief.json", brief)

        stage("outline")
        outline = plan.build_outline(llm, brief, style)
        storage.save_artifact(run_id, "outline.json", outline)

        stage("hook")
        hook_result = hook_stage.generate_hook(
            llm, brief, outline, style, recent_archetypes(style_id)
        )
        storage.save_artifact(run_id, "hook.json", hook_result)
        run.touch(
            hook_text=hook_result.selected.text,
            hook_archetype=hook_result.archetype_sampled,
        )

        stage("storyboard")
        storyboard = prompt.build_storyboard(style, hook_result, outline, brief.topic)
        storage.save_artifact(run_id, "storyboard.json", storyboard)

        stage("render")
        clips, errors = render.render_storyboard(
            video, storyboard, out_dir / "shots", persona_ref
        )
        if errors:
            storage.save_raw(run_id, "render_errors.json", {str(k): v for k, v in errors.items()})
        if not clips:
            raise RuntimeError(f"all shots failed to render: {errors}")

        stage("post")
        final = post.finish(
            clips, storyboard.shots, style.caption, out_dir, out_dir / "final.mp4", bgm
        )

        stage("qc")
        report = qc.evaluate(final, style, storyboard)
        storage.save_artifact(run_id, "qc_report.json", report)

        run.touch(
            status=RunStatus.DONE,
            stage="done",
            qc_score=report.score,
            artifacts={
                "final_video": str(final),
                "brief": str(out_dir / "brief.json"),
                "outline": str(out_dir / "outline.json"),
                "hook": str(out_dir / "hook.json"),
                "storyboard": str(out_dir / "storyboard.json"),
                "qc_report": str(out_dir / "qc_report.json"),
            },
        )
    except Exception as exc:
        run.touch(status=RunStatus.FAILED, error=f"{type(exc).__name__}: {exc}")

    storage.save_run(run)
    return run
