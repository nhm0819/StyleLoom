"""CLI entry point.

The harness. Deliberately thin: every command builds a `Context`, calls one core
function, and formats the result. Nothing here decides pipeline behaviour, which
is what makes an API or worker a peer of this file rather than a rewrite of it.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Annotated

import typer
from styleloom_core import Settings, build_plan
from styleloom_core import __version__ as core_version
from styleloom_core.media import resolve_font
from styleloom_core.tools import registry

from .commands import export as export_cmd
from .commands import hook as hook_cmd
from .commands import models as models_cmd
from .commands import run as run_cmd
from .commands import runs as runs_cmd
from .commands import style as style_cmd

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help=(
        "StyleLoom -- 레퍼런스 숏폼 영상의 스타일을 분해해 재사용 가능한 자산으로 만들고, "
        "임의의 입력을 같은 스타일의 영상으로 생성합니다."
    ),
)

app.add_typer(style_cmd.app, name="style")
app.add_typer(runs_cmd.app, name="runs")
app.add_typer(hook_cmd.app, name="hook")
# run / batch stay top level: they are the two verbs people actually type.
app.registered_commands += run_cmd.app.registered_commands
app.registered_commands += export_cmd.app.registered_commands
app.registered_commands += models_cmd.app.registered_commands


@app.command("plan")
def show_plan(
    no_qc: Annotated[bool, typer.Option("--no-qc")] = False,
) -> None:
    """실행될 파이프라인과 각 단계의 입출력 아티팩트를 출력합니다."""
    plan = build_plan(include_qc=not no_qc)
    typer.secho(f"plan: {plan.name}", bold=True)
    for line in plan.describe():
        typer.echo(f"  {line}")
    typer.secho(
        "\n순서는 데이터이고 실행 전에 검증됩니다. 단계가 자기보다 뒤에서 만들어지는 "
        "아티팩트를 읽으면 렌더링 전에 실패합니다.",
        fg="bright_black",
    )


@app.command("doctor")
def doctor(
    data_dir: Annotated[Path | None, typer.Option("--data-dir")] = None,
) -> None:
    """실행 전 환경 점검."""
    settings = Settings(data_dir=data_dir) if data_dir else Settings()

    def line(ok: bool, label: str, detail: str = "") -> None:
        typer.secho(f"{'ok  ' if ok else 'FAIL'}  {label:24s}{detail}",
                    fg="green" if ok else "red")

    typer.secho(f"styleloom-core {core_version}\n", bold=True)

    ffmpeg = shutil.which("ffmpeg")
    line(bool(ffmpeg), "ffmpeg", ffmpeg or "not on PATH -- rendering will fail")

    font = resolve_font()
    line(bool(font), "CJK font", font or "none found -- captions will be skipped")

    llm_provider = settings.resolved_llm_provider()
    video_provider = settings.resolved_video_provider()
    requested = f"  (requested: {settings.llm_provider}/{settings.video_provider})"

    line(True, "llm provider", llm_provider + (requested if "auto" in requested else ""))
    line(
        bool(settings.anthropic_api_key),
        "  anthropic key",
        "detected" if settings.anthropic_api_key
        else "not set -- ANTHROPIC_API_KEY or STYLELOOM_ANTHROPIC_API_KEY",
    )
    if settings.anthropic_api_key and llm_provider != "anthropic":
        typer.secho(
            "      key is present but llm_provider is pinned to "
            f"{settings.llm_provider!r}. Unset STYLELOOM_LLM_PROVIDER to use it.",
            fg="yellow",
        )

    line(True, "video provider", video_provider)
    # Both halves are reported separately: supplying one and forgetting the other
    # is the likely mistake, and "credentials missing" would not say which.
    line(
        bool(settings.kling_access_key),
        "  kling access key",
        "detected" if settings.kling_access_key else "not set -- KLING_ACCESS_KEY",
    )
    line(
        bool(settings.kling_secret_key),
        "  kling secret key",
        "detected" if settings.kling_secret_key else "not set -- KLING_SECRET_KEY",
    )
    if settings.kling_secret_key and video_provider != "kling":
        typer.secho(
            "      key is present but video_provider is pinned to "
            f"{settings.video_provider!r}. Unset STYLELOOM_VIDEO_PROVIDER to use it.",
            fg="yellow",
        )
    if video_provider == "kling":
        line(True, "  kling host", settings.kling_base_url)
        line(True, "  kling models", f"{settings.kling_t2i_model} / "
             f"{settings.kling_i2v_model} ({settings.kling_mode})")

    if llm_provider == "mock" or video_provider == "mock":
        typer.secho(
            "\nnote: a mock provider is active, so output will be structurally "
            "correct but not watchable. See README > Known limitations.",
            fg="yellow",
        )

    for label, path in (
        ("archetypes", settings.archetypes_path),
        ("kling models", settings.kling_models_path),
        ("casting pool", settings.casting_path),
    ):
        try:
            line(True, label, str(settings.resolve_config(path)))
        except Exception as exc:  # noqa: BLE001 - reported, not raised
            line(False, label, str(exc))

    line(True, "render mode", settings.render_mode)
    if settings.render_mode == "multi_shot":
        typer.secho(
            "      cuts land inside the model's output, so shot timing is measured "
            "rather than guaranteed -- check qc's cut_timing_drift.",
            fg="yellow",
        )
    line(True, "data dir", str(settings.data_dir.resolve()))
    line(True, "tools", ", ".join(registry.names()))


@app.command("version")
def version() -> None:
    """버전 출력."""
    from . import __version__ as cli_version

    typer.echo(f"styleloom-cli {cli_version} / styleloom-core {core_version}")


if __name__ == "__main__":
    app()
