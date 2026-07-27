"""Shared option handling.

Settings come from the environment and `.env`; these flags override them for one
invocation. Keeping the override in one place means every command accepts the
same names.
"""

from __future__ import annotations

from pathlib import Path

import typer
from styleloom_core import Settings, StyleLoomError, build_context
from styleloom_core.context import Context

from .console import ConsoleSink


def make_context(
    data_dir: Path | None = None,
    llm: str | None = None,
    video: str | None = None,
    render_mode: str | None = None,
    quiet: bool = False,
    show_run_id: bool = False,
) -> Context:
    """Build a core Context, turning core errors into clean CLI exits.

    A traceback is the right output for a bug and the wrong output for a missing
    API key, so deliberate errors are reported as messages.
    """
    overrides: dict[str, object] = {}
    if data_dir is not None:
        overrides["data_dir"] = data_dir
    if llm is not None:
        overrides["llm_provider"] = llm
    if video is not None:
        overrides["video_provider"] = video
    if render_mode is not None:
        overrides["render_mode"] = render_mode

    try:
        settings = Settings(**overrides)  # type: ignore[arg-type]
        ctx = build_context(
            settings, events=ConsoleSink(quiet=quiet, show_run_id=show_run_id)
        )
        if not quiet:
            # Printed before any work: discovering afterwards that a run used the
            # offline mock is the single most expensive misunderstanding here.
            typer.secho(
                f"providers: {settings.provider_summary()} "
                f"render={settings.render_mode}",
                fg="bright_black",
            )
        return ctx
    except StyleLoomError as exc:
        typer.secho(f"error: {exc}", fg="red", err=True)
        raise typer.Exit(2) from exc


def abort(message: str, code: int = 1) -> None:
    typer.secho(f"error: {message}", fg="red", err=True)
    raise typer.Exit(code)
