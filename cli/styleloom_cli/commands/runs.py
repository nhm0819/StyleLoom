"""`styleloom runs ...` -- inspect what a run did.

Resuming one lives under `styleloom run resume`, next to the command that started
it, rather than here among the read-only inspection commands.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from styleloom_core import HookResult, StyleLoomError

from ..options import abort, make_context

app = typer.Typer(no_args_is_help=True, help="실행 이력을 조회합니다.")


@app.command("ls")
def list_runs(
    style: Annotated[str | None, typer.Option(help="스타일로 필터")] = None,
    limit: Annotated[int, typer.Option("-n", "--limit")] = 20,
    data_dir: Annotated[Path | None, typer.Option("--data-dir")] = None,
) -> None:
    """최근 실행 목록."""
    ctx = make_context(data_dir=data_dir, quiet=True)
    records = ctx.runs.list_records(style_id=style, limit=limit)
    if not records:
        typer.echo("(no runs yet)")
        return
    for r in records:
        colour = {"done": "green", "failed": "red"}.get(r.status.value, "yellow")
        typer.secho(
            f"{r.run_id}  {r.status.value:7s} {r.style_id:16s} "
            f"qc={str(r.qc_score):6s} [{r.hook_archetype or '-'}] {r.hook_text or ''}",
            fg=colour,
        )


@app.command("show")
def show(
    run_id: str,
    data_dir: Annotated[Path | None, typer.Option("--data-dir")] = None,
) -> None:
    """실행 1건의 상태와 산출물 경로."""
    ctx = make_context(data_dir=data_dir, quiet=True)
    try:
        record = ctx.runs.load(run_id)
    except StyleLoomError as exc:
        abort(str(exc))
        return
    typer.echo(record.model_dump_json(indent=2))


@app.command("hook")
def show_hook(
    run_id: str,
    data_dir: Annotated[Path | None, typer.Option("--data-dir")] = None,
) -> None:
    """이 실행의 훅 결정 과정 전체 (후보·점수·선택 방식)."""
    ctx = make_context(data_dir=data_dir, quiet=True)
    path = ctx.runs.dir_for(run_id) / "hook.json"
    if not path.exists():
        abort(f"no hook.json for run {run_id!r}")
        return
    result = HookResult.model_validate_json(path.read_text(encoding="utf-8"))

    typer.echo(f"pool     : {result.archetype_pool}")
    typer.secho(f"sampled  : {result.archetype_sampled}", fg="cyan")
    typer.echo(f"entropy  : {result.entropy_source}")
    typer.echo(f"selection: {result.selection_method}\n")
    for c in sorted(result.candidates, key=lambda c: c.score, reverse=True):
        mark = ">" if c.text == result.selected.text else " "
        typer.secho(f"{mark} {c.score:.4f}  {c.text}",
                    fg="bright_white" if mark == ">" else None,
                    bold=mark == ">")
