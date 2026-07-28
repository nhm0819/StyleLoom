"""`styleloom style ...` -- manage the reusable style assets."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from styleloom_core import StyleLoomError, StyleSchema, extract_style

from ..options import abort, make_context

app = typer.Typer(no_args_is_help=True, help="레퍼런스 영상에서 스타일을 추출·관리합니다.")


@app.command("extract")
def extract(
    style_id: Annotated[str, typer.Argument(help="저장할 스타일 ID")],
    refs: Annotated[
        list[Path],
        typer.Argument(
            help="레퍼런스 영상 (1개 이상). 파일명만 주면 <data-dir>/uploads/ 에서 "
            "찾고, 없으면 준 경로 그대로 씁니다 (절대경로 가능)."
        ),
    ],
    notes: Annotated[str, typer.Option(help="스타일에 남길 메모")] = "",
    data_dir: Annotated[Path | None, typer.Option("--data-dir")] = None,
    llm: Annotated[str | None, typer.Option(help="mock | anthropic")] = None,
    force: Annotated[bool, typer.Option("--force", help="기존 스타일 덮어쓰기")] = False,
) -> None:
    """레퍼런스 영상을 분해해 style.json을 만듭니다."""
    ctx = make_context(data_dir=data_dir, llm=llm, quiet=True)
    if ctx.styles.exists(style_id) and not force:
        abort(f"style {style_id!r} already exists. Use --force to overwrite.")

    try:
        style = extract_style(ctx, style_id, list(refs), notes=notes)
    except StyleLoomError as exc:
        abort(str(exc))
        return

    path = ctx.styles.save(style)
    typer.secho(f"saved {path}", fg="green")
    typer.echo(
        f"  avg_shot={style.pacing.avg_shot_sec}s  total={style.total_duration}s  "
        f"shots={style.pacing.shot_count}  bpm~{style.audio.bpm}"
    )
    typer.echo(
        f"  grade={style.look.grade}  sat={style.look.saturation}  "
        f"con={style.look.contrast}  warm={style.look.warmth}"
    )
    typer.echo(f"  hook: {style.hook_style.cut_count} cuts in "
               f"{style.hook_style.window_sec}s, {style.hook_style.shot_size}")
    typer.secho(
        "  추출기가 잘못 라벨링했다면 style.json을 직접 고치세요. "
        "계약은 스키마이고, 추출기가 아닙니다.",
        fg="bright_black",
    )


@app.command("ls")
def list_styles(
    data_dir: Annotated[Path | None, typer.Option("--data-dir")] = None,
) -> None:
    """저장된 스타일 목록."""
    ctx = make_context(data_dir=data_dir, quiet=True)
    ids = ctx.styles.list_ids()
    if not ids:
        typer.echo("(no styles yet -- run `styleloom style extract` first)")
        return
    for sid in ids:
        style = ctx.styles.load(sid)
        recent = ctx.history.recent_values(sid, "hook")
        typer.echo(
            f"{sid:24s} avg_shot={style.pacing.avg_shot_sec:<6} "
            f"dur={style.total_duration:<6} refs={len(style.source_refs)} "
            f"recent_hooks={recent or '-'}"
        )


@app.command("show")
def show(
    style_id: str,
    data_dir: Annotated[Path | None, typer.Option("--data-dir")] = None,
) -> None:
    """style.json 원문 출력."""
    ctx = make_context(data_dir=data_dir, quiet=True)
    try:
        style = ctx.styles.load(style_id)
    except StyleLoomError as exc:
        abort(str(exc))
        return
    typer.echo(style.model_dump_json(indent=2))


@app.command("set")
def set_style(
    style_id: str,
    path: Annotated[Path, typer.Argument(help="교체할 style.json 경로")],
    data_dir: Annotated[Path | None, typer.Option("--data-dir")] = None,
) -> None:
    """직접 편집한 style.json으로 교체합니다 (스키마 검증 후 저장)."""
    ctx = make_context(data_dir=data_dir, quiet=True)
    try:
        style = StyleSchema.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        abort(f"invalid style file: {exc}")
        return
    if style.style_id != style_id:
        abort(f"style_id mismatch: file says {style.style_id!r}, you said {style_id!r}")
    typer.secho(f"saved {ctx.styles.save(style)}", fg="green")


@app.command("history")
def choice_history(
    style_id: str,
    kind: Annotated[
        str | None, typer.Option(help="hook | creator | setting (기본: 전체)")
    ] = None,
    limit: Annotated[int, typer.Option("-n", "--limit")] = 30,
    data_dir: Annotated[Path | None, typer.Option("--data-dir")] = None,
) -> None:
    """이 스타일이 최근 사용한 훅·크리에이터·배경 이력.

    다음 실행의 recency penalty가 무엇을 감쇠시킬지 보여줍니다.
    """
    ctx = make_context(data_dir=data_dir, quiet=True)
    uses = (
        ctx.history.recent(style_id, kind, limit)
        if kind
        else ctx.history.all_recent(style_id, limit)
    )
    if not uses:
        typer.echo("(no history for this style yet)")
        return
    for use in uses:
        typer.echo(f"{use.at}  {use.kind:8s} {use.value:22s} {use.detail}")
    typer.secho(
        f"\n각 종류의 최근 {ctx.settings.hook_recency_window}개는 다음 실행에서 "
        f"가중치 x{ctx.settings.hook_recency_penalty}로 감쇠됩니다.",
        fg="bright_black",
    )
