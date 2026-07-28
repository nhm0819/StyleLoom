"""`styleloom run` / `styleloom batch` -- input to video."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from styleloom_core import (
    RunInputs,
    RunStatus,
    StyleLoomError,
    build_plan,
    run_batch,
    run_once,
)

from ..options import abort, make_context

app = typer.Typer(no_args_is_help=True)


def _report(records: list, total: int) -> None:
    ok = [r for r in records if r.status is RunStatus.DONE]
    typer.echo()
    typer.secho(f"{len(ok)}/{total} done", fg="green" if len(ok) == total else "yellow")
    for r in records:
        if r.status is RunStatus.DONE:
            typer.echo(
                f"  {r.run_id}  [{r.hook_archetype}] {r.hook_text}  "
                f"qc={r.qc_score}  {r.artifacts.get('final_video', '')}"
            )
        else:
            typer.secho(f"  {r.run_id}  FAILED at {r.stage}: {r.error}", fg="red")
    if len(ok) < total:
        raise typer.Exit(1)


@app.command("run")
def run(
    style_id: Annotated[str, typer.Argument(help="사용할 스타일 ID")],
    text: Annotated[str, typer.Option("--text", "-t", help="텍스트 입력")] = "",
    file: Annotated[Path | None, typer.Option("--file", "-f",
                    help="이미지 또는 영상 입력. 파일명만 주면 "
                         "<data-dir>/uploads/ 에서 찾습니다")] = None,
    bgm: Annotated[Path | None, typer.Option(help="배경음악 (uploads/ 조회 동일)")] = None,
    persona: Annotated[Path | None, typer.Option(
        help="크리에이터 참조 이미지 (Kling v3 계열에서만 반영, "
             "uploads/ 조회 동일)")] = None,
    lang: Annotated[str, typer.Option(help="출력 언어")] = "ko",
    no_qc: Annotated[bool, typer.Option("--no-qc", help="정합성 검사 생략")] = False,
    data_dir: Annotated[Path | None, typer.Option("--data-dir")] = None,
    llm: Annotated[str | None, typer.Option(help="mock | anthropic")] = None,
    video: Annotated[str | None, typer.Option(help="mock | fal")] = None,
    render_mode: Annotated[
        str | None,
        typer.Option(
            "--render-mode",
            help="per_shot (기본, 검증됨) | multi_shot (샷당 하한 없음, 비용 절감)",
        ),
    ] = None,
) -> None:
    """입력 1개로 숏폼 영상 1편을 생성합니다."""
    if not text and file is None:
        abort("need --text or --file")
    ctx = make_context(data_dir=data_dir, llm=llm, video=video, render_mode=render_mode)
    inputs = RunInputs(
        text=text, file_path=file, bgm=bgm, persona_ref=persona, language=lang
    )
    try:
        record = run_once(ctx, style_id, inputs, plan=build_plan(include_qc=not no_qc))
    except StyleLoomError as exc:
        abort(str(exc))
        return
    _report([record], 1)


@app.command("batch")
def batch(
    style_id: Annotated[str, typer.Argument(help="사용할 스타일 ID")],
    text: Annotated[list[str] | None, typer.Option("--text", "-t",
                    help="텍스트 입력 (여러 번 반복 가능)")] = None,
    file: Annotated[list[Path] | None, typer.Option("--file", "-f",
                    help="파일 입력 (여러 번 반복 가능). 파일명만 주면 "
                         "<data-dir>/uploads/ 에서 찾습니다")] = None,
    inputs_file: Annotated[Path | None, typer.Option("--inputs-file",
                           help="한 줄에 하나씩 입력이 적힌 텍스트 파일")] = None,
    lang: Annotated[str, typer.Option(help="출력 언어")] = "ko",
    no_qc: Annotated[bool, typer.Option("--no-qc")] = False,
    data_dir: Annotated[Path | None, typer.Option("--data-dir")] = None,
    llm: Annotated[str | None, typer.Option(help="mock | anthropic")] = None,
    video: Annotated[str | None, typer.Option(help="mock | fal")] = None,
    render_mode: Annotated[
        str | None,
        typer.Option(
            "--render-mode",
            help="per_shot (기본, 검증됨) | multi_shot (샷당 하한 없음, 비용 절감)",
        ),
    ] = None,
) -> None:
    """서로 다른 입력 여러 개를 같은 시스템에 통과시킵니다.

    같은 파이프라인, 다른 입력, 다른 결과 -- 이게 제출물이 증명해야 하는 명제입니다.
    """
    items: list[RunInputs] = []
    for t in text or []:
        items.append(RunInputs(text=t, language=lang))
    for f in file or []:
        items.append(RunInputs(file_path=f, language=lang))
    if inputs_file is not None:
        try:
            lines = inputs_file.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            abort(f"cannot read {inputs_file}: {exc}")
            return
        items += [RunInputs(text=ln.strip(), language=lang) for ln in lines if ln.strip()]

    if not items:
        abort("need at least one --text, --file, or --inputs-file")

    ctx = make_context(
        data_dir=data_dir, llm=llm, video=video,
        render_mode=render_mode, show_run_id=True,
    )
    typer.echo(f"{len(items)} inputs -> style {style_id}\n")
    try:
        records = run_batch(
            ctx, style_id, items, plan=build_plan(include_qc=not no_qc)
        )
    except StyleLoomError as exc:
        abort(str(exc))
        return

    archetypes = {r.hook_archetype for r in records if r.hook_archetype}
    _report(records, len(items))
    typer.echo(
        f"\ndistinct hook archetypes: {len(archetypes)}/{len(items)} "
        f"{sorted(archetypes)}"
    )
