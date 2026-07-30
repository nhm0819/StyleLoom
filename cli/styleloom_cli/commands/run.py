"""`styleloom run` / `styleloom run resume` / `styleloom batch` -- input to video."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from styleloom_core import (
    RunInputs,
    RunStatus,
    StyleLoomError,
    default_plan,
    resume_run,
    run_batch,
    run_once,
)

from ..default_group import DefaultCommandGroup
from ..options import abort, make_context

app = typer.Typer(no_args_is_help=True)


class _RunGroup(DefaultCommandGroup):
    # `styleloom run my_style` is the common case and takes no subcommand, so an
    # unrecognised first token is a style id, not a typo.
    default_cmd = "start"


# `run` is a group rather than a plain command so that `run resume` can live under
# it. The generating command keeps the group's own name at the call site --
# `styleloom run <style_id>` -- and is hidden, because `styleloom run start ...` is
# an implementation detail of that dispatch rather than an interface to document.
run_group = typer.Typer(
    cls=_RunGroup,
    no_args_is_help=True,
    help=(
        "입력 1개로 숏폼 영상 1편을 생성합니다: styleloom run <스타일ID> -t \"...\"\n\n"
        "옵션 전체는 `styleloom run start --help` 로 확인할 수 있습니다. "
        "중단된 실행을 이어서 하려면 `styleloom run resume <run_id>`."
    ),
)


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


@run_group.command("start", hidden=True)
def run(
    style_id: Annotated[str, typer.Argument(help="사용할 스타일 ID")],
    text: Annotated[str, typer.Option("--text", "-t", help="텍스트 입력")] = "",
    file: Annotated[Path | None, typer.Option("--file", "-f",
                    help="이미지 또는 영상 입력. 파일명만 주면 "
                         "<data-dir>/uploads/ 에서 찾습니다")] = None,
    bgm: Annotated[Path | None, typer.Option(help="배경음악 (uploads/ 조회 동일)")] = None,
    lang: Annotated[str, typer.Option(help="출력 언어")] = "ko",
    no_qc: Annotated[bool, typer.Option("--no-qc", help="정합성 검사 생략")] = False,
    data_dir: Annotated[Path | None, typer.Option("--data-dir")] = None,
    llm: Annotated[str | None, typer.Option(help="mock | anthropic")] = None,
    video: Annotated[str | None, typer.Option(help="mock | kling")] = None,
) -> None:
    """입력 1개로 숏폼 영상 1편을 생성합니다."""
    if not text and file is None:
        abort("need --text or --file")
    ctx = make_context(data_dir=data_dir, llm=llm, video=video)
    inputs = RunInputs(
        text=text, file_path=file, bgm=bgm, language=lang
    )
    try:
        record = run_once(
            ctx, style_id, inputs, plan=default_plan(ctx, include_qc=not no_qc)
        )
    except StyleLoomError as exc:
        abort(str(exc))
        return
    _report([record], 1)


@run_group.command("resume")
def resume(
    run_id: Annotated[str, typer.Argument(help="이어서 실행할 run ID")],
    from_stage: Annotated[
        str | None,
        typer.Option(
            "--from-stage",
            help="이 단계부터 다시 실행 (기본: 실패했던 단계). "
                 "성공했지만 다시 하고 싶은 단계도 지정 가능",
        ),
    ] = None,
    data_dir: Annotated[Path | None, typer.Option("--data-dir")] = None,
    llm: Annotated[str | None, typer.Option(help="mock | anthropic")] = None,
    video: Annotated[str | None, typer.Option(help="mock | kling")] = None,
) -> None:
    """실패했던(또는 다시 하고 싶은) 단계부터 이어서 실행합니다.

    앞 단계들의 산출물은 다시 만들지 않고 디스크에 저장된 그대로 읽어옵니다 --
    렌더가 끝난 뒤 assemble에서 죽었다면 그 렌더된 클립들을 다시 만들 필요가
    없다는 뜻입니다. 이 실행이 원래 어떤 plan으로 시작됐는지는 run.json에
    기록돼 있고, 기록이 없는 예전 실행은 남아 있는 파일에서 plan을 추론합니다.
    """
    ctx = make_context(data_dir=data_dir, llm=llm, video=video, show_run_id=True)
    try:
        record = resume_run(ctx, run_id, from_stage=from_stage)
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
    video: Annotated[str | None, typer.Option(help="mock | kling")] = None,
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
        show_run_id=True,
    )
    typer.echo(f"{len(items)} inputs -> style {style_id}\n")
    try:
        records = run_batch(
            ctx, style_id, items, plan=default_plan(ctx, include_qc=not no_qc)
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
