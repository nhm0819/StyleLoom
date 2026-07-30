"""`styleloom hook preview` -- prove the hook is not frozen."""

from __future__ import annotations

import collections
from pathlib import Path
from typing import Annotated

import typer
from styleloom_core import RunInputs, StyleLoomError
from styleloom_core.schema import RunRecord
from styleloom_core.session import RunSession
from styleloom_core.tools import casting as casting_tool
from styleloom_core.tools import hook as hook_tool
from styleloom_core.tools import ingest as ingest_tool
from styleloom_core.tools import outline as outline_tool

from ..options import abort, make_context

app = typer.Typer(no_args_is_help=True, help="훅 생성만 따로 확인합니다.")


@app.command("preview")
def preview(
    style_id: Annotated[str, typer.Argument(help="사용할 스타일 ID")],
    text: Annotated[str, typer.Option("--text", "-t", help="입력 텍스트")],
    n: Annotated[int, typer.Option("-n", help="같은 입력으로 반복 생성할 횟수")] = 8,
    lang: Annotated[str, typer.Option(help="출력 언어")] = "ko",
    data_dir: Annotated[Path | None, typer.Option("--data-dir")] = None,
    llm: Annotated[str | None, typer.Option(help="mock | anthropic")] = None,
) -> None:
    """같은 입력을 n번 통과시켜 훅이 고정되지 않았음을 직접 확인합니다.

    영상 렌더링 없이 ingest -> outline까지만 한 번 수행한 뒤, 훅 생성만 n회
    반복합니다. 비결정성이 주장이 아니라 측정값으로 나오는 게 요점입니다.
    """
    ctx = make_context(data_dir=data_dir, llm=llm, quiet=True)
    try:
        style = ctx.styles.load(style_id)
    except StyleLoomError as exc:
        abort(str(exc))
        return

    # A throwaway session: the hook tool needs somewhere to read artifacts from,
    # but a preview must not leave a run on disk or touch archetype history.
    inputs = RunInputs(text=text, language=lang)
    session = RunSession(
        record=RunRecord(run_id="preview", style_id=style_id),
        inputs=inputs,
        store=ctx.runs,
    )
    session.artifacts["style"] = style

    try:
        brief = ingest_tool.ingest(ctx, session)
        session.artifacts["brief"] = brief
        # Casting before outline, as in a real run: both stages size their prompts
        # against the style tokens, and those carry the drawn creator and setting.
        casting = casting_tool.casting(ctx, session)
        session.artifacts["casting"] = casting
        outline = outline_tool.outline(ctx, session)
        session.artifacts["outline"] = outline
    except StyleLoomError as exc:
        abort(str(exc))
        return

    typer.echo(f"topic  : {brief.topic}")
    typer.echo(f"payoff : {outline.payoff}")
    typer.secho(f"\n{n} independent generations, same input:\n", bold=True)

    recent: list[str] = ctx.history.recent_values(style_id, "hook")
    archetypes: list[str] = []
    texts: list[str] = []
    for i in range(n):
        try:
            result = hook_tool.generate(
                ctx, brief, outline, style, casting, recent_archetypes=recent
            )
        except StyleLoomError as exc:
            abort(str(exc))
            return
        archetypes.append(result.archetype_sampled)
        texts.append(result.selected.text)
        typer.echo(f"{i + 1:2d}. [{result.archetype_sampled:14s}] {result.selected.text}")
        # Feed each draw back in, so the preview shows the same recency pressure a
        # real batch would apply rather than n independent cold starts.
        recent = ([result.archetype_sampled] + recent)[: ctx.settings.hook_recency_window]

    typer.echo()
    typer.secho(
        f"distinct archetypes: {len(set(archetypes))}/{n}   "
        f"distinct texts: {len(set(texts))}/{n}",
        fg="cyan",
        bold=True,
    )
    counts = collections.Counter(archetypes)
    typer.echo("  " + "  ".join(f"{k}={v}" for k, v in counts.most_common()))
