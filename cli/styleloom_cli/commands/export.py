"""`styleloom export` -- assemble the submission bundle.

The deliverable asks for each video paired with the original input prompt. Run
directories already hold both, but spread across `inputs.json` and `final.mp4` in
separate timestamped folders. This collects them into one folder a reviewer can
open without knowing anything about the layout.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Annotated

import typer
from styleloom_core import RunStatus, StyleLoomError
from styleloom_core.schema import Casting, HookResult, QCReport

from ..options import abort, make_context

app = typer.Typer(no_args_is_help=True)


def _read(path: Path, model):
    if not path.exists():
        return None
    try:
        return model.model_validate_json(path.read_text(encoding="utf-8"))
    except ValueError:
        return None


@app.command("export")
def export(
    out_dir: Annotated[Path, typer.Argument(help="번들을 만들 디렉터리")],
    style: Annotated[str | None, typer.Option(help="스타일로 필터")] = None,
    limit: Annotated[int, typer.Option("-n", "--limit", help="포함할 최신 실행 개수")] = 3,
    run_id: Annotated[
        list[str] | None, typer.Option("--run", help="특정 실행 지정 (반복 가능)")
    ] = None,
    data_dir: Annotated[Path | None, typer.Option("--data-dir")] = None,
) -> None:
    """완료된 실행들을 제출용 번들로 모읍니다 (입력 프롬프트 + mp4).

    기본값은 최신 성공 실행 3건입니다. `--run`으로 직접 고를 수도 있습니다.
    """
    ctx = make_context(data_dir=data_dir, quiet=True)

    try:
        if run_id:
            records = [ctx.runs.load(r) for r in run_id]
        else:
            records = [
                r
                for r in ctx.runs.list_records(style_id=style, limit=max(limit * 4, 20))
                if r.status is RunStatus.DONE
            ][:limit]
    except StyleLoomError as exc:
        abort(str(exc))
        return

    if not records:
        abort("no completed runs to export. Run `styleloom batch` first.")
    failed = [r.run_id for r in records if r.status is not RunStatus.DONE]
    if failed:
        abort(f"these runs did not complete: {', '.join(failed)}")

    # `list_records` is newest-first, which is right for browsing and wrong for a
    # bundle: a reviewer reads 01 expecting the first video that was generated.
    if not run_id:
        records.reverse()

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []
    prompt_lines: list[str] = []

    for i, record in enumerate(records, start=1):
        run_dir = ctx.runs.dir_for(record.run_id)
        final = run_dir / "final.mp4"
        if not final.exists():
            abort(f"{record.run_id} has no final.mp4")

        # Numbered, so the bundle reads in order rather than by hash.
        video_name = f"{i:02d}_{record.run_id}.mp4"
        shutil.copy2(final, out_dir / video_name)

        raw_inputs = _load_inputs(run_dir)
        prompt = raw_inputs.get("text") or raw_inputs.get("file_path") or ""
        cast = _read(run_dir / "casting.json", Casting)
        hook = _read(run_dir / "hook.json", HookResult)
        qc = _read(run_dir / "qc_report.json", QCReport)

        entry = {
            "index": i,
            "video": video_name,
            "run_id": record.run_id,
            "style_id": record.style_id,
            "input_prompt": prompt,
            "input_kind": "file" if raw_inputs.get("file_path") else "text",
            "hook_text": record.hook_text,
            "hook_archetype": record.hook_archetype,
            "hook_selection": hook.selection_method if hook else None,
            "creator": cast.creator.label if cast else None,
            "setting": cast.setting.label if cast else None,
            "qc_score": record.qc_score,
            "qc_passed": qc.passed if qc else None,
        }
        manifest.append(entry)

        # A plain-text companion, because the requirement is the prompt "원본" and a
        # reviewer should not have to parse JSON to read it.
        prompt_lines += [
            f"[{i:02d}] {video_name}",
            f"  input  : {prompt}",
            f"  hook   : {record.hook_text}  ({record.hook_archetype})",
            f"  cast   : {entry['creator']} / {entry['setting']}",
            f"  qc     : {record.qc_score}",
            "",
        ]
        # And the full per-run artifacts, so the bundle stays auditable.
        shutil.copytree(
            run_dir, out_dir / "runs" / record.run_id, dirs_exist_ok=True,
            ignore=shutil.ignore_patterns("shots", "captioned", "raw", "cast"),
        )

    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "prompts.txt").write_text("\n".join(prompt_lines), encoding="utf-8")

    typer.secho(f"exported {len(records)} run(s) -> {out_dir}", fg="green")
    for entry in manifest:
        typer.echo(
            f"  {entry['video']}\n"
            f"    input : {entry['input_prompt']}\n"
            f"    hook  : {entry['hook_text']}  [{entry['hook_archetype']}]\n"
            f"    cast  : {entry['creator']} / {entry['setting']}"
        )
    typer.echo("\n  manifest.json  prompts.txt  runs/<run_id>/")

    archetypes = {e["hook_archetype"] for e in manifest}
    creators = {e["creator"] for e in manifest}
    settings_ = {e["setting"] for e in manifest}
    typer.secho(
        f"\nvariety: {len(archetypes)} hook archetypes, "
        f"{len(creators)} creators, {len(settings_)} settings across {len(manifest)} videos",
        fg="cyan",
    )


def _load_inputs(run_dir: Path) -> dict:
    path = run_dir / "inputs.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return {}
    return data if isinstance(data, dict) else {}
