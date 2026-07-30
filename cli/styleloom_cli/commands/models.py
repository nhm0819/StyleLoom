"""`styleloom models` -- price the endpoints against a real style.

The expensive decision in this system is invisible until you do the arithmetic:
every image-to-video endpoint bills a minimum clip length well above a short-form
cut, so a style with 1.2s cuts pays for 3-4s per shot and throws most of it away.
How much depends on the style's own shot count, which is why this reads a saved
`style.json` rather than quoting a generic number.

It prices the endpoint the pipeline actually renders on -- image-to-video -- and
shows what one request per cut *would* have cost beside what a windowed request
costs, because that difference is why the per-cut path was removed.

Reported in billed seconds rather than dollars. The official platform bills
credits against a subscription tier and publishes no per-second rate this file
could record, so a currency column would be empty on every row. Billed seconds
are the part of the arithmetic that does not depend on the price list: they are
what the plan is charged for, and the wasted share is what the pacing costs.

Nothing here calls an API; it is arithmetic over configs/kling_models.yaml.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Annotated

import typer
from styleloom_core import StyleLoomError
from styleloom_core.providers.kling import load_kling_specs

from ..options import abort, make_context

app = typer.Typer(no_args_is_help=True)

# Used when no style is given, so the command still says something useful on a
# fresh clone. Roughly a 30s vertical ad at short-form pacing.
GENERIC_DURATION = 30.0
GENERIC_AVG_SHOT = 1.5


def shot_plan(total_sec: float, avg_shot_sec: float) -> tuple[int, float]:
    shots = max(int(round(total_sec / max(avg_shot_sec, 0.3))), 1)
    return shots, total_sec / shots


def billed_per_shot_sec(spec: dict, shots: int, shot_sec: float) -> float:
    """Billed seconds for one generation per shot, at the endpoint floor."""
    return shots * max(shot_sec, float(spec.get("min_duration", 0)))


def multi_prompt_billed_sec(
    spec: dict, total_sec: float, shots: int
) -> tuple[float, int] | None:
    """Billed seconds and generation count using a multi-shot request.

    Two corrections over the naive version, both from the verified Kling schema:

    * Generations are capped by shot count as well as by window length. Six cuts
      per request means a 20-shot montage needs four generations even though its
      total length fits in one window.
    * Per-shot durations are an integer enum with a floor of 1s, so a sub-second
      cut is billed at 1s. Billed seconds are the sum of the rounded shots, not
      the delivered length -- multi_prompt removes the endpoint's 3s floor, not
      arithmetic.
    """
    # Either transport counts: a `multi_prompt` array on the legacy endpoints, a
    # semicolon shot list in the prompt on 3.0. The arithmetic is the same either
    # way -- several cuts in one generation -- and gating on the array name alone
    # reported "no saving available" for the endpoints the harness now defaults to.
    if not ({"multi_prompt", "semicolon_prompt"} & set(spec.get("capabilities") or [])):
        return None
    window = float(spec.get("max_shot_window_sec") or spec.get("max_duration", total_sec))
    max_shots = int(spec.get("max_shots_per_request", 0))

    shot_sec = total_sec / max(shots, 1)
    billed_per_shot = shot_sec
    if spec.get("api") == "v3" or spec.get("multi_prompt_duration_type") == "integer_string":
        billed_per_shot = max(1.0, float(round(shot_sec)))
    billed = shots * billed_per_shot

    by_time = math.ceil(billed / window) if window > 0 else 1
    by_count = math.ceil(shots / max_shots) if max_shots > 0 else 1
    generations = max(by_time, by_count, 1)
    return billed, generations


@app.command("models")
def models(
    style: Annotated[str | None, typer.Option(help="이 스타일의 실제 샷 수로 계산")] = None,
    data_dir: Annotated[Path | None, typer.Option("--data-dir")] = None,
) -> None:
    """엔드포인트별 스펙과 이 스타일 1편당 예상 비용을 출력합니다."""
    ctx = make_context(data_dir=data_dir, quiet=True)
    specs = load_kling_specs(ctx.settings)

    if style:
        try:
            loaded = ctx.styles.load(style)
        except StyleLoomError as exc:
            abort(str(exc))
            return
        total, avg = loaded.total_duration, loaded.pacing.avg_shot_sec
        label = f"style '{style}'"
    else:
        total, avg = GENERIC_DURATION, GENERIC_AVG_SHOT
        label = "generic 30s short-form (pass --style for your own numbers)"

    shots, shot_sec = shot_plan(total, avg)
    typer.secho(f"{label}", bold=True)
    typer.echo(f"  {total:.1f}s total, {shots} shots at ~{shot_sec:.2f}s each\n")

    source = specs.get("pricing_source", "unknown")
    current = ctx.settings.kling_i2v_model

    rows: list[tuple[str, str, str, str, str, bool]] = []
    # image_to_video, not text_to_video: the pipeline renders text -> image -> video,
    # and pricing the fallback endpoint would describe a path that does not run.
    for model_id, spec in specs.get("image_to_video", {}).items():
        floor = spec.get("min_duration", "?")
        elo = spec.get("quality_elo")
        billed = billed_per_shot_sec(spec, shots, shot_sec)
        waste = (1 - total / billed) * 100 if billed else 0.0
        per_shot_text = f"{billed:.0f}s billed  ({waste:.0f}% wasted)"

        multi = multi_prompt_billed_sec(spec, total, shots)
        if multi is None:
            multi_text = "not supported"
        else:
            billed_multi, generations = multi
            plural = "s" if generations > 1 else ""
            multi_text = (
                f"{billed_multi:.0f}s billed  ({generations} call{plural})"
            )

        rows.append(
            (
                model_id,
                f"{floor}s",
                str(elo) if elo else "-",
                per_shot_text,
                multi_text,
                model_id == current,
            )
        )

    width = max(len(r[0]) for r in rows)
    header = (
        f"  {'model_name':<{width}}  {'floor':>5}  {'elo':>5}  "
        f"{'one call per cut':<26}  multi_shot (today)"
    )

    typer.secho(header, bold=True)
    typer.echo("  " + "-" * (len(header) - 2))
    for model_id, floor, elo, per_shot, multi, is_current in rows:
        marker = "*" if is_current else " "
        line = (
            f"{marker} {model_id:<{width}}  {floor:>5}  {elo:>5}  "
            f"{per_shot:<26}  {multi}"
        )
        typer.secho(line, fg="cyan" if is_current else None, bold=is_current)

    typer.echo()
    typer.secho(f"  * current default ({current})", fg="cyan")
    typer.echo(
        "\n  one call per cut is what this repo used to do: every cut billed at the\n"
        "  endpoint's minimum length, with the wasted share paid for and trimmed off.\n"
        "  multi_shot carries a window of cuts in one request, which lowers the floor\n"
        "  from the endpoint's minimum to 1s per cut -- not to zero: a per-cut\n"
        "  duration is a whole number of seconds, so a 0.76s cut is still billed and\n"
        "  delivered at 1s. It is now the only path, and the cost of that is that the\n"
        "  cuts come from the model: qc's cut_timing_drift is what decides whether\n"
        "  the timeline survived."
    )

    t2i = specs.get("text_to_image", {})
    if ctx.settings.use_first_frame:
        # The generation count, not the cut count. One anchor for the run plus one
        # opening frame per request -- which is what makes this affordable at all,
        # since the removed design bought one image per cut and held nothing.
        spec = specs.get("image_to_video", {}).get(ctx.settings.kling_i2v_model, {})
        windowed = multi_prompt_billed_sec(spec, total, shots)
        requests = windowed[1] if windowed else shots
        typer.secho(
            f"  first frames: 1 anchor + {requests} opening frame(s) = "
            f"{requests + 1} image call(s)",
            bold=True,
        )
        typer.echo(f"  endpoints: {', '.join(t2i) or 'none'}")
        typer.echo(
            "  The anchor is generated once from the cast creator and reused as the\n"
            "  reference for every opening frame, so identity survives across\n"
            "  generations. One request holds it within its own output and cannot\n"
            "  hold it between requests.\n"
            "  STYLELOOM_USE_FIRST_FRAME=false drops both the image\n"
            "  calls and that guarantee."
        )
    else:
        typer.secho("  first frames: off (text-to-video)", bold=True)
        typer.echo(
            "  No image calls, and no cross-request identity: each generation rolls\n"
            "  its own presenter. STYLELOOM_USE_FIRST_FRAME=true adds one anchor plus\n"
            "  one frame per generation."
        )
    typer.secho(f"\n  rates: {source}", fg="bright_black")
    typer.secho(
        "  Rates move and endpoints change. Re-check the model page before "
        "budgeting.",
        fg="bright_black",
    )
