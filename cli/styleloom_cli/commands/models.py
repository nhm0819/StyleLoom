"""`styleloom models` -- price the endpoints against a real style.

The expensive decision in this system is invisible until you do the arithmetic:
every image-to-video endpoint bills a minimum clip length well above a short-form
cut, so a style with 1.2s cuts pays for 3-4s per shot and throws most of it away.
How much depends on the style's own shot count, which is why this reads a saved
`style.json` rather than quoting a generic number.

It also shows what `multi_prompt` would cost on endpoints that support it, since
that is the difference between paying for the footage you use and paying for the
floor. Nothing here calls an API; it is arithmetic over configs/fal_models.yaml.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Annotated

import typer
from styleloom_core import StyleLoomError
from styleloom_core.providers.video import load_fal_specs

from ..options import abort, make_context

app = typer.Typer(no_args_is_help=True)

# Used when no style is given, so the command still says something useful on a
# fresh clone. Roughly a 30s vertical ad at short-form pacing.
GENERIC_DURATION = 30.0
GENERIC_AVG_SHOT = 1.5


def shot_plan(total_sec: float, avg_shot_sec: float) -> tuple[int, float]:
    shots = max(int(round(total_sec / max(avg_shot_sec, 0.3))), 1)
    return shots, total_sec / shots


def per_shot_cost(spec: dict, shots: int, shot_sec: float) -> tuple[float, float] | None:
    """Cost and billed seconds for one generation per shot, at the endpoint floor."""
    rate = spec.get("price_per_sec")
    if rate is None:
        return None
    billed_per_shot = max(shot_sec, float(spec.get("min_duration", 0)))
    billed = shots * billed_per_shot
    return billed * float(rate), billed


def multi_prompt_cost(
    spec: dict, total_sec: float, shots: int
) -> tuple[float, float, int] | None:
    """Cost, billed seconds and generation count using a multi-shot request.

    Two corrections over the naive version, both from the verified Kling schema:

    * Generations are capped by shot count as well as by window length. Six cuts
      per request means a 20-shot montage needs four generations even though its
      total length fits in one window.
    * Per-shot durations are an integer enum with a floor of 1s, so a sub-second
      cut is billed at 1s. Billed seconds are the sum of the rounded shots, not
      the delivered length -- multi_prompt removes the endpoint's 3s floor, not
      arithmetic.
    """
    rate = spec.get("price_per_sec")
    if rate is None or "multi_prompt" not in (spec.get("capabilities") or []):
        return None
    window = float(spec.get("max_shot_window_sec") or spec.get("max_duration", total_sec))
    max_shots = int(spec.get("max_shots_per_request", 0))

    shot_sec = total_sec / max(shots, 1)
    billed_per_shot = shot_sec
    if spec.get("multi_prompt_duration_type") == "integer_string":
        billed_per_shot = max(1.0, float(round(shot_sec)))
    billed = shots * billed_per_shot

    by_time = math.ceil(billed / window) if window > 0 else 1
    by_count = math.ceil(shots / max_shots) if max_shots > 0 else 1
    generations = max(by_time, by_count, 1)
    return billed * float(rate), billed, generations


@app.command("models")
def models(
    style: Annotated[str | None, typer.Option(help="이 스타일의 실제 샷 수로 계산")] = None,
    data_dir: Annotated[Path | None, typer.Option("--data-dir")] = None,
) -> None:
    """엔드포인트별 스펙과 이 스타일 1편당 예상 비용을 출력합니다."""
    ctx = make_context(data_dir=data_dir, quiet=True)
    specs = load_fal_specs(ctx.settings)

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
    current = ctx.settings.fal_i2v_model

    rows: list[tuple[str, str, str, str, str, bool]] = []
    for model_id, spec in specs.get("image_to_video", {}).items():
        floor = spec.get("min_duration", "?")
        elo = spec.get("quality_elo")
        caps = spec.get("capabilities") or []

        shot_mode = per_shot_cost(spec, shots, shot_sec)
        if shot_mode is None:
            per_shot_text = "rate not recorded"
        else:
            cost, billed = shot_mode
            waste = (1 - total / billed) * 100
            per_shot_text = f"${cost:,.2f}  ({waste:.0f}% wasted)"

        multi = multi_prompt_cost(spec, total, shots)
        if multi is None:
            multi_text = "not supported" if "multi_prompt" not in caps else "rate not recorded"
        else:
            cost, _billed, generations = multi
            multi_text = f"${cost:,.2f}  ({generations} call{'s' if generations > 1 else ''})"

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
        f"  {'endpoint':<{width}}  {'floor':>5}  {'elo':>5}  "
        f"{'per-shot (today)':<26}  multi_prompt"
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
        "\n  per-shot is what this repo does today: one generation per cut, billed at\n"
        "  the endpoint's minimum length. The wasted share is footage paid for and\n"
        "  trimmed off, because holding the reference's pacing matters more than the\n"
        "  saving. multi_prompt carries several cuts in one request, which lowers\n"
        "  the floor from the endpoint's minimum to 1s per cut -- not to zero: the\n"
        "  per-shot duration is an integer, so a 0.76s cut is still billed as 1s and\n"
        "  delivered as 1s. On a style whose cuts run under a second that trades\n"
        "  money for pacing fidelity, which is the opposite of what this system is\n"
        "  for. Opt-in: --render-mode multi_shot, or STYLELOOM_RENDER_MODE=multi_shot.\n"
        "  Not the default, because the cuts then come from the model and qc's\n"
        "  cut_timing_drift is what decides whether the timeline survived."
    )

    t2i = specs.get("text_to_image", {})
    typer.secho(
        f"\n  current mode: {ctx.settings.render_mode}", fg="cyan", bold=True
    )
    typer.secho(f"  keyframes: {shots} needed per video in per_shot mode", bold=True)
    typer.echo(f"  endpoints: {', '.join(t2i) or 'none'}")
    typer.echo(
        "  In multi_shot mode only one start image per generation is needed, so this\n"
        "  drops to the generation count regardless of how many cuts there are."
    )
    typer.secho(f"\n  rates: {source}", fg="bright_black")
    typer.secho(
        "  Rates move and endpoints change. Re-check the model page before "
        "budgeting.",
        fg="bright_black",
    )
