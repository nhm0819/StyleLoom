"""A debuggable entry point for the core pipeline.

    python -m styleloom_core.debug

`styleloom run` does the same work, but a Typer command is an awkward thing to
step through: the arguments arrive through a decorator, errors are converted to
clean exits before you can inspect them, and `abort()` swallows the traceback you
actually wanted. This module calls `run_once` directly, so the frame you land on
is the pipeline rather than the CLI framework.

It is a driver, not a second interface. Everything configurable lives in the
block below and is meant to be edited in place -- there is no argument parsing,
because the point is to set a breakpoint and press F5, not to type flags. Use
`styleloom run` when you want flags.

WHERE TO PUT BREAKPOINTS. The plan is a fixed sequence, so each stage has one
obvious place to stop:

    ingest      tools/ingest.py:ingest              raw input -> Brief
    casting     tools/casting.py:casting            creator + setting sampling
    outline     tools/outline.py:outline            body beats, and the payoff
    hook        tools/hook.py:hook                  the non-deterministic choice
    storyboard  tools/storyboard.py:storyboard      beats -> per-shot prompts
    render      tools/render.py:render_shot         one generation per cut
    assemble    tools/assemble.py:assemble          captions, then concat
    qc          tools/qc.py:qc                      re-measure against the style

For the wiring rather than a stage, `runner/executor.py:execute` is the loop that
runs them, and `planner/plan.py:Plan.validate` is where a misordered plan fails.

OFFLINE BY DEFAULT, and deliberately so. Mock providers need no keys and cost
nothing, which is the only way a breakpoint in `render` is affordable -- stepping
through a real run bills every restart. Set PROVIDERS below to go live.
"""

from __future__ import annotations

import json

from .config import REPO_ROOT, Settings
from .context import build_context
from .errors import StyleLoomError
from .events import Event
from .planner import build_plan
from .runner import run_once
from .schema import RunInputs, RunStatus, StyleSchema

# --------------------------------------------------------------------------- #
# Edit this block.
# --------------------------------------------------------------------------- #

STYLE_ID = "biodance_asmr_montage"

INPUT = RunInputs(
    text="세라믹 텀블러인데 보온이 6시간 간다",
    language="ko",
)

# "mock" for both keeps this free and keyless. "auto" picks the real provider when
# its key is present, which is what `styleloom run` defaults to.
PROVIDERS = {"llm_provider": "mock", "video_provider": "mock"}

# per_shot is the verified path. multi_shot exercises windowing, the Kling
# multi_prompt payload, and the qc drift check instead.
RENDER_MODE = "per_shot"

INCLUDE_QC = True

# Where a missing style is installed from, so a fresh clone runs without first
# calling `styleloom style set`.
SAMPLE_STYLES = REPO_ROOT / "samples" / "styles"


# --------------------------------------------------------------------------- #


class PrintSink:
    """Stage transitions on stdout.

    Local rather than the CLI's ConsoleSink: agent-core does not import from the
    CLI distribution, and the whole claim of the layout is that it does not have
    to. Satisfying the EventSink Protocol is the entire contract.
    """

    def emit(self, event: Event) -> None:
        stage = event.stage or "-"
        message = f"  {event.message}" if event.message else ""
        print(f"[{event.kind.value:14}] {stage:11}{message}")


def install_sample_style(ctx, style_id: str) -> bool:
    """Copy a bundled sample into the data dir if that style is not installed.

    Returns True if it installed one. A debugger session that dies on "style not
    found" before reaching any of the code under test is pure friction, and this
    is the one setup step `styleloom run` requires up front.
    """
    if ctx.styles.exists(style_id):
        return False
    source = SAMPLE_STYLES / style_id / "style.json"
    if not source.exists():
        available = sorted(p.parent.name for p in SAMPLE_STYLES.glob("*/style.json"))
        raise StyleLoomError(
            f"style {style_id!r} is not installed and no sample exists at {source}. "
            f"Samples available: {available}"
        )
    ctx.styles.save(StyleSchema(**json.loads(source.read_text(encoding="utf-8"))))
    return True


def main() -> int:
    # data_dir defaults to the relative Path("data") with no repo-root fallback,
    # so an IDE launching this from a different cwd would silently read and write
    # somewhere else. Pinning it is what makes the launch directory irrelevant.
    settings = Settings(
        data_dir=REPO_ROOT / "data",
        render_mode=RENDER_MODE,
        **PROVIDERS,
    )
    ctx = build_context(settings, events=PrintSink())
    print(f"providers: {settings.provider_summary()}")
    print(f"data_dir : {settings.data_dir}")

    if install_sample_style(ctx, STYLE_ID):
        print(f"installed sample style {STYLE_ID!r}")

    record = run_once(
        ctx, STYLE_ID, INPUT, plan=build_plan(include_qc=INCLUDE_QC)
    )

    print()
    if record.status is RunStatus.DONE:
        print(f"run      : {record.run_id}")
        print(f"hook     : [{record.hook_archetype}] {record.hook_text}")
        print(f"qc       : {record.qc_score}")
        print(f"video    : {record.artifacts.get('final_video', '-')}")
        print(f"artifacts: {ctx.runs.dir_for(record.run_id)}")
        return 0

    # run_once returns a failed record rather than raising, so a batch can carry
    # on past one bad input. In a debugger that would mean losing the traceback,
    # which is the opposite of useful -- hence reporting the stage and error here.
    print(f"FAILED at {record.stage}: {record.error}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
