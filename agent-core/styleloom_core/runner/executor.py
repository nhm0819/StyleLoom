"""The execution loop.

One function, `run_once`, is the entire public entry point for producing a video.
The CLI calls it directly; an API route would call it from a background task and
a worker would call it from a queue consumer. None of them need anything else,
which is the property that makes the transport layers thin.

Progress is reported by emitting events on the context rather than printing, so
the core stays silent and each transport renders progress its own way.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ..config import Settings
from ..context import Context, new_run_id
from ..errors import NotFoundError
from ..events import EventKind
from ..planner import Plan, build_plan
from ..schema import AssembleResult, RunInputs, RunRecord, RunStatus
from ..session import RunSession
from ..tools import registry

if TYPE_CHECKING:
    from ..schema import StyleSchema

# Every caller-supplied file on RunInputs. All three get the same lookup as a
# reference video: one rule for "a path the user typed", not one per option.
INPUT_ASSET_FIELDS = ("file_path", "bgm")


def resolve_inputs(settings: Settings, inputs: RunInputs) -> RunInputs:
    """Point every asset path at a file that exists, or say which ones do not.

    Checked here rather than where each asset is first opened, because those
    points are spread across ingest, render and assemble -- a missing bgm would
    otherwise surface as a failure several minutes into a paid run, after the
    renders had already been billed. This is the last place before any work
    starts that still sees all three together.

    The resolved paths are what `prepare_session` snapshots into `inputs.json`,
    so a run records the file it actually read rather than the argument it was
    handed.
    """
    updates: dict[str, Path] = {}
    missing: list[str] = []
    for field in INPUT_ASSET_FIELDS:
        given: Path | None = getattr(inputs, field)
        if given is None:
            continue
        found = settings.resolve_ref(given)
        if found is None:
            tried = ", ".join(str(c) for c in settings.ref_candidates(given))
            missing.append(f"{field}={given} (tried {tried})")
        elif found != given:
            updates[field] = found
    if missing:
        raise NotFoundError("input file not found: " + "; ".join(missing))
    return inputs.model_copy(update=updates) if updates else inputs


def prepare_session(
    ctx: Context,
    style_id: str,
    inputs: RunInputs,
    run_id: str | None = None,
) -> RunSession:
    """Create a run directory, load the style, and seed the artifact bag.

    The style is loaded here rather than by a tool because it is an input to the
    run, not something the run produces. It is snapshotted into the run folder so
    a later hand-edit of style.json cannot silently change what a past run meant.
    """
    inputs = resolve_inputs(ctx.settings, inputs)
    style: StyleSchema = ctx.styles.load(style_id)  # raises NotFoundError
    record = RunRecord(run_id=run_id or new_run_id(), style_id=style_id)
    ctx.runs.save(record)

    session = RunSession(record=record, inputs=inputs, store=ctx.runs)
    session.put("style", style)
    session.save_raw("inputs.json", inputs.model_dump(mode="json"))
    return session


def execute(ctx: Context, session: RunSession, plan: Plan) -> RunRecord:
    """Run a validated plan against a prepared session."""
    ctx.emit(
        EventKind.RUN_STARTED,
        session.run_id,
        message=f"plan={plan.name} steps={len(plan.steps)}",
        style_id=session.style_id,
        steps=list(plan.steps),
    )

    stage = "plan"
    try:
        # Validated against what the session actually holds, not an assumption
        # about it -- so a caller that pre-seeded extra artifacts still gets a
        # correct check.
        plan.validate(frozenset(session.artifacts))

        for stage in plan.steps:
            spec = registry.get(stage)
            session.start_stage(stage)
            ctx.emit(EventKind.STAGE_STARTED, session.run_id, stage=stage)

            result = spec(ctx, session)
            if spec.writes and result is not None:
                session.put(spec.writes, result)

            ctx.emit(EventKind.STAGE_FINISHED, session.run_id, stage=stage)

        if session.has("assemble"):
            final = session.get("assemble", AssembleResult).final_video
            session.record.artifacts["final_video"] = str(final)

        record = session.finish()
        ctx.emit(
            EventKind.RUN_FINISHED,
            session.run_id,
            stage="done",
            message=record.hook_text or "",
            qc_score=record.qc_score,
            final_video=record.artifacts.get("final_video"),
        )
        return record

    except Exception as exc:
        record = session.fail(stage, exc)
        ctx.emit(
            EventKind.RUN_FAILED,
            session.run_id,
            stage=stage,
            message=record.error or "",
        )
        return record


def run_once(
    ctx: Context,
    style_id: str,
    inputs: RunInputs,
    plan: Plan | None = None,
    run_id: str | None = None,
) -> RunRecord:
    """Input -> one video. The single entry point every transport shares.

    Never raises for a failed render: the returned record carries `status` and
    `error`, because a batch must be able to continue after one input fails.
    Configuration and missing-style errors do propagate, since those mean the
    caller asked for something impossible.
    """
    session = prepare_session(ctx, style_id, inputs, run_id=run_id)
    return execute(ctx, session, plan or build_plan())


def run_batch(
    ctx: Context,
    style_id: str,
    inputs: list[RunInputs],
    plan: Plan | None = None,
) -> list[RunRecord]:
    """Several inputs through one unchanged system.

    Sequential on purpose. The hook's recency penalty reads this style's archetype
    history, so run N has to see what run N-1 chose; running them in parallel
    would let all three draw the same archetype and lose exactly the variety the
    batch exists to demonstrate.
    """
    plan = plan or build_plan()
    return [run_once(ctx, style_id, item, plan=plan) for item in inputs]


def succeeded(records: list[RunRecord]) -> list[RunRecord]:
    return [r for r in records if r.status is RunStatus.DONE]
