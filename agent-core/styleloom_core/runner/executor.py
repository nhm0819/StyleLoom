"""The execution loop.

One function, `run_once`, is the entire public entry point for producing a video.
The CLI calls it directly; an API route would call it from a background task and
a worker would call it from a queue consumer. None of them need anything else,
which is the property that makes the transport layers thin.

Progress is reported by emitting events on the context rather than printing, so
the core stays silent and each transport renders progress its own way.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from ..config import Settings
from ..context import Context, new_run_id
from ..errors import NotFoundError, ToolError
from ..events import EventKind
from ..planner import OPTIONAL_STEPS, STANDARD_STEPS, Plan, build_plan
from ..schema import AssembleResult, RunInputs, RunRecord, RunStatus
from ..session import ARTIFACT_FILES, RunSession
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


def default_plan(ctx: Context, include_qc: bool = True) -> Plan:
    """The standard plan, minus the stages this context turns off.

    Every transport goes through here rather than calling `build_plan` with its
    own arguments. `use_first_frame` is a setting, so a caller that built the plan
    itself would have to remember to read it -- and forgetting means the keyframe
    stage runs and bills for images that render was configured not to use.
    """
    return build_plan(
        include_qc=include_qc, include_keyframe=ctx.settings.use_first_frame
    )


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
        # Recorded once, on the run's first attempt, and never overwritten after --
        # a resume calls `execute` again with a *shorter* plan (just what is left),
        # and if this ran unconditionally every time, a second failure after a
        # resume would leave plan_steps holding only the tail, silently forgetting
        # that earlier stages like "render" were ever part of this run. That would
        # make a later `--from-stage render` fail with "not a step in this run's
        # plan" even though render.json is sitting right there on disk.
        if not session.record.plan_steps:
            session.mark(plan_steps=list(plan.steps))

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
    return execute(ctx, session, plan or default_plan(ctx))


def _wrote_its_artifact(ctx: Context, run_id: str, step: str) -> bool:
    """Whether a step left its output file in this run's directory."""
    spec = registry.get(step)
    filename = ARTIFACT_FILES.get(spec.writes or "")
    if filename is None:
        return False
    return (ctx.runs.dir_for(run_id) / filename).exists()


def infer_plan_steps(ctx: Context, run_id: str, stage: str) -> list[str]:
    """Reconstruct a plan for a run whose record does not carry one.

    Runs created before `plan_steps` was recorded still have everything that
    matters on disk -- that is the whole premise of keeping each stage's output as
    a readable file. Refusing to resume them because a bookkeeping field is absent
    threw away a finished, already-billed render over metadata.

    Only the optional steps are in question, since the rest of the pipeline is
    fixed. They are decided from two different sources depending on which side of
    the resume point they fall on, because only one of the two is knowable:

      * Before the resume point, the disk is the evidence. `keyframe.json` present
        means the stage ran and its output is there to seed; absent means it was
        not in the original plan, and inventing it would try to load a file that
        does not exist.
      * At or after the resume point, nothing has run yet, so there is no evidence
        to read and current settings are the only available answer.
    """
    steps: list[str] = []
    past_resume_point = False
    for name in STANDARD_STEPS:
        if name == stage:
            past_resume_point = True
        if name in OPTIONAL_STEPS:
            keep = (
                # `qc` is the only optional step after render, and nothing reads it,
                # so it is safe to add back for the remainder of the run.
                (name != "keyframe" or ctx.settings.use_first_frame)
                if past_resume_point
                else _wrote_its_artifact(ctx, run_id, name)
            )
            if not keep:
                continue
        steps.append(name)
    return steps


def resume_session(
    ctx: Context, run_id: str, from_stage: str | None = None
) -> tuple[RunSession, Plan]:
    """Rebuild a session from a run's own artifacts, picking up where it left off.

    Every stage writes its output as a plain JSON file for exactly this reason: a
    render that finished several clips and then broke in assemble should not have
    to render them again to try assemble a second time. This reopens what the
    earlier stages wrote and hands back a plan trimmed to what is left.

    `from_stage` defaults to the stage `runs ls` shows the run failed at. Naming it
    explicitly instead re-does a stage that technically succeeded but produced
    something wrong -- a corrupted clip in `shots/`, say -- without re-running
    everything before it. Either way it must be a step in the run's own recorded
    plan, not just any tool name: resuming re-enters that exact plan, not a new one.
    """
    record = ctx.runs.load(run_id)  # raises NotFoundError
    stage = from_stage or record.stage
    inferred = not record.plan_steps
    steps = list(record.plan_steps) or infer_plan_steps(ctx, run_id, stage)
    if stage not in steps:
        raise ToolError(
            f"{stage!r} is not a step in run {run_id!r}'s plan: {steps}. "
            + (
                "This run finished, so there is no failed stage to pick up from -- "
                "name one with --from-stage to redo it."
                if record.status is RunStatus.DONE
                else "Name one of those with --from-stage."
            )
        )

    raw_inputs = json.loads(
        (ctx.runs.dir_for(run_id) / "inputs.json").read_text(encoding="utf-8")
    )
    session = RunSession(
        record=record, inputs=RunInputs.model_validate(raw_inputs), store=ctx.runs
    )
    # "style" is preloaded rather than written by a plan step, so it never appears
    # in plan_steps -- but plan.validate() checks session.artifacts directly, not
    # the PRELOADED default, so it has to be seeded here regardless of from_stage.
    session.seed_from_disk("style")

    if inferred:
        # Said out loud: the run is being continued under a plan nobody recorded,
        # so if the original had dropped a stage this could differ from it.
        ctx.emit(
            EventKind.WARNING,
            run_id,
            stage="resume",
            message=(
                f"this run recorded no plan, so one was inferred from its files: "
                f"{steps}. Optional stages before {stage!r} were included only "
                "where their artifact is on disk."
            ),
        )
        session.mark(plan_steps=steps)

    resume_at = steps.index(stage)
    for name in steps[:resume_at]:
        spec = registry.get(name)
        if spec.writes:
            session.seed_from_disk(spec.writes)

    remaining = tuple(steps[resume_at:])
    plan = Plan(name=f"{record.style_id}_resume_from_{stage}", steps=remaining)
    return session, plan


def resume_run(ctx: Context, run_id: str, from_stage: str | None = None) -> RunRecord:
    """Continue a run from where it stopped, or from a named stage.

    Reuses `execute` unchanged rather than adding a resume-aware branch to it: a
    resumed run is an ordinary run whose session happens to already hold some
    artifacts, not a different code path that needs its own testing.
    """
    session, plan = resume_session(ctx, run_id, from_stage)
    return execute(ctx, session, plan)


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
    plan = plan or default_plan(ctx)
    return [run_once(ctx, style_id, item, plan=plan) for item in inputs]


