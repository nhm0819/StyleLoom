"""Measure whether the output actually conforms to the style."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..media import probe_video
from ..schema import (
    AssembleResult,
    QCCheck,
    QCReport,
    RenderResult,
    Storyboard,
    StyleSchema,
)
from .registry import tool

if TYPE_CHECKING:
    from ..context import Context
    from ..session import RunSession

PASS_THRESHOLD = 0.7

# The timing checks are relative: a fixed second means something different at 7s
# and at 60s, and this is a short-form system.
#
# (fraction of the target, absolute floor). The floor keeps a very short cut from
# demanding frame-accuracy that cut detection cannot resolve.
RELATIVE_TOLERANCES = {
    "avg_shot_sec": (0.15, 0.08),
    "total_duration": (0.10, 0.40),
}

TOLERANCES = {
    "saturation": 0.18,
    "contrast": 0.20,
    "warmth": 0.12,
    "hook_window_shots": 0.5,
    # Mean distance from each requested cut to the nearest one in the output. Near
    # zero in per_shot mode by construction, so a non-zero value there means a
    # trim failed; in multi_shot it measures whether the model obeyed.
    "cut_timing_drift": 0.35,
}


def tolerance_for(name: str, target: float) -> float:
    """Tolerance for one check, scaled to the target where the check is a duration."""
    if name in RELATIVE_TOLERANCES:
        fraction, floor = RELATIVE_TOLERANCES[name]
        return round(max(floor, abs(target) * fraction), 3)
    return TOLERANCES[name]


def cut_drift(requested: list[float], detected: list[float]) -> float:
    """Mean distance from each requested cut to the nearest detected one.

    Nearest-neighbour rather than positional pairing on purpose: cut detection
    misses low-contrast cuts and invents none, so pairing by index would punish a
    correct timeline for one missed detection. This measures whether the cuts we
    asked for exist somewhere close, which is the question worth asking.
    """
    if not requested:
        return 0.0
    if not detected:
        # No cuts found at all. Report the full offset rather than 0.0, which would
        # read as a pass.
        return max(requested)
    return sum(min(abs(r - d) for d in detected) for r in requested) / len(requested)


def check(name: str, target: float, actual: float, tolerance: float) -> QCCheck:
    return QCCheck(
        name=name,
        target=round(target, 3),
        actual=round(actual, 3),
        tolerance=tolerance,
        passed=abs(actual - target) <= tolerance,
    )


@tool("qc", reads=("style", "storyboard", "render", "assemble"), writes="qc")
def qc(ctx: Context, session: RunSession) -> QCReport:
    """Re-probe the finished video and compare it to the style it claims to follow.

    The claim being made is "a system that reproduces a style", not "one good
    video". That claim is only credible if conformance is measured, so this runs
    the same probe used on the reference and diffs the numbers.
    """
    style = session.get("style", StyleSchema)
    board = session.get("storyboard", Storyboard)
    assembled = session.get("assemble", AssembleResult)
    rendered = session.get("render", RenderResult)

    m = probe_video(assembled.final_video)
    hook_shots = float(len([s for s in board.shots if s.role == "hook"]))
    drift = cut_drift(rendered.cut_timeline(), m["cuts"])

    checks = [
        check("avg_shot_sec", style.pacing.avg_shot_sec, m["avg_shot_sec"],
              tolerance_for("avg_shot_sec", style.pacing.avg_shot_sec)),
        check("total_duration", style.total_duration, m["duration"],
              tolerance_for("total_duration", style.total_duration)),
        check("saturation", style.look.saturation, m["saturation"],
              TOLERANCES["saturation"]),
        check("contrast", style.look.contrast, m["contrast"], TOLERANCES["contrast"]),
        check("warmth", style.look.warmth, m["warmth"], TOLERANCES["warmth"]),
        check("hook_window_shots", float(style.hook_style.cut_count), hook_shots,
              TOLERANCES["hook_window_shots"]),
        check("cut_timing_drift", 0.0, drift, TOLERANCES["cut_timing_drift"]),
    ]
    score = sum(1 for c in checks if c.passed) / len(checks)
    report = QCReport(
        style_id=style.style_id,
        checks=checks,
        score=round(score, 3),
        passed=score >= PASS_THRESHOLD,
    )
    session.mark(qc_score=report.score)
    return report
