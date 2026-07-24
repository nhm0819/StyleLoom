"""Stage 8 - Style conformance check.

The claim being graded is "a system that reproduces a style", not "one good
video". That claim is only credible if conformance is measured. This re-runs the
same probe used on the reference and compares the numbers.
"""

from __future__ import annotations

from pathlib import Path

from ..schema import QCCheck, QCReport, StyleSchema, Storyboard
from .analyze import probe_video


def _check(name: str, target: float, actual: float, tolerance: float) -> QCCheck:
    return QCCheck(
        name=name,
        target=round(target, 3),
        actual=round(actual, 3),
        tolerance=tolerance,
        passed=abs(actual - target) <= tolerance,
    )


def evaluate(final_video: Path, style: StyleSchema, storyboard: Storyboard) -> QCReport:
    m = probe_video(final_video)

    checks = [
        _check("avg_shot_sec", style.pacing.avg_shot_sec, m["avg_shot_sec"], 0.6),
        _check("total_duration", style.total_duration, m["duration"], 6.0),
        _check("saturation", style.look.saturation, m["saturation"], 0.18),
        _check("contrast", style.look.contrast, m["contrast"], 0.20),
        _check("warmth", style.look.warmth, m["warmth"], 0.12),
        _check(
            "hook_window_shots",
            float(style.hook_style.cut_count),
            float(len([s for s in storyboard.shots if s.role == "hook"])),
            0.5,
        ),
    ]
    score = sum(1 for c in checks if c.passed) / len(checks)
    return QCReport(
        style_id=style.style_id,
        checks=checks,
        score=round(score, 3),
        passed=score >= 0.7,
    )
