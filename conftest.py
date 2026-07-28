"""Session preflight, shared by both distributions' suites.

At the repo root rather than inside one tests/ directory so it applies however the
suite is invoked: bare `pytest`, `pytest cli/tests`, a single file, or the VS Code
debug configs.

Both suites build their fixture videos by shelling out to ffmpeg. When it is
missing that is a missing prerequisite, not fifty test failures -- and on Windows
the raw symptom is `FileNotFoundError: [WinError 2]` repeated once per affected
test, which names neither ffmpeg nor PATH.
"""

from __future__ import annotations

import shutil

import pytest


def pytest_sessionstart(session: pytest.Session) -> None:
    if shutil.which("ffmpeg") is not None:
        return
    pytest.exit(
        "ffmpeg is not on PATH.\n"
        "Both test suites build fixture videos with it, so the media tests would "
        "fail with a bare FileNotFoundError that names neither ffmpeg nor PATH.\n"
        "Install it (README > Install), open a NEW terminal so the PATH change is "
        "picked up, then run `styleloom doctor` to confirm.",
        returncode=pytest.ExitCode.USAGE_ERROR,
    )
