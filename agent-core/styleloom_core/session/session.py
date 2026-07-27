"""Run session: the mutable state of one execution.

Holds the run record, the caller's raw inputs, and the artifact bag that tools
read from and write to. Tools receive `(ctx, session)` and take what they need
by artifact key; the plan declares those keys so the wiring stays checkable
without every tool growing a bespoke signature.

The session is also the only writer of run state, so status transitions happen
in one place.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

from ..errors import ToolError
from ..memory import RunStore
from ..schema import RunInputs, RunRecord, RunStatus

M = TypeVar("M", bound=BaseModel)

# Artifact key -> filename on disk. A run directory is meant to be read by a
# person, so the names are stable and boring.
ARTIFACT_FILES = {
    # Snapshotted, not just referenced: style.json is meant to be hand-corrected
    # between runs, so a run's folder has to record the version it actually used
    # or the run stops being reproducible.
    "style": "style_used.json",
    "brief": "brief.json",
    "casting": "casting.json",
    "outline": "outline.json",
    "hook": "hook.json",
    "storyboard": "storyboard.json",
    "render": "render.json",
    "assemble": "assemble.json",
    "qc": "qc_report.json",
}


class RunSession:
    def __init__(
        self,
        record: RunRecord,
        inputs: RunInputs,
        store: RunStore,
    ) -> None:
        self.record = record
        self.inputs = inputs
        self.store = store
        self.artifacts: dict[str, BaseModel] = {}

    # --- identity ----------------------------------------------------------

    @property
    def run_id(self) -> str:
        return self.record.run_id

    @property
    def style_id(self) -> str:
        return self.record.style_id

    @property
    def dir(self) -> Path:
        return self.store.dir_for(self.run_id)

    def workspace(self, *parts: str) -> Path:
        """A subdirectory of this run's folder, created on demand."""
        path = self.dir.joinpath(*parts)
        path.mkdir(parents=True, exist_ok=True)
        return path

    # --- artifacts ---------------------------------------------------------

    def put(self, key: str, artifact: BaseModel) -> Path | None:
        """Store an artifact in memory and, if it has a filename, on disk."""
        self.artifacts[key] = artifact
        filename = ARTIFACT_FILES.get(key)
        if filename is None:
            return None
        path = self.store.save_artifact(self.run_id, filename, artifact)
        self.record.artifacts[key] = str(path)
        return path

    def get(self, key: str, expected: type[M]) -> M:
        """Fetch an artifact, asserting its type.

        Raises rather than returning None: a missing artifact means the plan was
        executed out of order, which is a bug in the plan, not a runtime
        condition to handle.
        """
        artifact = self.artifacts.get(key)
        if artifact is None:
            raise ToolError(
                f"artifact {key!r} is not available. "
                f"Have: {sorted(self.artifacts)}. Check the plan's step order."
            )
        if not isinstance(artifact, expected):
            raise ToolError(
                f"artifact {key!r} is {type(artifact).__name__}, expected {expected.__name__}"
            )
        return artifact

    def has(self, key: str) -> bool:
        return key in self.artifacts

    def save_raw(self, name: str, payload: dict[str, Any]) -> Path:
        return self.store.save_raw(self.run_id, name, payload)

    # --- state -------------------------------------------------------------

    def mark(self, **fields: Any) -> RunRecord:
        self.record.touch(**fields)
        self.store.save(self.record)
        return self.record

    def start_stage(self, name: str) -> None:
        self.mark(status=RunStatus.RUNNING, stage=name)

    def fail(self, stage: str, exc: BaseException) -> RunRecord:
        return self.mark(
            status=RunStatus.FAILED,
            stage=stage,
            error=f"{type(exc).__name__}: {exc}",
        )

    def finish(self) -> RunRecord:
        return self.mark(status=RunStatus.DONE, stage="done")
