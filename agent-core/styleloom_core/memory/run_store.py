"""Run persistence.

`data/runs/<run_id>/` *is* the deliverable. Every stage artifact is a JSON file
someone can open and read, so a finished run is fully inspectable and a failed
run shows exactly which stage broke. A database would hide precisely the
intermediate state that makes the system gradeable, and would then need an
export step to produce what the filesystem already gives.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel

from ..config import Settings
from ..errors import NotFoundError
from ..schema import RunRecord


class RunStore:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def dir_for(self, run_id: str) -> Path:
        return self.settings.runs_dir / run_id

    def path_for(self, run_id: str) -> Path:
        return self.dir_for(run_id) / "run.json"

    def save(self, run: RunRecord) -> Path:
        path = self.path_for(run.run_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(run.model_dump_json(indent=2), encoding="utf-8")
        return path

    def load(self, run_id: str) -> RunRecord:
        path = self.path_for(run_id)
        if not path.exists():
            raise NotFoundError(f"run not found: {run_id!r}")
        return RunRecord.model_validate_json(path.read_text(encoding="utf-8"))

    def list_records(self, style_id: str | None = None, limit: int = 50) -> list[RunRecord]:
        root = self.settings.runs_dir
        if not root.exists():
            return []
        # Sort directory names before loading: run ids are time-ordered, so this
        # reads `limit` files rather than every run on disk.
        names = sorted((p.name for p in root.iterdir() if (p / "run.json").exists()),
                       reverse=True)
        out: list[RunRecord] = []
        for name in names:
            record = self.load(name)
            if style_id and record.style_id != style_id:
                continue
            out.append(record)
            if len(out) >= limit:
                break
        return out

    def save_artifact(self, run_id: str, name: str, model: BaseModel) -> Path:
        path = self.dir_for(run_id) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(model.model_dump_json(indent=2), encoding="utf-8")
        return path

    def save_raw(self, run_id: str, name: str, payload: dict) -> Path:
        path = self.dir_for(run_id) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        return path
