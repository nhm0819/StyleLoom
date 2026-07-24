"""Filesystem-backed store. Chosen over a DB because every intermediate
artifact must be human-readable for grading; `data/runs/<run_id>/` *is* the
deliverable."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel

from .config import settings
from .schema import RunRecord, StyleSchema


def _write_json(path: Path, model: BaseModel) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(model.model_dump_json(indent=2), encoding="utf-8")
    return path


# --- styles ---------------------------------------------------------------- #


def style_dir(style_id: str) -> Path:
    return settings.styles_dir / style_id


def save_style(style: StyleSchema) -> Path:
    return _write_json(style_dir(style.style_id) / "style.json", style)


def load_style(style_id: str) -> StyleSchema:
    path = style_dir(style_id) / "style.json"
    if not path.exists():
        raise FileNotFoundError(f"style not found: {style_id}")
    return StyleSchema.model_validate_json(path.read_text(encoding="utf-8"))


def list_styles() -> list[str]:
    return sorted(p.name for p in settings.styles_dir.iterdir() if (p / "style.json").exists())


# --- runs ------------------------------------------------------------------ #


def run_dir(run_id: str) -> Path:
    return settings.runs_dir / run_id


def save_run(run: RunRecord) -> Path:
    return _write_json(run_dir(run.run_id) / "run.json", run)


def load_run(run_id: str) -> RunRecord:
    path = run_dir(run_id) / "run.json"
    if not path.exists():
        raise FileNotFoundError(f"run not found: {run_id}")
    return RunRecord.model_validate_json(path.read_text(encoding="utf-8"))


def list_runs() -> list[RunRecord]:
    runs = []
    for p in settings.runs_dir.iterdir():
        if (p / "run.json").exists():
            runs.append(load_run(p.name))
    return sorted(runs, key=lambda r: r.created_at, reverse=True)


def save_artifact(run_id: str, name: str, model: BaseModel) -> Path:
    return _write_json(run_dir(run_id) / name, model)


def save_raw(run_id: str, name: str, payload: dict) -> Path:
    path = run_dir(run_id) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
