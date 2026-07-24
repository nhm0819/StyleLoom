"""Run endpoints: submit an input, poll, fetch artifacts.

`POST /runs/batch` takes several inputs at once because the deliverable is three
videos from three different inputs through one unchanged system. Running them in
one call also lets the hook recency penalty see the other two, which is what
keeps the three openings from converging.
"""

from __future__ import annotations

import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from ... import storage
from ...config import settings
from ...pipeline import runner
from ...schema import RunRecord

router = APIRouter(prefix="/runs", tags=["runs"])


def _new_run_id() -> str:
    return f"{datetime.now(timezone.utc):%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}"


def _save_upload(upload: UploadFile | None, run_id: str, label: str) -> Path | None:
    if upload is None or not upload.filename:
        return None
    dest = settings.uploads_dir / run_id
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / f"{label}_{upload.filename}"
    with path.open("wb") as fh:
        shutil.copyfileobj(upload.file, fh)
    return path


def _enqueue(
    tasks: BackgroundTasks,
    style_id: str,
    text: str,
    file_path: Path | None = None,
    bgm: Path | None = None,
    persona: Path | None = None,
    language: str = "ko",
) -> RunRecord:
    storage.load_style(style_id)  # 404 early rather than inside the background task
    run = RunRecord(run_id=_new_run_id(), style_id=style_id)
    storage.save_run(run)
    tasks.add_task(
        runner.execute,
        run_id=run.run_id,
        style_id=style_id,
        text=text,
        file_path=file_path,
        bgm=bgm,
        persona_ref=persona,
        language=language,
    )
    return run


@router.post("", response_model=RunRecord, status_code=202)
async def create_run(
    background: BackgroundTasks,
    style_id: str = Form(...),
    text: str = Form("", description="free-form input; may be empty if a file is given"),
    language: str = Form("ko"),
    input_file: UploadFile | None = File(None, description="image or video input"),
    bgm: UploadFile | None = File(None),
    persona: UploadFile | None = File(None, description="creator reference image"),
) -> RunRecord:
    if not text.strip() and input_file is None:
        raise HTTPException(400, "provide `text`, `input_file`, or both")
    run_id = _new_run_id()
    return _enqueue(
        background,
        style_id,
        text,
        _save_upload(input_file, run_id, "input"),
        _save_upload(bgm, run_id, "bgm"),
        _save_upload(persona, run_id, "persona"),
        language,
    )


class BatchRequest(BaseModel):
    style_id: str
    inputs: list[str] = Field(..., min_length=1, max_length=10)
    language: str = "ko"


@router.post("/batch", response_model=list[RunRecord], status_code=202)
def create_batch(req: BatchRequest, background: BackgroundTasks) -> list[RunRecord]:
    return [_enqueue(background, req.style_id, text, language=req.language) for text in req.inputs]


@router.get("", response_model=list[RunRecord])
def list_runs() -> list[RunRecord]:
    return storage.list_runs()


@router.get("/{run_id}", response_model=RunRecord)
def get_run(run_id: str) -> RunRecord:
    return storage.load_run(run_id)


@router.get("/{run_id}/video")
def get_video(run_id: str) -> FileResponse:
    run = storage.load_run(run_id)
    path = Path(run.artifacts.get("final_video", ""))
    if not path.exists():
        raise HTTPException(409, f"no video yet (status={run.status}, stage={run.stage})")
    return FileResponse(path, media_type="video/mp4", filename=f"{run_id}.mp4")


@router.get("/{run_id}/artifacts/{name}")
def get_artifact(run_id: str, name: str) -> FileResponse:
    if "/" in name or ".." in name:
        raise HTTPException(400, "invalid artifact name")
    path = storage.run_dir(run_id) / name
    if not path.exists():
        raise HTTPException(404, f"artifact not found: {name}")
    return FileResponse(path, media_type="application/json")
