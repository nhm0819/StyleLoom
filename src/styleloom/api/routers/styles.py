"""Style endpoints: extract a style DNA from reference videos, read it, edit it.

`PUT /styles/{id}` exists because automated extraction gets pacing and grade
right but sometimes mislabels camera vocabulary. A human should be able to
correct the schema without re-running extraction -- the schema is the contract,
not the extractor.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from ... import storage
from ...config import settings
from ...pipeline.analyze import extract_style
from ...providers.llm import get_llm
from ...schema import StyleSchema

router = APIRouter(prefix="/styles", tags=["styles"])

VIDEO_SUFFIXES = {".mp4", ".mov", ".webm", ".mkv", ".m4v"}


@router.post("/extract", response_model=StyleSchema, status_code=201)
async def extract(
    style_id: str = Form(..., description="identifier to store this style under"),
    files: list[UploadFile] = File(..., description="one or more reference videos"),
    notes: str = Form(""),
) -> StyleSchema:
    ref_dir = settings.uploads_dir / "refs" / style_id
    ref_dir.mkdir(parents=True, exist_ok=True)

    saved: list[Path] = []
    for upload in files:
        suffix = Path(upload.filename or "ref.mp4").suffix.lower()
        if suffix not in VIDEO_SUFFIXES:
            raise HTTPException(400, f"unsupported reference format: {suffix}")
        dest = ref_dir / (upload.filename or f"ref{len(saved)}.mp4")
        with dest.open("wb") as fh:
            shutil.copyfileobj(upload.file, fh)
        saved.append(dest)

    if not saved:
        raise HTTPException(400, "no reference videos provided")

    try:
        style = extract_style(style_id, saved, get_llm(), notes=notes)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc

    storage.save_style(style)
    return style


@router.get("", response_model=list[str])
def list_styles() -> list[str]:
    return storage.list_styles()


@router.get("/{style_id}", response_model=StyleSchema)
def get_style(style_id: str) -> StyleSchema:
    return storage.load_style(style_id)


@router.put("/{style_id}", response_model=StyleSchema)
def update_style(style_id: str, style: StyleSchema) -> StyleSchema:
    if style.style_id != style_id:
        raise HTTPException(400, "style_id in body does not match path")
    storage.load_style(style_id)  # 404 if absent
    storage.save_style(style)
    return style
