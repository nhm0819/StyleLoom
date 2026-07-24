"""FastAPI entrypoint.

Job execution uses FastAPI `BackgroundTasks` and a filesystem run store rather
than Celery/Redis. Rationale: a render is a single long HTTP-bound job with no
fan-out, and the artifacts have to be readable on disk anyway. If you need
multi-worker durability, swap `runner.execute` onto a queue -- nothing else in
the codebase changes.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from ..config import settings
from .routers import hooks, runs, styles

app = FastAPI(
    title="StyleLoom",
    version="0.1.0",
    description=(
        "Decompose a reference short-form video into a reusable style schema, "
        "then generate new videos in that style from arbitrary inputs."
    ),
)

app.include_router(styles.router)
app.include_router(hooks.router)
app.include_router(runs.router)


@app.get("/health", tags=["meta"])
def health() -> dict:
    return {
        "status": "ok",
        "llm_provider": settings.llm_provider,
        "video_provider": settings.video_provider,
        "data_dir": str(settings.data_dir.resolve()),
    }


@app.exception_handler(FileNotFoundError)
def not_found(_, exc: FileNotFoundError) -> JSONResponse:
    return JSONResponse(status_code=404, content={"detail": str(exc)})
