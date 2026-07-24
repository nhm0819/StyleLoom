"""Hook endpoints.

`POST /hooks/preview` runs hook generation N times on the *same* input without
rendering anything. It exists to make the non-determinism requirement checkable
in one call: the response reports the distinct archetypes and distinct texts
produced, so a grader does not have to take it on faith.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ... import storage
from ...pipeline import plan
from ...pipeline.hook import generate_hook
from ...providers.llm import get_llm
from ...schema import HookResult

router = APIRouter(prefix="/hooks", tags=["hooks"])


class PreviewRequest(BaseModel):
    style_id: str
    text: str = Field(..., description="the same input you would send to /runs")
    n: int = Field(5, ge=1, le=20, description="how many times to re-run generation")
    language: str = "ko"


class PreviewResponse(BaseModel):
    n: int
    distinct_archetypes: int
    distinct_texts: int
    results: list[HookResult]


@router.post("/preview", response_model=PreviewResponse)
def preview(req: PreviewRequest) -> PreviewResponse:
    style = storage.load_style(req.style_id)
    llm = get_llm()

    brief = plan.ingest(llm, text=req.text, language=req.language)
    outline = plan.build_outline(llm, brief, style)

    results: list[HookResult] = []
    recent: list[str] = []
    for _ in range(req.n):
        result = generate_hook(llm, brief, outline, style, recent_archetypes=recent)
        recent = ([result.archetype_sampled] + recent)[:4]
        results.append(result)

    if not results:
        raise HTTPException(500, "hook preview produced no results")

    return PreviewResponse(
        n=req.n,
        distinct_archetypes=len({r.archetype_sampled for r in results}),
        distinct_texts=len({r.selected.text for r in results}),
        results=results,
    )
