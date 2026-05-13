"""
/similarity  — find top-K similar books for a query.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.config import get_config
from app.db import build_text_for_query
from app.indexer import SearchResult, search
from app.registry import get_registry

from logging import getLogger

logger = getLogger(__name__)
logger.info("Initializing similarity router")
router = APIRouter(prefix="/similarity", tags=["similarity"])


class SimilarityRequest(BaseModel):
    query: dict[str, Any] = Field(
        ...,
        description=(
            "Book fields used to build the query embedding. "
            "Any combination of: title, authors, synopsis, subjects, "
            "dewey_decimal, publisher, isbn_language, languages. "
            "Unknown fields are silently ignored."
        ),
        examples=[{
            "title": "Dune",
            "authors": "Frank Herbert",
            "synopsis": "A saga of politics, religion and ecology set on the desert planet Arrakis.",
            "subjects": "Science fiction, Epic",
        }],
    )
    model: str = Field("minilm", description="Name of the model to use (must match config).")
    top_k: int = Field(10, ge=1, le=100, description="Number of results to return.")


class SimilarityMatch(BaseModel):
    rank: int
    candidate_id: str
    score: float


class SimilarityResponse(BaseModel):
    model: str
    query_text: str
    results: list[SimilarityMatch]


@router.post("", response_model=SimilarityResponse)
def similarity(req: SimilarityRequest) -> SimilarityResponse:
    """
    Return the top-K books most similar to the supplied query fields.

    The query is converted to text using the same template configured for
    index build, ensuring comparable vector space representation.
    """
    cfg = get_config()

    try:
        cfg.model_by_name(req.model)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    registry = get_registry()
    if not registry.is_loaded(req.model):
        raise HTTPException(
            status_code=409,
            detail=f"Model '{req.model}' is not loaded. Call POST /models/{req.model}/load first.",
        )

    query_text = build_text_for_query(cfg, req.query)
    if not query_text.strip():
        raise HTTPException(status_code=422, detail="Query fields produced an empty text string.")

    try:
        raw_results: list[SearchResult] = search(cfg, registry, req.model, query_text, req.top_k)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=409, detail=str(exc))

    return SimilarityResponse(
        model=req.model,
        query_text=query_text,
        results=[
            SimilarityMatch(rank=r.rank, candidate_id=r.candidate_id, score=r.score)
            for r in raw_results
        ],
    )
