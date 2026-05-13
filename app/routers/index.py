"""
/index  — build and inspect FAISS indexes.
"""

from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

from app.config import get_config
from app.db import load_all_candidates
from app.indexer import BuildResult, build_index, index_status
from app.registry import get_registry

from logging import getLogger

logger = getLogger(__name__)
logger.info("Initializing index router")
router = APIRouter(prefix="/index", tags=["index"])

# Track in-progress builds so concurrent requests don't kick off duplicates
_building: set[str] = set()


class BuildRequest(BaseModel):
    model: str


class BuildResponse(BaseModel):
    model_name: str
    doc_count: int
    dim: int
    build_time_s: float
    docs_per_sec: float


class AsyncBuildResponse(BaseModel):
    model: str
    message: str


def _do_build(model_name: str) -> None:
    cfg = get_config()
    registry = get_registry()

    # Auto-load the model if it is not yet in memory
    if not registry.is_loaded(model_name):
        model_cfg = cfg.model_by_name(model_name)
        registry.load(model_name, model_cfg.path)

    candidate_ids, texts = load_all_candidates(cfg)
    build_index(cfg, registry, model_name, candidate_ids, texts)
    _building.discard(model_name)


@router.post("/build", response_model=BuildResponse | AsyncBuildResponse)
def build(
    req: BuildRequest,
    background_tasks: BackgroundTasks,
    async_mode: bool = False,
) -> BuildResponse | AsyncBuildResponse:
    """
    Build (or rebuild) the FAISS index for a model.

    - `async_mode=false` (default): blocks until done, returns stats.
    - `async_mode=true`: starts in background, returns immediately.
    """
    cfg = get_config()
    logger.info(f"Building index for model: {req.model}")
    try:
        cfg.model_by_name(req.model)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    if req.model in _building:
        raise HTTPException(status_code=409, detail=f"Build for '{req.model}' already in progress.")

    if async_mode:
        _building.add(req.model)
        background_tasks.add_task(_do_build, req.model)
        return AsyncBuildResponse(model=req.model, message="build started in background")

    # Synchronous path
    registry = get_registry()
    if not registry.is_loaded(req.model):
        model_cfg = cfg.model_by_name(req.model)
        registry.load(req.model, model_cfg.path)

    candidate_ids, texts = load_all_candidates(cfg)
    result: BuildResult = build_index(cfg, registry, req.model, candidate_ids, texts)

    return BuildResponse(
        model_name=result.model_name,
        doc_count=result.doc_count,
        dim=result.dim,
        build_time_s=result.build_time_s,
        docs_per_sec=result.docs_per_sec,
    )


@router.get("/status")
def status() -> list[dict]:
    """Return build metadata for every configured model's index."""
    cfg = get_config()
    statuses = index_status(cfg)
    # Annotate which are currently building
    for s in statuses:
        if s.get("model_name") in _building:
            s["status"] = "building"
    return statuses
