"""
/models  — list configured models and manage load/unload.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.config import get_config
from app.registry import get_registry
from logging import getLogger
logger = getLogger(__name__)
logger.info("Initializing models router")
router = APIRouter(prefix="/models", tags=["models"])


class ModelStatus(BaseModel):
    name: str
    hf_path: str
    loaded: bool
    embedding_dim: int | None = None
    load_time_s: float | None = None

class LoadRequest(BaseModel):
    name: str

class LoadResponse(BaseModel):
    name: str
    message: str
    load_time_s: float


@router.get("", response_model=list[ModelStatus])
def list_models() -> list[ModelStatus]:
    """List all models declared in config and whether each is currently loaded."""
    logger.info("Listing models")
    cfg = get_config()
    registry = get_registry()
    loaded_info = {s["name"]: s for s in registry.status()}

    result: list[ModelStatus] = []
    for m in cfg.models:
        info = loaded_info.get(m.name)
        result.append(
            ModelStatus(
                name=m.name,
                hf_path=m.path,
                loaded=info is not None,
                embedding_dim=info["embedding_dim"] if info else None,
                load_time_s=info["load_time_s"] if info else None,
            )
        )
    return result


@router.post("/load", response_model=LoadResponse)
def load_model(req: LoadRequest) -> LoadResponse:
    """Load a model into memory (downloads from HuggingFace Hub on first call)."""
    cfg = get_config()
    logger.info(f"Loading model: {req.name}")
    name = req.name
    try:
        model_cfg = cfg.model_by_name(name)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    registry = get_registry()
    registry.load(name, model_cfg.path)

    info = next((s for s in registry.status() if s["name"] == name), None)
    load_time = info["load_time_s"] if info else 0.0
    return LoadResponse(
        name=name,
        message="loaded" if load_time > 0 else "already loaded",
        load_time_s=load_time,
    )


@router.delete("/{name}", status_code=204)
def unload_model(name: str) -> None:
    """Unload a model from memory to free RAM."""
    cfg = get_config()
    try:
        cfg.model_by_name(name)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    get_registry().unload(name)
