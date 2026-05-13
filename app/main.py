"""
emblore — embedding model benchmark app for book similarity search.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_config
from app.registry import get_registry
from app.routers import index, models, similarity
from logging import getLogger

logger = getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    On startup: load the app config and eagerly load any models that are
    already indexed (so /similarity works without a manual /load call).
    """
    logger.info("Starting lifespan")
    cfg = get_config()
    registry = get_registry()

    # Auto-load models that already have a built index on disk
    from app.indexer import _faiss_path  # noqa: PLC0415

    for model_cfg in cfg.models:
        if _faiss_path(cfg, model_cfg.name).exists():
            try:
                registry.load(model_cfg.name, model_cfg.path)
                logger.info(f"auto-loaded '{model_cfg.name}' (index found on disk)")
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"could not auto-load '{model_cfg.name}': {exc}")

    yield

    # Cleanup (no-op for in-process models, but good practice)
    logger.info("Stopping lifespan")


app = FastAPI(
    title="emblore",
    description=(
        "Benchmark self-hosted embedding models for book similarity search. "
        "Load models, build per-model FAISS indexes from a MariaDB book catalog, "
        "and query top-K similar books."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(models.router)
app.include_router(index.router)
app.include_router(similarity.router)


@app.get("/", tags=["health"])
def health() -> dict:
    cfg = get_config()
    registry = get_registry()
    return {
        "service": "emblore",
        "version": "0.1.0",
        "configured_models": [m.name for m in cfg.models],
        "loaded_models": [s["name"] for s in registry.status()],
    }
