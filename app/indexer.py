"""
FAISS-backed vector index — one index per embedding model.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from app.config import AppConfig
from app.registry import ModelRegistry
from logging import getLogger
logger = getLogger(__name__)
logger.info("Initializing indexer")


def _index_dir(cfg: AppConfig, model_name: str) -> Path:
    return Path(cfg.index.storage_dir) / model_name


def _faiss_path(cfg: AppConfig, model_name: str) -> Path:
    return _index_dir(cfg, model_name) / "index.faiss"


def _candidate_ids_path(cfg: AppConfig, model_name: str) -> Path:
    return _index_dir(cfg, model_name) / "candidate_ids.json"


def _meta_path(cfg: AppConfig, model_name: str) -> Path:
    return _index_dir(cfg, model_name) / "meta.json"



@dataclass
class BuildResult:
    model_name: str
    doc_count: int
    dim: int
    build_time_s: float
    docs_per_sec: float


def build_index(
    cfg: AppConfig,
    registry: ModelRegistry,
    model_name: str,
    candidate_ids: list[str],
    texts: list[str],
) -> BuildResult:
    """
    Embed *texts* with *model_name* and persist a FAISS IndexFlatIP to disk.

    Vectors are expected to be L2-normalised (registry.encode normalises by
    default), so inner product equals cosine similarity.
    """
    logger.info(f"Building index for model: {model_name}")
    if not candidate_ids or not texts:
        raise ValueError("candidate_ids and texts must be non-empty")
    if len(candidate_ids) != len(texts):
        raise ValueError("candidate_ids and texts must have the same length")

    batch_size = cfg.index.batch_size
    t0 = time.perf_counter()

    # Encode in batches, show progress via show_progress=True
    vectors: np.ndarray = registry.encode(
        model_name,
        texts,
        batch_size=batch_size,
        normalize=True,
        show_progress=True,
    )
    vectors = vectors.astype(np.float32)

    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    elapsed = time.perf_counter() - t0

    # Persist
    out_dir = _index_dir(cfg, model_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(_faiss_path(cfg, model_name)))

    with _candidate_ids_path(cfg, model_name).open("w") as f:
        json.dump(candidate_ids, f)

    meta: dict[str, Any] = {
        "model_name": model_name,
        "model_path": cfg.model_by_name(model_name).path,
        "doc_count": len(candidate_ids),
        "dim": dim,
        "build_time_s": round(elapsed, 3),
        "built_at": datetime.now(timezone.utc).isoformat(),
    }
    with _meta_path(cfg, model_name).open("w") as f:
        json.dump(meta, f, indent=2)

    return BuildResult(
        model_name=model_name,
        doc_count=len(candidate_ids),
        dim=dim,
        build_time_s=round(elapsed, 3),
        docs_per_sec=round(len(candidate_ids) / elapsed, 1),
    )



@dataclass
class SearchResult:
    candidate_id: str
    score: float
    rank: int


def search(
    cfg: AppConfig,
    registry: ModelRegistry,
    model_name: str,
    query_text: str,
    top_k: int = 10,
) -> list[SearchResult]:
    """
    Embed *query_text* and return the top-K most similar books from the
    pre-built FAISS index for *model_name*.
    """
    logger.info(f"Searching for model: {model_name}")
    faiss_file = _faiss_path(cfg, model_name)
    candidate_ids_file = _candidate_ids_path(cfg, model_name)

    if not faiss_file.exists():
        raise FileNotFoundError(
            f"No index found for model '{model_name}'. "
            f"Call POST /index/build first."
        )

    logger.info(f"Reading index from: {faiss_file}")
    index = faiss.read_index(str(faiss_file))

    with candidate_ids_file.open() as f:
        candidate_ids: list[str] = json.load(f)

    logger.info(f"Encoding query text: {query_text}")
    query_vec = registry.encode(model_name, [query_text], normalize=True)
    query_vec = query_vec.astype(np.float32)

    logger.info(f"Searching index for top {top_k} results")
    k = min(top_k, index.ntotal)
    scores, indices = index.search(query_vec, k)

    results: list[SearchResult] = []
    
    for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
        if idx < 0:
            continue
        results.append(SearchResult(candidate_id=candidate_ids[idx], score=float(score), rank=rank))
       
    logger.info(f"Found {len(results)} results")

    return results



def index_status(cfg: AppConfig) -> list[dict[str, Any]]:
    """
    Return metadata for every index that has been built on disk.
    Models without a built index are reported as absent.
    """
    statuses: list[dict[str, Any]] = []
    for model_cfg in cfg.models:
        meta_file = _meta_path(cfg, model_cfg.name)
        if meta_file.exists():
            with meta_file.open() as f:
                meta = json.load(f)
            statuses.append({"status": "ready", **meta})
        else:
            statuses.append({"status": "not_built", "model_name": model_cfg.name})
    return statuses
