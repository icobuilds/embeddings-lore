"""
ModelRegistry — loads, caches, and unloads sentence-transformers models.

One SentenceTransformer instance is kept alive in memory per model name.
Loading is thread-safe via a per-name lock so concurrent requests don't
trigger a double-load race.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from sentence_transformers import SentenceTransformer

if TYPE_CHECKING:
    import numpy as np


@dataclass
class _Entry:
    model: SentenceTransformer
    model_path: str
    load_time_s: float


class ModelRegistry:
    def __init__(self) -> None:
        self._models: dict[str, _Entry] = {}
        self._locks: dict[str, threading.Lock] = {}
        self._global_lock = threading.Lock()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get_lock(self, name: str) -> threading.Lock:
        with self._global_lock:
            if name not in self._locks:
                self._locks[name] = threading.Lock()
            return self._locks[name]

    # ── Public API ────────────────────────────────────────────────────────────

    def load(self, name: str, model_path: str, device: str = "cpu") -> None:
        """
        Load *hf_path* from HuggingFace Hub (or local cache) and register
        it under *name*.  Idempotent — does nothing if already loaded.
        """
        lock = self._get_lock(name)
        with lock:
            if name in self._models:
                return

            import time
            t0 = time.perf_counter()
            model = SentenceTransformer(model_path, device=device)
            elapsed = time.perf_counter() - t0

            self._models[name] = _Entry(model=model, model_path=model_path, load_time_s=elapsed)

    def unload(self, name: str) -> None:
        """Remove a model from memory."""
        lock = self._get_lock(name)
        with lock:
            self._models.pop(name, None)

    def is_loaded(self, name: str) -> bool:
        return name in self._models

    def get(self, name: str) -> SentenceTransformer:
        entry = self._models.get(name)
        if entry is None:
            raise KeyError(f"Model '{name}' is not loaded. Call /models/{name}/load first.")
        return entry.model

    def encode(
        self,
        name: str,
        texts: list[str],
        batch_size: int = 64,
        normalize: bool = True,
        show_progress: bool = False,
    ) -> "np.ndarray":
        """
        Encode *texts* with the named model.
        Returns a float32 ndarray of shape (len(texts), dim).
        Vectors are L2-normalised by default so dot-product == cosine sim.
        """
        model = self.get(name)
        return model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=normalize,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )

    def status(self) -> list[dict]:
        """Return a list of dicts describing every registered (loaded) model."""
        return [
            {
                "name": name,
                "model_path": e.model_path,
                "load_time_s": round(e.load_time_s, 3),
                "embedding_dim": e.model.get_sentence_embedding_dimension(),
            }
            for name, e in self._models.items()
        ]


# ── Module-level singleton (shared across FastAPI lifespan) ──────────────────

_registry: ModelRegistry | None = None


def get_registry() -> ModelRegistry:
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
    return _registry
