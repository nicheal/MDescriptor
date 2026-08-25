"""Immutable loaded artifacts and per-descriptor model sessions."""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from weakref import WeakValueDictionary

from ..core.errors import ClosedDescriptorError, ModelLoadError
from .resolver import ResolvedModel


@dataclass(frozen=True)
class LoadedModel:
    """Validated CPU-side model state safe to share between instances."""

    path: Path
    digest: str
    source: str
    loader_kind: str
    loader_schema: int
    config: Any
    weights: Any


_CACHE: WeakValueDictionary[tuple[str, int, str], LoadedModel] = WeakValueDictionary()
_CACHE_LOCK = threading.RLock()


def shared_loaded_model(
    resolved: ResolvedModel,
    *,
    loader_kind: str,
    loader_schema: int,
    loader: Callable[[ResolvedModel], tuple[Any, Any]],
) -> LoadedModel:
    """Return a live shared artifact, loading it once per identity.

    The weak cache intentionally keeps no ownership after the last session is
    closed.  Failed loaders never enter the cache.
    """

    key = (str(loader_kind), int(loader_schema), resolved.digest)
    with _CACHE_LOCK:
        cached = _CACHE.get(key)
        if cached is not None:
            return cached
        try:
            config, weights = loader(resolved)
            loaded = LoadedModel(
                path=resolved.path,
                digest=resolved.digest,
                source=resolved.source,
                loader_kind=str(loader_kind),
                loader_schema=int(loader_schema),
                config=config,
                weights=weights,
            )
        except ModelLoadError:
            raise
        except Exception as exc:
            raise ModelLoadError(
                f"failed to load {loader_kind} model from {resolved.path}"
            ) from exc
        _CACHE[key] = loaded
        return loaded


def clear_loaded_model_cache() -> None:
    """Clear the process cache, primarily for isolated tests."""

    with _CACHE_LOCK:
        _CACHE.clear()


def discard_loaded_model(model: LoadedModel) -> None:
    """Remove one artifact after a later adapter-stage validation failure."""

    key = (model.loader_kind, model.loader_schema, model.digest)
    with _CACHE_LOCK:
        if _CACHE.get(key) is model:
            del _CACHE[key]


class ModelSession:
    """Device-bound execution state owned by one descriptor instance."""

    __slots__ = ("_model", "device", "runtime_dtype", "runtime", "closed")

    def __init__(
        self,
        model: LoadedModel,
        *,
        device: str = "cpu",
        runtime_dtype: str | None = None,
        runtime: Any = None,
    ) -> None:
        self._model: LoadedModel | None = model
        self.device = str(device)
        self.runtime_dtype = None if runtime_dtype is None else str(runtime_dtype)
        self.runtime = runtime
        self.closed = False

    @property
    def model(self) -> LoadedModel:
        if self._model is None:
            raise ClosedDescriptorError("model session is closed")
        return self._model

    def ensure_open(self) -> None:
        if self.closed or self._model is None:
            raise ClosedDescriptorError("model session is closed")

    def close(self) -> None:
        if self.closed:
            return
        self.runtime = None
        self.runtime_dtype = None
        self._model = None
        self.closed = True

    def __enter__(self) -> ModelSession:
        self.ensure_open()
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()


__all__ = [
    "LoadedModel",
    "ModelSession",
    "clear_loaded_model_cache",
    "discard_loaded_model",
    "shared_loaded_model",
]
