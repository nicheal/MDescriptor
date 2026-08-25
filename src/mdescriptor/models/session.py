"""Immutable loaded artifacts and per-descriptor model sessions."""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields, is_dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any
from weakref import WeakValueDictionary

from ..core.errors import ClosedDescriptorError, ModelLoadError
from .resolver import ResolvedModel


class _TensorSnapshot:
    """Private CPU tensor storage whose public reads are independent clones."""

    __slots__ = ("__value",)

    def __init__(self, value: Any) -> None:
        self.__value = value.detach().cpu().clone()

    def materialize(self) -> Any:
        return self.__value.clone()


class _ArraySnapshot:
    """Private NumPy storage whose public reads are independent copies."""

    __slots__ = ("__value",)

    def __init__(self, value: Any) -> None:
        self.__value = value.copy()
        self.__value.setflags(write=False)

    def materialize(self) -> Any:
        return self.__value.copy()


class _FrozenMapping(Mapping[Any, Any]):
    """A read-only mapping that never exposes its mutable stored values."""

    __slots__ = ("__values",)

    def __init__(self, values: Mapping[Any, Any]) -> None:
        self.__values = MappingProxyType(dict(values))

    def __getitem__(self, key: Any) -> Any:
        return _materialize(self.__values[key])

    def __iter__(self):
        return iter(self.__values)

    def __len__(self) -> int:
        return len(self.__values)


def _is_torch_tensor(value: Any) -> bool:
    """Detect Torch tensors without importing the optional dependency."""

    module = type(value).__module__.split(".", 1)[0]
    return module == "torch" and all(
        hasattr(value, attribute) for attribute in ("detach", "cpu", "clone")
    )


def _is_numpy_array(value: Any) -> bool:
    """Detect NumPy arrays without making the model extra mandatory."""

    return (
        type(value).__module__.split(".", 1)[0] == "numpy"
        and hasattr(value, "setflags")
        and hasattr(value, "copy")
    )


def _freeze(value: Any) -> Any:
    """Recursively snapshot model state into immutable public containers."""

    if isinstance(value, _FrozenMapping):
        return value
    if isinstance(value, Mapping):
        return _FrozenMapping({key: _freeze(item) for key, item in value.items()})
    if _is_torch_tensor(value):
        return _TensorSnapshot(value)
    if _is_numpy_array(value):
        return _ArraySnapshot(value)
    if is_dataclass(value) and not isinstance(value, type):
        dataclass_params = getattr(type(value), "__dataclass_params__", None)
        if dataclass_params is None or not dataclass_params.frozen:
            raise ModelLoadError(
                f"model configuration dataclass {type(value).__name__} must be frozen"
            )
        frozen_fields = {
            item.name: _freeze(getattr(value, item.name)) for item in fields(value)
        }
        try:
            return replace(value, **frozen_fields)
        except (TypeError, ValueError) as exc:
            raise ModelLoadError(
                f"model configuration dataclass {type(value).__name__} cannot be frozen"
            ) from exc
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze(item) for item in value)
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, memoryview):
        return bytes(value)
    if value is None or isinstance(value, (bool, int, float, complex, str, bytes, Path)):
        return value
    if type(value).__module__.split(".", 1)[0] == "numpy" and hasattr(value, "item"):
        return _freeze(value.item())
    raise ModelLoadError(
        f"loaded model state contains unsupported mutable value {type(value).__name__}"
    )


def _materialize(value: Any) -> Any:
    """Return a safe per-read value without exposing shared mutable storage."""

    if isinstance(value, (_TensorSnapshot, _ArraySnapshot)):
        return value.materialize()
    if isinstance(value, _FrozenMapping):
        return value
    if isinstance(value, tuple):
        return tuple(_materialize(item) for item in value)
    if isinstance(value, frozenset):
        return frozenset(_materialize(item) for item in value)
    return value


def _materialize_mutable(value: Any) -> Any:
    """Build an isolated loader payload from the immutable shared snapshot."""

    if isinstance(value, (_TensorSnapshot, _ArraySnapshot)):
        return value.materialize()
    if isinstance(value, _FrozenMapping):
        return {
            key: _materialize_mutable(item)
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        # Checkpoint readers may require list-valued configuration fields.
        return [_materialize_mutable(item) for item in value]
    if isinstance(value, frozenset):
        return {_materialize_mutable(item) for item in value}
    return value


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

    def __post_init__(self) -> None:
        object.__setattr__(self, "config", _freeze(self.config))
        object.__setattr__(self, "weights", _freeze(self.weights))

    def materialize_weights(self) -> Any:
        """Return an isolated mutable payload for one runtime loader."""

        return _materialize_mutable(self.weights)


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
