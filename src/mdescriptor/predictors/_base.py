"""Shared model, input, control, result, and lifetime handling."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from os import PathLike, fspath
from typing import Any

import numpy as np

from ..core.control import ComputeControl, _unwrap_native_control
from ..core.errors import (
    CancelledError,
    ClosedDescriptorError,
    DescriptorConfigError,
    DescriptorInputError,
    MDescriptorError,
    _InputValidationError,
    is_cuda_cancelled_error,
    is_native_cancelled_error,
    translate_backend_error,
)
from ..core.input import StructureBatch, StructureInput, coerce_batch
from ..core.options import ExecutionOptions
from ..core.prediction_result import PredictionResult
from ..models import (
    LoadedModel,
    ModelResolver,
    ModelResource,
    ModelSession,
    ResolvedModel,
    shared_loaded_model,
)
from ..models.session import identity_model_artifact


def _coerce_model(model: Any, default: ModelResource) -> ModelResource:
    if model is None:
        return default
    if isinstance(model, ModelResource):
        return model
    if isinstance(model, (str, PathLike)):
        raw = fspath(model)
        if isinstance(raw, bytes):
            raw = raw.decode()
        try:
            return ModelResource.explicit(raw)
        except DescriptorConfigError as exc:
            raise DescriptorConfigError(
                str(exc), code=exc.code, path=exc.path or ["model"], details=exc.details
            ) from exc
    raise DescriptorConfigError(
        "model must be None, a path string, a PathLike, or a ModelResource",
        code="invalid_option_type",
        path=["model"],
    )


def _cancelled(control: Any) -> bool:
    checker = getattr(control, "cancelled", None)
    return bool(checker() if callable(checker) else checker)


class _Predictor:
    """Own one resolved model and present a single prediction lifecycle."""

    name = "predictor"
    loader_schema = 1

    def __init__(
        self,
        model: Any,
        execution: ExecutionOptions,
        *,
        default_model: ModelResource,
        loader: Callable[[ResolvedModel], tuple[Any, Any]] | None = None,
    ) -> None:
        if not isinstance(execution, ExecutionOptions):
            raise DescriptorConfigError(
                "execution must be ExecutionOptions",
                code="invalid_option_type",
                path=["execution"],
            )
        self.execution = execution
        self.model_resource = _coerce_model(model, default_model)
        self.resolved_model = ModelResolver().resolve(self.model_resource)
        load = loader or identity_model_artifact
        self.loaded_model: LoadedModel | None = shared_loaded_model(
            self.resolved_model,
            loader_kind=self.name,
            loader_schema=self.loader_schema,
            loader=load,
        )
        self.session = ModelSession(self.loaded_model, device=execution.device)
        self._backend: Any = None
        self._closed = False
        self._model_snapshot: dict[str, Any] = {
            **self.model_resource.to_dict(),
            "resolved": {
                "digest": self.resolved_model.digest,
                "source": self.resolved_model.source,
            },
        }

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def metadata(self) -> Mapping[str, Any]:
        return deepcopy(
            {
                "descriptor": self.name,
                "backend": self._backend_name,
                "execution": {
                    "device": self.execution.device,
                    "num_threads": self.execution.num_threads,
                },
                "model": dict(self._model_snapshot),
                **self._metadata_fields(),
            }
        )

    def _metadata_fields(self) -> Mapping[str, Any]:
        return {}

    @property
    def _backend_name(self) -> str:
        return "mdescriptor-cpp" if self.execution.device == "cpu" else "mdescriptor-cuda"

    def predict(
        self,
        value: StructureInput,
        control: ComputeControl | None = None,
    ) -> PredictionResult:
        self._ensure_open()
        if _cancelled(control):
            raise CancelledError("prediction was cancelled")
        try:
            batch = coerce_batch(value)
        except DescriptorInputError:
            raise
        except (TypeError, ValueError) as exc:
            path = list(exc.path) if isinstance(exc, _InputValidationError) else ["input"]
            raise DescriptorInputError(str(exc), path=path) from exc

        self._validate_batch(batch)
        if batch.atoms == 0:
            energy, atom_energy, forces = self._empty_prediction(batch, control)
        else:
            try:
                backend = self._ensure_backend()
                raw = self._predict_backend(backend, batch, control)
            except CancelledError:
                raise
            except MDescriptorError:
                raise
            except Exception as exc:
                if is_native_cancelled_error(exc) or is_cuda_cancelled_error(exc):
                    raise CancelledError("prediction was cancelled") from exc
                if isinstance(exc, ValueError):
                    raise DescriptorInputError(str(exc), path=["input"]) from exc
                if self.execution.device == "cuda":
                    raise translate_backend_error(
                        exc,
                        unavailable_message="CUDA backend is unavailable",
                        failure_message="CUDA backend failed",
                        unavailable_details=False,
                    ) from exc
                raise
            if not isinstance(raw, (tuple, list)) or len(raw) != 3:
                raise MDescriptorError(
                    f"{self.name} predictor returned an invalid result",
                    code="backend_error",
                )
            energy, atom_energy, forces = raw
            if _cancelled(control):
                raise CancelledError("prediction was cancelled")

        metadata = dict(self.metadata)
        try:
            return PredictionResult(
                energy,
                atom_energy,
                forces,
                batch.ids,
                batch.offsets,
                metadata,
            )
        except (TypeError, ValueError) as exc:
            raise MDescriptorError(
                f"{self.name} predictor returned invalid arrays",
                code="backend_error",
            ) from exc

    def _validate_batch(self, batch: StructureBatch) -> None:
        del batch

    def _empty_prediction(
        self,
        batch: StructureBatch,
        control: Any,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        reset = getattr(control, "reset", None)
        mark_completed = getattr(control, "mark_completed", None)
        if callable(reset):
            reset(batch.structures)
        if callable(mark_completed):
            for _ in range(batch.structures):
                if _cancelled(control):
                    raise CancelledError("prediction was cancelled")
                mark_completed()
        return (
            np.zeros(batch.structures, dtype=np.float64),
            np.zeros(0, dtype=np.float64),
            np.zeros((0, 3), dtype=np.float64),
        )

    def _predict_backend(
        self,
        backend: Any,
        batch: StructureBatch,
        control: ComputeControl | None,
    ) -> Any:
        backend_control = (
            control if self.execution.device == "cuda" else _unwrap_native_control(control)
        )
        return backend.predict(batch, backend_control)

    def _ensure_backend(self) -> Any:
        if self._backend is None:
            self._backend = self._create_backend()
        return self._backend

    def _create_backend(self) -> Any:
        raise NotImplementedError

    def _ensure_open(self) -> None:
        if self._closed:
            raise ClosedDescriptorError(f"predictor {self.name!r} is closed")
        self.session.ensure_open()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        backend, self._backend = self._backend, None
        try:
            if backend is not None:
                close = getattr(backend, "close", None)
                if callable(close):
                    close()
        finally:
            try:
                self.session.close()
            finally:
                self._release_payload()
                self.loaded_model = None
                self.resolved_model = ResolvedModel(
                    self.resolved_model.path,
                    self.resolved_model.digest,
                    self.resolved_model.source,
                    None,
                )

    def _release_payload(self) -> None:
        """Drop model snapshots owned by one concrete predictor."""

    def __enter__(self) -> _Predictor:
        self._ensure_open()
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()
