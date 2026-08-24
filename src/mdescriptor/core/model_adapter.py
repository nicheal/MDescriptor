"""Shared public wrapper for model-backed legacy calculators."""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from .errors import (
    CancelledError,
    DescriptorConfigError,
    DescriptorError,
    DescriptorInputError,
    ModelLoadError,
    NativeCancelledError,
)
from .input import StructureBatch, StructureInput
from .legacy_adapter import LegacyDescriptorAdapter
from ..models import LoadedModel, ModelResolver, ModelResource, ModelSession


class ModelBackedAdapter(LegacyDescriptorAdapter):
    """Normalize the single public ``model=`` parameter."""

    model_keyword: ClassVar[str] = "model_path"
    default_model: ClassVar[ModelResource | Path | None] = None
    wrap_value_errors_as_config: ClassVar[bool] = False

    def __init__(self, model: str | Path | ModelResource | None = None, **kwargs: Any) -> None:
        self.model_resource: ModelResource | None = None
        self.session: ModelSession | None = None
        self._resolved_model_path = False
        if self.model_keyword in kwargs:
            raise DescriptorConfigError(
                f"{self.name} accepts a model resource through model=, not {self.model_keyword}="
            )
        if model is None and self.default_model is not None:
            model = self.default_model
        if model is not None:
            try:
                self.model_resource = (
                    model
                    if isinstance(model, ModelResource)
                    else ModelResource.from_value(model)
                )
            except (TypeError, ValueError) as exc:
                raise DescriptorConfigError(f"invalid model resource: {exc}") from exc
            try:
                resolved_path = ModelResolver().resolve(self.model_resource)
                self.model_resource = ModelResource(
                    resolved_path,
                    expected_sha256=self.model_resource.expected_sha256,
                    identifier=self.model_resource.identifier,
                )
                kwargs[self.model_keyword] = resolved_path
                self._resolved_model_path = True
            except ModelLoadError:
                raise
        try:
            super().__init__(**kwargs)
        except (DescriptorConfigError, DescriptorInputError, ModelLoadError):
            raise
        except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
            raise ModelLoadError(f"failed to load {self.name} model") from exc
        if self.model_resource is None:
            path = getattr(self._legacy, "model_path", None)
            if path is not None:
                self.model_resource = ModelResource.from_value(path)
        if self.model_resource is not None:
            model_path = self.model_resource.path.resolve()
            self.session = ModelSession(
                LoadedModel(
                    path=model_path,
                    config=getattr(self._legacy, "_config", None),
                    weights=getattr(self._legacy, "_native", None),
                ),
                device=str(getattr(self._legacy, "device_name", "cpu")),
            )
        try:
            resolved_features = int(getattr(self._legacy, "feature_count"))
        except (AttributeError, TypeError, ValueError, RuntimeError):
            resolved_features = 0
        if resolved_features > 0:
            self._feature_count = resolved_features

    @property
    def model_path(self) -> str | None:
        """Resolved model path exposed as stable metadata, not legacy passthrough."""

        return None if self.model_resource is None else str(self.model_resource.path)

    def close(self) -> None:
        if self.closed:
            return
        try:
            super().close()
        finally:
            if self.session is not None:
                self.session.close()

    def _ensure_model_session(self) -> None:
        if self.session is not None:
            self.session.ensure_open()

    def _compute_batch(self, batch: StructureBatch, *, control: Any = None):
        self._ensure_model_session()
        return super()._compute_batch(batch, control=control)


class TorchModelAdapter(ModelBackedAdapter):
    """Model wrapper with the DPA spin/charge override contract."""

    def compute(
        self,
        value: StructureInput,
        control: Any = None,
        *,
        spin: np.ndarray | None = None,
        charge_spin: np.ndarray | None = None,
    ):
        self._ensure_open()
        self._ensure_model_session()
        try:
            batch = self._as_batch(value)
        except DescriptorInputError:
            raise
        except (TypeError, ValueError) as exc:
            raise DescriptorInputError(str(exc)) from exc
        try:
            result = self._legacy.compute(
                batch,
                control,
                spin=spin,
                charge_spin=charge_spin,
            )
        except NativeCancelledError as exc:
            raise CancelledError("descriptor computation was cancelled") from exc
        except DescriptorError:
            raise
        except ValueError as exc:
            raise DescriptorInputError(str(exc)) from exc
        except RuntimeError as exc:
            raise ModelLoadError(f"failed to compute {self.name}") from exc
        except Exception as exc:
            raise ModelLoadError(f"failed to compute {self.name}") from exc
        return self._adapt_result(result)
