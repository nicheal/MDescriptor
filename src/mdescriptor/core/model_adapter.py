"""Deep model-backed adapter seam: resource, artifact, and session."""

from __future__ import annotations

from collections.abc import Mapping
from os import PathLike
from types import MappingProxyType
from typing import Any, ClassVar

from ..models import (
    LoadedModel,
    ModelResolver,
    ModelResource,
    ModelSession,
    ResolvedModel,
    discard_loaded_model,
    shared_loaded_model,
)
from .adapter import DescriptorAdapter
from .errors import (
    DescriptorConfigError,
    DescriptorInputError,
    MDescriptorError,
    ModelLoadError,
    NativeCancelledError,
)
from .input import StructureBatch
from .options import DescriptorConfiguration
from .result import DescriptorResult


class ModelBackedAdapter(DescriptorAdapter):
    """Normalize ``model=`` and own one per-instance model session."""

    model_keyword: ClassVar[str] = "model_path"
    default_model: ClassVar[ModelResource | None] = None
    loader_kind: ClassVar[str] = "generic"
    loader_schema: ClassVar[int] = 1
    wrap_value_errors_as_config: ClassVar[bool] = False

    def _initialize_public(self, options: Mapping[str, Any]) -> None:
        options = dict(options)
        model = options.pop("model", None)
        self.model_resource: ModelResource | None = None
        self.resolved_model: ResolvedModel | None = None
        self.loaded_model: LoadedModel | None = None
        self.session: ModelSession | None = None
        self._preloaded_model_weights: Any = None
        self._resolved_model_path = False

        if self.model_keyword in options:
            raise DescriptorConfigError(
                f"{self.name} accepts a model resource through model=, not {self.model_keyword}="
            )
        if model is None and self.default_model is not None:
            model = self.default_model
        if model is not None:
            self.model_resource = self._coerce_model_resource(model)
            self.resolved_model = ModelResolver().resolve(self.model_resource)
            self.loaded_model = shared_loaded_model(
                self.resolved_model,
                loader_kind=self.loader_kind or self.name,
                loader_schema=self.loader_schema,
                loader=self._load_shared_artifact,
            )
            self._preloaded_model_weights = self.loaded_model.weights
            options[self.model_keyword] = self.resolved_model.path
            self._resolved_model_path = True

        try:
            super()._initialize(options)
        except (DescriptorConfigError, DescriptorInputError, ModelLoadError):
            if self.loaded_model is not None:
                discard_loaded_model(self.loaded_model)
            raise
        except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
            if self.loaded_model is not None:
                discard_loaded_model(self.loaded_model)
            raise ModelLoadError(f"failed to load {self.name} model") from exc

        snapshot = dict(self.configuration.parameters)
        snapshot.pop(self.model_keyword, None)
        if self.model_resource is None:
            snapshot["model"] = None
        else:
            snapshot["model"] = self.model_resource.to_dict()
        self._configuration = DescriptorConfiguration(
            self.configuration.schema_version,
            self.name,
            snapshot,
        )

        if self.loaded_model is not None:
            device = self._execution_options.device
            native_runtime = getattr(self._kernel, "_native", None)
            self.session = ModelSession(
                self.loaded_model,
                device=device,
                runtime_dtype=getattr(native_runtime, "compute_dtype", None),
                runtime=native_runtime,
            )
        try:
            resolved_features = int(self._kernel.feature_count)
        except (AttributeError, TypeError, ValueError, RuntimeError):
            resolved_features = 0
        if resolved_features > 0:
            self._feature_count = resolved_features

    @staticmethod
    def _coerce_model_resource(model: Any) -> ModelResource:
        if isinstance(model, ModelResource):
            return model
        if isinstance(model, (str, bytes)):
            raise DescriptorConfigError(
                "model must be None, a PathLike, or a ModelResource; use Path(...) for paths"
            )
        if isinstance(model, PathLike):
            return ModelResource.explicit(model)
        raise DescriptorConfigError(
            "model must be None, a PathLike, or a ModelResource"
        )

    def _load_shared_artifact(self, resolved: ResolvedModel) -> tuple[Any, Any]:
        """Load immutable CPU-side artifact data using the safe checkpoint path."""

        if resolved.path.suffix.lower() == ".pt":
            try:
                import torch

                weights = torch.load(
                    str(resolved.path),
                    map_location="cpu",
                    weights_only=True,
                )
            except ImportError as exc:  # pragma: no cover - optional model extra
                raise ModelLoadError(
                    f"{self.name} requires the project's PyTorch dependency"
                ) from exc
            if self.name == "DPA4":
                from ..descriptors.model_backed.dpa4._vendor.official import (
                    validate_official_dpa4_checkpoint,
                )

                return validate_official_dpa4_checkpoint(weights), weights
            if self.name == "DPA4C":
                from ..descriptors.model_backed.dpa4c._vendor.model import (
                    validate_dpa4c_checkpoint,
                )

                return validate_dpa4c_checkpoint(weights), weights
            return self._checkpoint_identity(resolved, weights), weights
        try:
            return MappingProxyType(
                {"format": resolved.path.suffix.lower().lstrip("."), "digest": resolved.digest}
            ), resolved.path.read_bytes()
        except OSError as exc:
            raise ModelLoadError(f"cannot read model resource {resolved.path}") from exc

    def _checkpoint_identity(self, resolved: ResolvedModel, checkpoint: Any) -> Any:
        """Keep a small immutable identity beside the shared CPU checkpoint."""

        descriptor_type = None
        type_map: tuple[str, ...] = ()
        if isinstance(checkpoint, Mapping):
            model = checkpoint.get("model", checkpoint)
            if isinstance(model, Mapping):
                extra = model.get("_extra_state")
                if isinstance(extra, Mapping):
                    params = extra.get("model_params")
                    if isinstance(params, Mapping):
                        descriptor = params.get("descriptor")
                        if isinstance(descriptor, Mapping):
                            descriptor_type = descriptor.get("type")
                        raw_type_map = params.get("type_map", ())
                        if isinstance(raw_type_map, (list, tuple)):
                            type_map = tuple(str(item) for item in raw_type_map)
        return MappingProxyType(
            {
                "format": resolved.path.suffix.lower().lstrip("."),
                "digest": resolved.digest,
                "descriptor_type": descriptor_type,
                "type_map": type_map,
            }
        )

    @property
    def model_path(self) -> str | None:
        """Concrete model path retained as diagnostic information."""

        return None if self.resolved_model is None else str(self.resolved_model.path)

    def close(self) -> None:
        if self.closed:
            return
        try:
            super().close()
        finally:
            if self.session is not None:
                self.session.close()
            self.loaded_model = None
            self._preloaded_model_weights = None

    def _ensure_model_session(self) -> None:
        if self.session is not None:
            self.session.ensure_open()

    def _compute_batch(self, batch: StructureBatch, *, control: Any = None):
        self._ensure_model_session()
        return super()._compute_batch(batch, control=control)


class TorchModelAdapter(ModelBackedAdapter):
    """Model wrapper using spin/charge fields carried by ``StructureBatch``."""

    def _compute_batch(
        self,
        batch: StructureBatch,
        *,
        control: Any = None,
    ) -> DescriptorResult:
        self._ensure_open()
        self._ensure_model_session()
        try:
            result = self._kernel.compute(batch, control)
        except NativeCancelledError:
            raise
        except MDescriptorError:
            raise
        except ValueError as exc:
            raise DescriptorInputError(str(exc)) from exc
        except RuntimeError as exc:
            raise ModelLoadError(f"failed to compute {self.name}") from exc
        except Exception as exc:
            raise ModelLoadError(f"failed to compute {self.name}") from exc
        adapted = self._adapt_result(result)
        if not isinstance(adapted, DescriptorResult):  # pragma: no cover - defensive
            raise MDescriptorError(
                f"{self.name} returned {type(adapted).__name__}, expected DescriptorResult"
            )
        return adapted


__all__ = ["ModelBackedAdapter", "TorchModelAdapter"]
