"""NEP energy and force predictor."""

from __future__ import annotations

from typing import Any

from ..core.errors import DescriptorInputError, ModelLoadError
from ..core.options import ExecutionOptions
from ..models import NEP_RESOURCE
from ..models.session import identity_model_artifact
from ._base import _Predictor

_DEFAULT_EXECUTION = ExecutionOptions()


class NEP(_Predictor):
    """Predict NEP total/per-atom energies and forces for structure batches."""

    name = "NEP"

    def __init__(
        self,
        model: Any = None,
        execution: ExecutionOptions = _DEFAULT_EXECUTION,
    ) -> None:
        super().__init__(
            model,
            execution,
            default_model=NEP_RESOURCE,
            loader=identity_model_artifact,
        )
        self._options = {
            "model_path": str(self.resolved_model.path),
            "model_digest": self.resolved_model.digest,
            "model_data": self.resolved_model.content,
        }
        if execution.device == "cpu":
            try:
                self._backend = self._make_cpu_backend()
                self._species = tuple(int(value) for value in self._backend.species)
            except ModelLoadError:
                self.close()
                raise
            except Exception as exc:
                self.close()
                raise ModelLoadError("failed to load NEP model for prediction") from exc

    def _make_cpu_backend(self) -> Any:
        from .. import _native

        options = _native.NepOptions()
        options.model_path = self._options["model_path"]
        options.model_digest = self._options["model_digest"]
        options.model_data = self._options["model_data"]
        options.num_threads = self.execution.num_threads or 0
        predictor = getattr(_native, "NepPredictor", None)
        if predictor is None:
            raise ModelLoadError("native module does not provide NepPredictor; rebuild MDescriptor")
        return predictor(options)

    def _create_backend(self) -> Any:
        if self.execution.device == "cpu":
            return self._make_cpu_backend()
        from .._runtime import create_cuda_predictor

        return create_cuda_predictor("NEP", dict(self._options))

    def _validate_batch(self, batch: Any) -> None:
        if batch.spins is not None:
            raise DescriptorInputError(
                "NEP prediction does not support spins",
                code="unsupported_input",
                path=["input", "spins"],
            )
        if batch.charge_spin is not None:
            raise DescriptorInputError(
                "NEP prediction does not support charge_spin",
                code="unsupported_input",
                path=["input", "charge_spin"],
            )
        if self.execution.device == "cpu":
            allowed = set(self._species)
            unsupported = sorted(set(int(number) for number in batch.numbers) - allowed)
            if unsupported:
                raise DescriptorInputError(
                    "NEP prediction model does not support atomic number(s): "
                    + ", ".join(map(str, unsupported)),
                    code="unsupported_species",
                    path=["input", "numbers"],
                    details={"atomic_numbers": unsupported},
                )

    def _release_payload(self) -> None:
        self._options.clear()


__all__ = ["NEP"]
