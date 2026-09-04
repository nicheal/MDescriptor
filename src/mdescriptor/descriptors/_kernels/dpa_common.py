"""Shared dispatch for the native DPA kernel adapters."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar, Literal

import numpy as np

from ...core.errors import ModelLoadError
from ...core.input import StructureBatch
from ...core.result import DescriptorResult
from ...models.resolver import ResolvedModel
from ..model_backed.dpa import compute_batch, load_dpa_checkpoint, new_runtime
from ..model_backed.graph import _ATOMIC_SYMBOLS


def compute_native_batch(
    calculator: Any,
    type_mapper: Any,
    batch: StructureBatch,
    control: Any,
) -> np.ndarray:
    """Initialize progress, map a validated batch, and call either DPA calculator."""

    if control is not None:
        reset = getattr(control, "reset", None)
        if callable(reset):
            reset(batch.structures)

    symbols: list[str] = []
    for number in batch.numbers.tolist():
        try:
            symbols.append(_ATOMIC_SYMBOLS[int(number)])
        except KeyError as exc:
            raise ValueError(
                f"atomic number {number} is absent from the checkpoint type_map"
            ) from exc
    try:
        type_indices = type_mapper.symbols_to_atype(symbols).astype(np.int32, copy=False)
    except KeyError as exc:
        raise ValueError(
            f"element {exc.args[0]!r} is absent from the checkpoint type_map"
        ) from exc
    return calculator.compute(
        batch.numbers,
        batch.positions,
        batch.cells,
        batch.pbc,
        batch.offsets,
        type_indices,
        control,
    )


def _type_numbers(type_map: Any) -> np.ndarray:
    """Translate a checkpoint type order to public atomic numbers."""

    numbers_by_symbol = {symbol: number for number, symbol in _ATOMIC_SYMBOLS.items()}
    return np.asarray(
        [int(numbers_by_symbol.get(str(symbol), -1)) for symbol in type_map],
        dtype=np.int32,
    )


class DpaKernelBase:
    """Own the lifecycle shared by the DPA4 and DPA4C kernel adapters.

    Subclasses only provide model-specific payload, labels, and metadata.  The
    model path, checkpoint validation, runtime creation, native calculator
    selection, progress-aware compute, and close semantics stay behind this
    private seam so the two public descriptors cannot drift apart.
    """

    name: ClassVar[str]
    checkpoint_descriptor: ClassVar[Literal["DPA4", "DPA4C"]]
    runtime_descriptor_name: ClassVar[str]
    native_calculator_name: ClassVar[str]
    default_model: ClassVar[Path]
    accepts_preloaded_checkpoint: ClassVar[bool] = True
    configuration_defaults: ClassVar[dict[str, Any]] = {}

    @classmethod
    def load_model_artifact(
        cls,
        resolved: ResolvedModel,
    ) -> tuple[Any, Any]:
        """Load and validate the checkpoint for this concrete DPA variant."""

        return load_dpa_checkpoint(
            resolved.path,
            expected_descriptor=cls.checkpoint_descriptor,
        )

    def _initialize_dpa(
        self,
        model_path: str | Path | None,
        num_threads: int | None,
        checkpoint: Mapping[str, Any] | None,
    ) -> None:
        path = self.default_model if model_path is None else Path(model_path)
        if not str(path):
            raise ValueError(f"{self.name} model path cannot be empty")
        self.model_path = str(path.expanduser())
        if checkpoint is None:
            _info, checkpoint = load_dpa_checkpoint(
                Path(self.model_path),
                expected_descriptor=self.checkpoint_descriptor,
            )
        self._native = new_runtime(Path(self.model_path), checkpoint)
        descriptor = self._native.descriptor
        if descriptor.__class__.__name__ != self.runtime_descriptor_name:
            raise ValueError(f"checkpoint did not construct a {self.name} descriptor")

        requested_num_threads = num_threads
        if num_threads is None:
            num_threads = 1
        if (
            isinstance(num_threads, bool)
            or not isinstance(num_threads, int)
            or num_threads <= 0
        ):
            raise ValueError("num_threads must be a positive integer")
        self.num_threads = int(num_threads)
        self._metadata_num_threads = (
            None if requested_num_threads is None else self.num_threads
        )
        self._cpp = None
        payload = self._native_payload(descriptor, num_threads=self.num_threads)
        self._cuda_model_payload = payload
        if payload is not None:
            from mdescriptor import _native as native

            calculator = getattr(native, self.native_calculator_name, None)
            if calculator is None:  # pragma: no cover - guarded by payload builders
                raise ModelLoadError(
                    f"native module does not provide {self.native_calculator_name}"
                )
            self._cpp = calculator(payload)
        self._closed = False

    def _native_payload(
        self,
        descriptor: Any,
        *,
        num_threads: int,
    ) -> dict[str, Any] | None:
        raise NotImplementedError

    def _cuda_payload(self) -> dict[str, Any]:
        """Return the private device handoff without changing public state."""

        return {
            "version": 1,
            "model": self._cuda_model_payload,
            "labels": self._labels(),
            "type_numbers": _type_numbers(self._native.type_map),
            "feature_count": self.feature_count,
        }

    @property
    def feature_count(self) -> int:
        return int(self._native.dim_out)

    @property
    def descriptor_dim(self) -> int:
        return self.feature_count

    def compute(self, value: Any, control: Any = None) -> DescriptorResult:
        if self._closed or self._native is None:
            raise RuntimeError(f"{self.name} descriptor is closed")
        if self._cpp is None:
            values = compute_batch(self._native, value, control=control)
        else:
            values = compute_native_batch(self._cpp, self._native, value, control)
        return DescriptorResult(
            values,
            "atom",
            value.ids,
            value.offsets.copy(),
            self._labels(),
            self._metadata(),
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._cpp is not None:
            self._cpp.close()
            self._cpp = None
        self._native = None

    def _labels(self) -> tuple[str, ...]:
        raise NotImplementedError

    def _metadata(self) -> dict[str, Any]:
        raise NotImplementedError

    def _metadata_base(self, descriptor: Any) -> dict[str, Any]:
        """Return model-independent metadata shared by both DPA variants."""

        backend = self.name.lower()
        return {
            "backend": (
                f"mdescriptor-{backend}-cpp"
                if self._cpp is not None
                else f"mdescriptor-{backend}-numpy"
            ),
            "descriptor": self.name,
            "type_map": tuple(self._native.type_map),
            "rcut": float(self._native.rcut),
            "channels": int(descriptor.channels),
            "lmax": int(descriptor.lmax),
            "basis_type": str(getattr(descriptor, "basis_type", "unknown")),
            "n_radial": int(getattr(descriptor, "n_radial", 0)),
            "precision": str(getattr(descriptor, "precision", "float64")),
            "use_spin": getattr(descriptor, "use_spin", None),
            "add_chg_spin_ebd": bool(getattr(descriptor, "add_chg_spin_ebd", False)),
        }


__all__ = ["DpaKernelBase", "compute_native_batch"]
