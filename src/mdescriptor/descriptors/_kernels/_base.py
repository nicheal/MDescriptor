"""Shared lifecycle seam for the native descriptor kernels.

Thread/dtype validation, the calculator ``close`` protocol, the backend
metadata preamble, and the flat-batch compute template live here so the
individual kernel modules stay focused on their native option mapping.
"""

from __future__ import annotations

from typing import Any

from ...core.input import StructureBatch, coerce_batch
from ...core.result import DescriptorResult, format_values
from ...core.species import require_species, validate_batch_species

_as_batch = coerce_batch


def _threads(num_threads: int | None) -> int:
    """Coerce an optional thread count into the native non-negative int."""

    value = 0 if num_threads is None else int(num_threads)
    if value < 0:
        raise ValueError("num_threads must be non-negative")
    return value


def _optional_threads(num_threads: int | None) -> int | None:
    """Validate a thread count that may stay ``None`` for auto-selection."""

    if num_threads is not None and int(num_threads) <= 0:
        raise ValueError("num_threads must be a positive integer or None")
    return num_threads


def _validate_dtype(dtype: str) -> str:
    value = str(dtype)
    if value not in {"float32", "float64"}:
        raise ValueError("dtype must be 'float32' or 'float64'")
    return value


def _cpp_metadata(name: str, **fields: Any) -> dict[str, Any]:
    """Backend metadata preamble shared by every C++ kernel."""

    return {"backend": "mdescriptor-cpp", "descriptor": name, **fields}


class _Kernel:
    """Shared tail of the stateful native calculator kernels.

    Subclasses either adopt the :meth:`compute` template (supplying
    ``_ensure_native``, ``_labels``, and ``_metadata``) or override it when
    their result shape differs.
    """

    name = "descriptor"
    dtype = "float64"
    sparse = False
    _closed = False
    _native: Any = None
    _metadata_template: Any = None

    @property
    def feature_count(self) -> int:
        return int(getattr(self, "_feature_count", 0))

    def compute(self, value: Any, control: Any = None) -> DescriptorResult:
        batch = _as_batch(value)
        self._ensure_native(batch)
        values = format_values(
            self._native.compute(
                batch.numbers,
                batch.positions,
                batch.cells,
                batch.pbc,
                batch.offsets,
                control,
            ),
            dtype=self.dtype,
            sparse=self.sparse,
        )
        return DescriptorResult(
            values,
            "atom",
            batch.ids,
            batch.offsets.copy(),
            self._labels(),
            self._result_metadata(),
        )

    def _ensure_native(self, batch: StructureBatch) -> None:
        raise NotImplementedError

    def _result_metadata(self) -> Any:
        return self._metadata_template if self._metadata_template is not None else self._metadata()

    def close(self) -> None:
        self._closed = True
        if self._native is not None:
            self._native.close()


class _AtomKernel(_Kernel):
    """Atom-level kernel seam with a fixed species declaration."""

    def __init__(self, species: Any = None, num_threads: int | None = None):
        self.species = require_species(species, descriptor=self.name)
        self.num_threads = _threads(num_threads)

    def _species_for(self, batch: StructureBatch) -> tuple[int, ...]:
        return validate_batch_species(batch, self.species, descriptor=self.name)


class _StructureKernel(_Kernel):
    """Structure-level kernel seam whose width resolves on first compute."""

    level = "structure"
