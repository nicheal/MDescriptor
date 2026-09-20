"""NEP model-backed descriptor adapter.

NEP descriptors are determined by a trained ``nep*.txt`` model rather than by
an independent species/cutoff configuration.  The native implementation
parses the model coefficients and computes the same scaled per-atom ``q``
vector exposed by NEPAdapters.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ...core.errors import ModelLoadError
from ...core.result import (
    DescriptorLevel,
    normalize_metadata,
)
from ...models import NEP_MODEL
from ._base import _cpp_metadata, _Kernel, _optional_threads, _validate_dtype
from .core import StructureBatch, _cpp


class NepKernel(_Kernel):
    """Compute the per-atom NEP descriptor defined by a ``nep.txt`` model."""

    name = "NEP"

    def __init__(
        self,
        model_path: str | Path | None = None,
        model_digest: str | None = None,
        model_data: bytes | None = None,
        dtype: str = "float64",
        sparse: bool = False,
        num_threads: int | None = None,
    ) -> None:
        if model_path is None:
            model_path = NEP_MODEL
        if str(model_path) == "":
            raise ValueError("NEP model path cannot be empty")
        self.model_path = str(Path(model_path).expanduser())
        self.dtype = _validate_dtype(dtype)
        self.sparse = bool(sparse)
        self.num_threads = _optional_threads(num_threads)

        options = _cpp.NepOptions()
        options.model_path = self.model_path
        if model_digest is not None:
            options.model_digest = str(model_digest)
        if model_data is not None:
            try:
                options.model_data = bytes(model_data)
            except AttributeError as exc:
                raise ModelLoadError(
                    "native extension lacks immutable model snapshot support; rebuild MDescriptor"
                ) from exc
        options.num_threads = 0 if self.num_threads is None else int(self.num_threads)
        self._native = _cpp.NepCalculator(options)
        self.species = tuple(int(value) for value in self._native.species)
        self._labels_cache = tuple(f"nep:q{index + 1}" for index in range(self.feature_count))
        self._metadata_template = normalize_metadata(
            self._metadata(), DescriptorLevel.ATOM, self.feature_count
        )
        self._closed = False

    @property
    def feature_count(self) -> int:
        return int(self._native.feature_count)

    def _ensure_native(self, batch: StructureBatch) -> None:
        if self._closed:
            raise RuntimeError("NEP calculator is closed")

    def _labels(self) -> tuple[str, ...]:
        return self._labels_cache

    def _metadata(self) -> dict[str, Any]:
        return _cpp_metadata(
            self.name,
            model_path=self.model_path,
            species=self.species,
            feature_count=self.feature_count,
            radial_cutoff=float(self._native.radial_cutoff),
            angular_cutoff=float(self._native.angular_cutoff),
            n_max_radial=int(self._native.n_max_radial),
            n_max_angular=int(self._native.n_max_angular),
            l_max=int(self._native.l_max),
            dtype=self.dtype,
            sparse=self.sparse,
        )


__all__ = ["NepKernel"]
