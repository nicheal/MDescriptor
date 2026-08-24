"""NEP model-backed descriptor adapter.

NEP descriptors are determined by a trained ``nep*.txt`` model rather than by
an independent species/cutoff configuration.  The native implementation
parses the model coefficients and computes the same scaled per-atom ``q``
vector exposed by NEPAdapters.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .core import DescriptorResult, StructureBatch, _as_batch, _cpp, _format_output, _merge_config
from ..models import NEP_MODEL


class NepCalculator:
    """Compute the per-atom NEP descriptor defined by a ``nep.txt`` model."""

    name = "NEP"

    def __init__(
        self,
        model_path: str | Path | None = None,
        config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        config = _merge_config(config, kwargs)
        if model_path is None:
            model_path = config.get("model_path", config.get("model_file"))
        if model_path is None:
            model_path = NEP_MODEL
        if str(model_path) == "":
            raise ValueError("NepCalculator model_path cannot be empty")
        self.model_path = str(Path(model_path).expanduser())
        self.dtype = str(config.get("dtype", "float64"))
        self.sparse = bool(config.get("sparse", False))
        self.num_threads = config.get("num_threads")
        if self.dtype not in {"float32", "float64"}:
            raise ValueError("dtype must be 'float32' or 'float64'")
        if self.num_threads is not None and int(self.num_threads) <= 0:
            raise ValueError("num_threads must be a positive integer or None")

        options = _cpp.NepOptions()
        options.model_path = self.model_path
        options.num_threads = 0 if self.num_threads is None else int(self.num_threads)
        self._native = _cpp.NepCalculator(options)
        self.species = tuple(int(value) for value in self._native.species)
        self._closed = False

    @property
    def feature_count(self) -> int:
        return int(self._native.feature_count)

    @property
    def descriptor_dim(self) -> int:
        """Alias used by NEPAdapters-compatible callers."""

        return self.feature_count

    def compute(
        self,
        value: StructureBatch | Sequence[Any] | Any,
        control: Any = None,
    ) -> DescriptorResult:
        if self._closed:
            raise RuntimeError("NEP calculator is closed")
        batch = _as_batch(value)
        values = self._native.compute(
            batch.numbers,
            batch.positions,
            batch.cells,
            batch.pbc,
            batch.offsets,
            control,
        )
        values = _format_output(np.asarray(values), self.dtype, self.sparse)
        return DescriptorResult(
            values,
            "atom",
            batch.ids,
            batch.offsets.copy(),
            self._labels(),
            self._metadata(),
        )

    def create(self, value: StructureBatch | Sequence[Any] | Any, control: Any = None) -> Any:
        return self.compute(value, control).values

    def close(self) -> None:
        self._closed = True
        self._native.close()

    def _labels(self) -> tuple[str, ...]:
        return tuple(f"nep:q{index + 1}" for index in range(self.feature_count))

    def _metadata(self) -> dict[str, Any]:
        return {
            "backend": "mdescriptor-cpp",
            "descriptor": self.name,
            "model_path": self.model_path,
            "species": self.species,
            "feature_count": self.feature_count,
            "radial_cutoff": float(self._native.radial_cutoff),
            "angular_cutoff": float(self._native.angular_cutoff),
            "n_max_radial": int(self._native.n_max_radial),
            "n_max_angular": int(self._native.n_max_angular),
            "l_max": int(self._native.l_max),
            "dtype": self.dtype,
            "sparse": self.sparse,
        }


NEP = NepCalculator
NEPCalculator = NepCalculator

__all__ = ["NEP", "NEPCalculator", "NepCalculator"]
