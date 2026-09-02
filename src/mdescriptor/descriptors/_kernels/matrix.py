"""Python adapter for the structure-level matrix descriptor family."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from .core import DescriptorResult, StructureBatch, _as_batch, _cpp
from .structure import _StructureKernel

_MATRIX_KINDS = {
    "sine": 0,
    "ewald": 1,
    "coulomb": 2,
}


class _MatrixKernel(_StructureKernel):
    kind = "sine"

    def __init__(
        self,
        n_atoms_max: int | None = None,
        permutation: str = "sorted_l2",
        exponent: float = 2.4,
        num_threads: int | None = None,
    ):
        self.n_atoms_max = n_atoms_max
        self.permutation = str(permutation)
        self.exponent = float(exponent)
        self.num_threads = 0 if num_threads is None else int(num_threads)
        if self.n_atoms_max is not None and int(self.n_atoms_max) <= 0:
            raise ValueError("n_atoms_max must be positive")
        if self.permutation not in {"none", "sorted_l2", "eigenspectrum"}:
            raise ValueError("permutation must be 'none', 'sorted_l2', or 'eigenspectrum'")
        if self.num_threads < 0:
            raise ValueError("num_threads must be non-negative")

    def compute(self, value: StructureBatch | Sequence[Any] | Any, control: Any = None) -> DescriptorResult:
        batch = _as_batch(value)
        counts = np.diff(batch.offsets)
        max_atoms = int(self.n_atoms_max or (counts.max() if len(counts) else 0))
        columns = max_atoms if self.permutation == "eigenspectrum" else max_atoms * max_atoms
        values: Any
        if not len(counts) or not np.any(counts):
            values = np.zeros((len(counts), columns), dtype=np.float64)
        else:
            values = _cpp.compute_matrix(
                batch.numbers, batch.positions, batch.cells, batch.pbc, batch.offsets,
                max_atoms, self.permutation, self.exponent, _MATRIX_KINDS[self.kind],
                getattr(self, "accuracy", 1e-5), getattr(self, "w", 1.0),
                float(getattr(self, "r_cut", 0.0) or 0.0),
                float(getattr(self, "g_cut", 0.0) or 0.0),
                float(getattr(self, "a", 0.0) or 0.0), self.num_threads, control,
            )
        self._feature_count = int(values.shape[1])
        return DescriptorResult(
            np.asarray(values, dtype=np.float64), self.level, batch.ids, None,
            tuple(f"{self.name}:{index}" for index in range(values.shape[1])),
            {"backend": "mdescriptor-cpp", "descriptor": self.name},
        )


class CoulombMatrixKernel(_MatrixKernel):
    name = "CoulombMatrix"
    kind = "coulomb"


class SineMatrixKernel(_MatrixKernel):
    name = "SineMatrix"
    kind = "sine"


class EwaldSumMatrixKernel(_MatrixKernel):
    name = "EwaldSumMatrix"
    kind = "ewald"

    def __init__(
        self, n_atoms_max: int | None = None, permutation: str = "sorted_l2",
        accuracy: float = 1e-5, w: float = 1.0, r_cut: float | None = None,
        g_cut: float | None = None, a: float | None = None,
        num_threads: int | None = None,
    ):
        super().__init__(
            n_atoms_max=n_atoms_max,
            permutation=permutation,
            num_threads=num_threads,
        )
        self.accuracy, self.w, self.r_cut, self.g_cut, self.a = float(accuracy), float(w), r_cut, g_cut, a
        if not 0.0 < self.accuracy < 1.0 or self.w <= 0.0:
            raise ValueError("accuracy must be between zero and one and w must be positive")
        if (r_cut is None) != (g_cut is None):
            raise ValueError("r_cut and g_cut must be provided together")


__all__ = ["CoulombMatrixKernel", "SineMatrixKernel", "EwaldSumMatrixKernel"]
