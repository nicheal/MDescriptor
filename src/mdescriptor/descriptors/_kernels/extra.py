"""C++-backed histogram descriptors.

This module is deliberately a thin Python adapter. Descriptor kernels and
matrix permutations are computed by ``mdescriptor._native``; Python retains
the public array contract, while ASE is used only by :class:`StructureBatch`
input packing.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np

from ...core.species import require_species, validate_batch_species
from .core import DescriptorResult, StructureBatch, _as_batch, _cpp
from .matrix import CoulombMatrixKernel, EwaldSumMatrixKernel, SineMatrixKernel
from .mbtr_config import resolve_mbtr_config
from .structure import _StructureKernel


def _periodic_neighbors(
    batch: StructureBatch,
    structure: int,
    cutoff: float,
    *,
    include_self: bool = False,
) -> list[list[tuple[int, np.ndarray, float, tuple[int, int, int]]]]:
    """Compatibility view over the native neighbor graph used by tests/tools."""
    from math import sqrt

    start, stop = int(batch.offsets[structure]), int(batch.offsets[structure + 1])
    offsets, atoms, shifts, displacements, distance2 = _cpp.build_neighbor_graph(
        batch.numbers[start:stop], batch.positions[start:stop], batch.cells[structure], batch.pbc[structure], float(cutoff)
    )
    result: list[list[tuple[int, np.ndarray[Any, Any], float, tuple[int, int, int]]]] = []
    for center in range(stop - start):
        neighbors = []
        for index in range(int(offsets[center]), int(offsets[center + 1])):
            shift: tuple[int, int, int] = tuple(int(value) for value in shifts[index])  # type: ignore[assignment]
            atom = int(atoms[index])
            if not include_self and atom == center and shift == (0, 0, 0):
                continue
            neighbors.append((
                atom,
                np.asarray(displacements[index], dtype=np.float64),
                sqrt(max(float(distance2[index]), 0.0)),
                shift,
            ))
        result.append(neighbors)
    return result


class MBTRKernel(_StructureKernel):
    name = "MBTR"
    local = False
    result_level = "structure"

    def __init__(
        self, species: Iterable[int] | None = None, geometry: dict[str, Any] | None = None,
        grid: dict[str, Any] | None = None, weighting: dict[str, Any] | None = None,
        periodic: bool = True, normalize_gaussians: bool = True, normalization: str = "none",
        num_threads: int | None = None,
    ):
        self.species = tuple(sorted(require_species(species, descriptor=self.name)))
        self.geometry = dict(geometry or {"function": "distance"})
        self.grid = dict(grid or {"min": 0.0, "max": 6.0, "n": 50, "sigma": 0.1})
        self.weighting = dict(weighting or {"function": "exp", "scale": 0.5, "threshold": 1e-3})
        if not periodic:
            raise ValueError("only periodic MBTR is supported")
        if normalization not in {"none", "l2", "n_atoms", "valle_oganov"}:
            raise ValueError("unsupported MBTR normalization")
        self.normalize_gaussians, self.normalization = bool(normalize_gaussians), normalization
        self.num_threads = 0 if num_threads is None else int(num_threads)
        self._mbtr_config = resolve_mbtr_config(
            species=self.species,
            geometry=self.geometry,
            grid=self.grid,
            weighting=self.weighting,
            periodic=periodic,
            normalize_gaussians=self.normalize_gaussians,
            normalization=self.normalization,
            local=self.local,
            num_threads=self.num_threads,
        )
        self._feature_count = self._mbtr_config.feature_count

    def _options(self, batch: StructureBatch) -> tuple[tuple[int, ...], dict[str, Any]]:
        species = validate_batch_species(batch, self.species, descriptor=self.name)
        self.species = species
        return species, self._mbtr_config.native_kwargs()

    def _cuda_payload(self) -> dict[str, Any]:
        """Pass the same canonical configuration used by the CPU adapter."""

        return {"mbtr_config": self._mbtr_config.cuda_payload()}

    def compute(self, value: StructureBatch | Sequence[Any] | Any, control: Any = None) -> DescriptorResult:
        batch = _as_batch(value)
        species, options = self._options(batch)
        values = np.asarray(
            _cpp.compute_mbtr(
                batch.numbers, batch.positions, batch.cells, batch.pbc,
                batch.offsets, list(species), **options, control=control,
            ),
            dtype=np.float64,
        )
        self._feature_count = int(values.shape[1])
        return DescriptorResult(
            values, self.result_level, batch.ids,
            batch.offsets.copy() if self.local else None,
            tuple(f"{self.name}:{index}" for index in range(values.shape[1])),
            {"backend": "mdescriptor-cpp", "descriptor": self.name, "species": species},
        )


class LMBTRKernel(MBTRKernel):
    name = "LMBTR"
    local = True
    result_level = "atom"


class ValleOganovKernel(MBTRKernel):
    name = "ValleOganov"

    def __init__(
        self,
        species: Iterable[int] | None = None,
        function: str = "distance",
        n: int = 50,
        sigma: float = 0.1,
        r_cut: float = 6.0,
        geometry: dict[str, Any] | None = None,
        grid: dict[str, Any] | None = None,
        weighting: dict[str, Any] | None = None,
        periodic: bool = True,
        normalize_gaussians: bool = True,
        normalization: str | None = None,
        num_threads: int | None = None,
    ):
        if function == "distance":
            geometry = geometry or {"function": "distance"}
            grid = grid or {"min": 0.0, "max": r_cut, "n": n, "sigma": sigma}
            weighting = weighting or {"function": "inverse_square", "r_cut": r_cut}
        elif function == "angle":
            geometry = geometry or {"function": "angle"}
            grid = grid or {"min": 0.0, "max": 180.0, "n": n, "sigma": sigma}
            weighting = weighting or {"function": "smooth_cutoff", "r_cut": r_cut}
        else:
            raise ValueError("function must be 'distance' or 'angle'")
        super().__init__(
            species=species,
            geometry=geometry,
            grid=grid,
            weighting=weighting,
            periodic=periodic,
            normalize_gaussians=normalize_gaussians,
            normalization="valle_oganov" if normalization is None else normalization,
            num_threads=num_threads,
        )


# Keep the old private import path as a compatibility re-export.  The matrix
# implementation itself lives in ``_kernels.matrix``.
__all__ = [
    "CoulombMatrixKernel",
    "EwaldSumMatrixKernel",
    "SineMatrixKernel",
    "MBTRKernel",
    "LMBTRKernel",
    "ValleOganovKernel",
]
