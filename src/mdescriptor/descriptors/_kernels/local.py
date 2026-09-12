"""C++-backed local descriptor adapters.

The names mirror the supported local descriptor families, but no external package is
imported or required. Every value path enters ``_native`` directly.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np

from ...core.result import pair_samples
from ._base import _AtomKernel, _cpp_metadata, _threads
from .core import DescriptorResult, StructureBatch, _as_batch, _cpp


def _atom_result(values: np.ndarray, batch: StructureBatch, name: str, species: tuple[int, ...], *, level: str = "atom", offsets: np.ndarray | None = None, metadata: dict[str, Any] | None = None) -> DescriptorResult:
    values = np.asarray(values, dtype=np.float64)
    details = _cpp_metadata(name, species=species)
    if metadata:
        details.update(metadata)
    return DescriptorResult(values, level, batch.ids, offsets, tuple(f"{name}:{index}" for index in range(values.shape[1])), details)


class AtomicCompositionKernel(_AtomKernel):
    name = "AtomicComposition"

    def __init__(
        self,
        species: Iterable[int] | None = None,
        per_system: bool = True,
        num_threads: int | None = None,
    ):
        super().__init__(species, num_threads)
        self.per_system = bool(per_system)

    @property
    def feature_count(self) -> int:
        return len(self.species or ())

    def compute(self, value: StructureBatch | Sequence[Any] | Any, control: Any = None) -> DescriptorResult:
        batch = _as_batch(value)
        species = self._species_for(batch)
        self.species = species
        values = _cpp.compute_atomic_composition(
            batch.numbers, batch.positions, batch.cells, batch.pbc, batch.offsets,
            list(species), self.per_system, self.num_threads, control,
        )
        self._feature_count = int(values.shape[1])
        return _atom_result(values, batch, self.name, species, level="structure" if self.per_system else "atom", offsets=None if self.per_system else batch.offsets.copy())


class SortedDistancesKernel(_AtomKernel):
    name = "SortedDistances"

    def __init__(self, species: Iterable[int] | None = None, cutoff: float = 6.0, max_neighbors: int = 8, separate_neighbor_types: bool = True, num_threads: int | None = None):
        super().__init__(species, num_threads)
        self.cutoff, self.max_neighbors = float(cutoff), int(max_neighbors)
        self.separate_neighbor_types = bool(separate_neighbor_types)
        if self.cutoff <= 0.0 or self.max_neighbors <= 0:
            raise ValueError("cutoff and max_neighbors must be positive")

    def compute(self, value: StructureBatch | Sequence[Any] | Any, control: Any = None) -> DescriptorResult:
        batch = _as_batch(value)
        species = self._species_for(batch)
        values = _cpp.compute_sorted_distances(batch.numbers, batch.positions, batch.cells, batch.pbc, batch.offsets, list(species), self.cutoff, self.max_neighbors, self.separate_neighbor_types, self.num_threads, control)
        self._feature_count = int(values.shape[1])
        return _atom_result(values, batch, self.name, species, offsets=batch.offsets.copy())


class NeighborListKernel:
    name = "NeighborList"
    _feature_labels = ("dx", "dy", "dz", "distance")

    def __init__(
        self,
        cutoff: float = 6.0,
        full_neighbor_list: bool = True,
        self_pairs: bool = False,
        num_threads: int | None = None,
    ):
        self.cutoff, self.full_neighbor_list, self.self_pairs = float(cutoff), bool(full_neighbor_list), bool(self_pairs)
        self.num_threads = _threads(num_threads)
        if self.cutoff <= 0.0:
            raise ValueError("cutoff must be positive")

    @property
    def feature_count(self) -> int:
        return len(self._feature_labels)

    def _raw(self, batch: StructureBatch, control: Any = None) -> tuple[np.ndarray, np.ndarray]:
        values, offsets = _cpp.compute_neighbor_list(
            batch.numbers, batch.positions, batch.cells, batch.pbc, batch.offsets,
            self.cutoff, self.full_neighbor_list, self.self_pairs, self.num_threads, control,
        )
        return np.asarray(values, dtype=np.float64), np.asarray(offsets, dtype=np.int64)

    def compute(self, value: StructureBatch | Sequence[Any] | Any, control: Any = None) -> DescriptorResult:
        batch = _as_batch(value)
        values, offsets = self._raw(batch, control)
        records = values[:, :5]
        return DescriptorResult(
            values[:, 5:],
            "pair",
            batch.ids,
            offsets,
            self._feature_labels,
            _cpp_metadata(self.name),
            samples=pair_samples(records, offsets, batch.offsets),
            _atom_row_offsets=batch.offsets.copy(),
        )


class SphericalExpansionKernel(_AtomKernel):
    name = "SphericalExpansion"
    _kind = 0
    k_cutoff = 2.5
    exponent = 1
    radial_radius: float | None = None

    def __init__(self, species: Iterable[int] | None = None, cutoff: float = 6.0, density_width: float = 0.3, max_radial: int = 6, max_angular: int = 4, num_threads: int | None = None):
        super().__init__(species, num_threads)
        self.cutoff, self.density_width = float(cutoff), float(density_width)
        self.max_radial, self.max_angular = int(max_radial), int(max_angular)
        if self.cutoff <= 0.0 or self.density_width <= 0.0 or self.max_radial < 0 or self.max_angular < 0:
            raise ValueError("invalid spherical expansion parameters")

    def _compute_native(
        self,
        batch: StructureBatch,
        species: tuple[int, ...],
        control: Any,
    ) -> Any:
        radial_radius = self.cutoff if self.radial_radius is None else self.radial_radius
        return _cpp.compute_spherical_expansion(
            batch.numbers,
            batch.positions,
            batch.cells,
            batch.pbc,
            batch.offsets,
            list(species),
            self.cutoff,
            self.density_width,
            self.max_radial,
            self.max_angular,
            self._kind,
            self.k_cutoff,
            self.exponent,
            radial_radius,
            self.num_threads,
            control,
        )

    def _result_from_native(
        self,
        raw: Any,
        batch: StructureBatch,
        species: tuple[int, ...],
    ) -> DescriptorResult:
        values = np.asarray(raw, dtype=np.float64)
        self._feature_count = int(values.shape[1])
        return _atom_result(values, batch, self.name, species, offsets=batch.offsets.copy())

    def compute(self, value: StructureBatch | Sequence[Any] | Any, control: Any = None) -> DescriptorResult:
        batch = _as_batch(value)
        species = self._species_for(batch)
        raw = self._compute_native(batch, species, control)
        return self._result_from_native(raw, batch, species)


class SphericalExpansionByPairKernel(SphericalExpansionKernel):
    name = "SphericalExpansionByPair"
    _kind = 1

    def _compute_native(
        self,
        batch: StructureBatch,
        species: tuple[int, ...],
        control: Any,
    ) -> Any:
        return _cpp.compute_spherical_expansion_by_pair(
            batch.numbers,
            batch.positions,
            batch.cells,
            batch.pbc,
            batch.offsets,
            list(species),
            self.cutoff,
            self.density_width,
            self.max_radial,
            self.max_angular,
            self.num_threads,
            control,
        )

    def _result_from_native(
        self,
        raw: Any,
        batch: StructureBatch,
        species: tuple[int, ...],
    ) -> DescriptorResult:
        values, offsets, identifiers = raw
        values = np.asarray(values, dtype=np.float64)
        offsets = np.asarray(offsets, dtype=np.int64)
        self._feature_count = int(values.shape[1])
        return DescriptorResult(
            values,
            "pair",
            batch.ids,
            offsets,
            tuple(f"{self.name}:{i}" for i in range(values.shape[1])),
            _cpp_metadata(self.name, species=species),
            samples=pair_samples(identifiers[:, :5], offsets, batch.offsets),
            _atom_row_offsets=batch.offsets.copy(),
        )


class SoapRadialSpectrumKernel(SphericalExpansionKernel):
    name = "SoapRadialSpectrum"
    _kind = 2


class SoapPowerSpectrumKernel(SphericalExpansionKernel):
    name = "SoapPowerSpectrum"
    _kind = 3


class LodeSphericalExpansionKernel(SphericalExpansionKernel):
    name = "LodeSphericalExpansion"
    _kind = 4

    def __init__(
        self,
        species: Iterable[int] | None = None,
        cutoff: float = 6.0,
        density_width: float = 0.3,
        max_radial: int = 6,
        max_angular: int = 4,
        num_threads: int | None = None,
        k_cutoff: float = 2.5,
        exponent: int = 1,
        radial_radius: float | None = None,
    ):
        super().__init__(
            species=species,
            cutoff=cutoff,
            density_width=density_width,
            max_radial=max_radial,
            max_angular=max_angular,
            num_threads=num_threads,
        )
        self.k_cutoff, self.exponent = float(k_cutoff), int(exponent)
        self.radial_radius = self.cutoff if radial_radius is None else float(radial_radius)
        if self.exponent < 1 or self.exponent > 9:
            raise ValueError("LODE exponent must be between 1 and 9")


__all__ = ["AtomicCompositionKernel", "NeighborListKernel", "SortedDistancesKernel", "SphericalExpansionKernel", "SphericalExpansionByPairKernel", "SoapRadialSpectrumKernel", "SoapPowerSpectrumKernel", "LodeSphericalExpansionKernel"]
