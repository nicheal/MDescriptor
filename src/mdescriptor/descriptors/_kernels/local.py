"""C++-backed local descriptor adapters.

The names mirror the supported local descriptor families, but no external package is
imported or required. Every value path enters ``_native`` directly.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np

from ...core.result import pair_samples
from ...core.species import require_species, validate_batch_species
from .core import DescriptorResult, StructureBatch, _as_batch, _cpp


class _AtomKernel:
    name = "descriptor"

    def __init__(self, species: Iterable[int] | None = None, num_threads: int | None = None):
        self.species = require_species(species, descriptor=self.__class__.name)
        self.num_threads = 0 if num_threads is None else int(num_threads)
        if self.num_threads < 0:
            raise ValueError("num_threads must be non-negative")

    def _species_for(self, batch: StructureBatch) -> tuple[int, ...]:
        return validate_batch_species(batch, self.species, descriptor=self.name)

    @property
    def feature_count(self) -> int:
        return int(getattr(self, "_feature_count", 0))


def _atom_result(values: np.ndarray, batch: StructureBatch, name: str, species: tuple[int, ...], *, level: str = "atom", offsets: np.ndarray | None = None, metadata: dict[str, Any] | None = None) -> DescriptorResult:
    values = np.asarray(values, dtype=np.float64)
    details = {"backend": "mdescriptor-cpp", "descriptor": name, "species": species}
    if metadata:
        details.update(metadata)
    return DescriptorResult(values, level, batch.ids, offsets, tuple(f"{name}:{index}" for index in range(values.shape[1])), details)


class AtomicCompositionKernel:
    name = "AtomicComposition"

    def __init__(self, species: Iterable[int] | None = None, per_system: bool = True):
        self.species = require_species(species, descriptor=self.name)
        self.per_system = bool(per_system)

    @property
    def feature_count(self) -> int:
        return len(self.species or ())

    def compute(self, value: StructureBatch | Sequence[Any] | Any, control: Any = None) -> DescriptorResult:
        batch = _as_batch(value)
        species = validate_batch_species(batch, self.species, descriptor=self.name)
        self.species = species
        values = _cpp.compute_atomic_composition(batch.numbers, batch.positions, batch.cells, batch.pbc, batch.offsets, list(species), self.per_system, control)
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

    def __init__(self, cutoff: float = 6.0, full_neighbor_list: bool = True, self_pairs: bool = False):
        self.cutoff, self.full_neighbor_list, self.self_pairs = float(cutoff), bool(full_neighbor_list), bool(self_pairs)
        if self.cutoff <= 0.0:
            raise ValueError("cutoff must be positive")

    @property
    def feature_count(self) -> int:
        return 9

    def _raw(self, batch: StructureBatch, control: Any = None) -> tuple[np.ndarray, np.ndarray]:
        values, offsets = _cpp.compute_neighbor_list(batch.numbers, batch.positions, batch.cells, batch.pbc, batch.offsets, self.cutoff, self.full_neighbor_list, self.self_pairs, control)
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
            ("dx", "dy", "dz", "distance"),
            {"backend": "mdescriptor-cpp", "descriptor": self.name},
            samples=pair_samples(records, offsets, batch.offsets),
        )

    def pairs(self, value: StructureBatch | Sequence[Any] | Any) -> list[np.ndarray]:
        batch = _as_batch(value)
        values, offsets = self._raw(batch)
        return [values[int(offsets[index]):int(offsets[index + 1])] for index in range(batch.structures)]


class SphericalExpansionKernel(_AtomKernel):
    name = "SphericalExpansion"
    _kind = 0

    def __init__(self, species: Iterable[int] | None = None, cutoff: float = 6.0, density_width: float = 0.3, max_radial: int = 6, max_angular: int = 4, num_threads: int | None = None):
        super().__init__(species, num_threads)
        self.cutoff, self.density_width = float(cutoff), float(density_width)
        self.max_radial, self.max_angular = int(max_radial), int(max_angular)
        if self.cutoff <= 0.0 or self.density_width <= 0.0 or self.max_radial < 0 or self.max_angular < 0:
            raise ValueError("invalid spherical expansion parameters")

    def compute(self, value: StructureBatch | Sequence[Any] | Any, control: Any = None) -> DescriptorResult:
        batch = _as_batch(value)
        species = self._species_for(batch)
        if self.name == "SphericalExpansionByPair":
            values, offsets, identifiers = _cpp.compute_spherical_expansion_by_pair(batch.numbers, batch.positions, batch.cells, batch.pbc, batch.offsets, list(species), self.cutoff, self.density_width, self.max_radial, self.max_angular, self.num_threads, control)
            values = np.asarray(values, dtype=np.float64)
            self._feature_count = int(values.shape[1])
            return DescriptorResult(
                values,
                "pair",
                batch.ids,
                np.asarray(offsets, dtype=np.int64),
                tuple(f"{self.name}:{i}" for i in range(values.shape[1])),
                {"backend": "mdescriptor-cpp", "descriptor": self.name, "species": species},
                samples=pair_samples(identifiers[:, :5], offsets, batch.offsets),
            )
        values = _cpp.compute_spherical_expansion(batch.numbers, batch.positions, batch.cells, batch.pbc, batch.offsets, list(species), self.cutoff, self.density_width, self.max_radial, self.max_angular, self._kind, getattr(self, "k_cutoff", 2.5), getattr(self, "exponent", 1), getattr(self, "radial_radius", self.cutoff), self.num_threads, control)
        values = np.asarray(values, dtype=np.float64)
        self._feature_count = int(values.shape[1])
        return _atom_result(values, batch, self.name, species, offsets=batch.offsets.copy())


class SphericalExpansionByPairKernel(SphericalExpansionKernel):
    name = "SphericalExpansionByPair"
    _kind = 1


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


__all__ = ["AtomicCompositionKernel", "NeighborListKernel", "SortedDistancesKernel", "SphericalExpansionKernel", "SphericalExpansionByPairKernel", "SoapRadialSpectrumKernel", "SoapPowerSpectrumKernel", "LodeSphericalExpansionKernel"]
