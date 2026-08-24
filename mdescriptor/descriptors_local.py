"""C++-backed local descriptor adapters.

The names mirror the supported local descriptor families, but no external package is
imported or required. Every value path enters ``_descriptor_cpp`` directly.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np

from .descriptors import DescriptorResult, StructureBatch, _as_batch, _cpp


class _AtomCalculator:
    name = "descriptor"

    def __init__(self, species: Iterable[int] | None = None, num_threads: int | None = None):
        self.species = tuple(int(value) for value in species) if species is not None else None
        self.num_threads = 0 if num_threads is None else int(num_threads)
        if self.num_threads < 0:
            raise ValueError("num_threads must be non-negative")

    def _species_for(self, batch: StructureBatch) -> tuple[int, ...]:
        species = self.species or tuple(int(value) for value in np.unique(batch.numbers))
        if not species or any(value <= 0 for value in species) or len(set(species)) != len(species):
            raise ValueError("species must contain unique positive atomic numbers")
        missing = set(np.unique(batch.numbers)) - set(species)
        if missing:
            raise ValueError(f"batch contains species outside calculator species: {sorted(missing)}")
        self.species = species
        return species

    @property
    def feature_count(self) -> int:
        return int(getattr(self, "_feature_count", 0))

    def create(self, value: StructureBatch | Sequence[Any] | Any, control: Any = None) -> np.ndarray:
        return self.compute(value, control).values


def _atom_result(values: np.ndarray, batch: StructureBatch, name: str, species: tuple[int, ...], *, level: str = "atom", offsets: np.ndarray | None = None, metadata: dict[str, Any] | None = None) -> DescriptorResult:
    values = np.asarray(values, dtype=np.float64)
    details = {"backend": "mdescriptor-cpp", "descriptor": name, "species": species}
    if metadata:
        details.update(metadata)
    return DescriptorResult(values, level, batch.ids, offsets, tuple(f"{name}:{index}" for index in range(values.shape[1])), details)


class AtomicCompositionCalculator:
    name = "AtomicComposition"

    def __init__(self, species: Iterable[int] | None = None, per_system: bool = True):
        self.species = tuple(int(value) for value in species) if species is not None else None
        self.per_system = bool(per_system)

    @property
    def feature_count(self) -> int:
        return len(self.species or ())

    def compute(self, value: StructureBatch | Sequence[Any] | Any, control: Any = None) -> DescriptorResult:
        batch = _as_batch(value)
        species = self.species or tuple(int(item) for item in np.unique(batch.numbers))
        self.species = species
        values = _cpp.compute_atomic_composition(batch.numbers, batch.positions, batch.cells, batch.pbc, batch.offsets, list(species), self.per_system, control)
        self._feature_count = int(values.shape[1])
        return _atom_result(values, batch, self.name, species, level="structure" if self.per_system else "atom", offsets=None if self.per_system else batch.offsets.copy())

    def create(self, value: StructureBatch | Sequence[Any] | Any, control: Any = None) -> np.ndarray:
        return self.compute(value, control).values


class SortedDistancesCalculator(_AtomCalculator):
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


class NeighborListCalculator:
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
        return DescriptorResult(values, "pair", batch.ids, offsets, ("first", "second", "cell_shift_a", "cell_shift_b", "cell_shift_c", "dx", "dy", "dz", "distance"), {"backend": "mdescriptor-cpp", "descriptor": self.name})

    def create(self, value: StructureBatch | Sequence[Any] | Any, control: Any = None) -> np.ndarray:
        return self.compute(value, control).values

    def pairs(self, value: StructureBatch | Sequence[Any] | Any) -> list[np.ndarray]:
        batch = _as_batch(value)
        values, offsets = self._raw(batch)
        return [values[int(offsets[index]):int(offsets[index + 1])] for index in range(batch.structures)]


class SphericalExpansionCalculator(_AtomCalculator):
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
            return DescriptorResult(values, "pair", batch.ids, np.asarray(offsets, dtype=np.int64), tuple(f"{self.name}:{i}" for i in range(values.shape[1])), {"backend": "mdescriptor-cpp", "descriptor": self.name, "species": species, "pair_records": np.asarray(identifiers, dtype=np.float64)})
        values = _cpp.compute_spherical_expansion(batch.numbers, batch.positions, batch.cells, batch.pbc, batch.offsets, list(species), self.cutoff, self.density_width, self.max_radial, self.max_angular, self._kind, getattr(self, "k_cutoff", 2.5), getattr(self, "exponent", 1), getattr(self, "radial_radius", self.cutoff), self.num_threads, control)
        values = np.asarray(values, dtype=np.float64)
        self._feature_count = int(values.shape[1])
        return _atom_result(values, batch, self.name, species, offsets=batch.offsets.copy())


class SphericalExpansionByPairCalculator(SphericalExpansionCalculator):
    name = "SphericalExpansionByPair"
    _kind = 1


class SoapRadialSpectrumCalculator(SphericalExpansionCalculator):
    name = "SoapRadialSpectrum"
    _kind = 2


class SoapPowerSpectrumCalculator(SphericalExpansionCalculator):
    name = "SoapPowerSpectrum"
    _kind = 3


class LodeSphericalExpansionCalculator(SphericalExpansionCalculator):
    name = "LodeSphericalExpansion"
    _kind = 4

    def __init__(self, *args: Any, k_cutoff: float = 2.5, exponent: int = 1, radial_radius: float | None = None, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.k_cutoff, self.exponent = float(k_cutoff), int(exponent)
        self.radial_radius = self.cutoff if radial_radius is None else float(radial_radius)


__all__ = ["AtomicCompositionCalculator", "NeighborListCalculator", "SortedDistancesCalculator", "SphericalExpansionCalculator", "SphericalExpansionByPairCalculator", "SoapRadialSpectrumCalculator", "SoapPowerSpectrumCalculator", "LodeSphericalExpansionCalculator"]
