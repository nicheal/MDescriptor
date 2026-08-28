"""C00PS-MLFF radial/angular descriptor backed by the native C++ extension."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from ...core.result import format_values
from ...core.species import require_species, validate_batch_species
from .core import (
    DescriptorResult,
    StructureBatch,
    _as_batch,
    _cpp,
)

_CUTOFF_FUNCTIONS = {"bp": 0, "mo": 1, "rj": 2, "wmc": 3}


class C00PSMlffKernel:
    """Native C++ implementation of the C00 + PS MLFF descriptor."""

    name = "C00PSMLFF"

    def __init__(
        self,
        species: Iterable[int] | None = None,
        r_cut: float | None = None,
        cutoff: float | None = None,
        n_radial: int | None = None,
        n_max: int | None = None,
        l_max: int = 4,
        cutoff_function: str = "bp",
        radial_sigma: float = 0.5,
        include_radial: bool = True,
        include_angular: bool = True,
        normalize_radial: bool = False,
        normalize_angular: bool = False,
        super_vector: bool = False,
        radial_weight: float = 1.0,
        angular_weight: float = 1.0,
        exclude_self_interaction: bool = True,
        num_threads: int = 0,
        dtype: str = "float64",
        sparse: bool = False,
    ) -> None:
        self.species = require_species(species, descriptor=self.name)
        self.r_cut = float(r_cut if r_cut is not None else (cutoff if cutoff is not None else 6.0))
        self.n_radial = int(n_radial if n_radial is not None else (n_max if n_max is not None else 8))
        self.l_max = int(l_max)
        self.cutoff = str(cutoff_function).lower()
        self.radial_sigma = float(radial_sigma)
        self.include_radial = bool(include_radial)
        self.include_angular = bool(include_angular)
        self.normalize_radial = bool(normalize_radial)
        self.normalize_angular = bool(normalize_angular)
        self.super_vector = bool(super_vector)
        self.radial_weight = float(radial_weight)
        self.angular_weight = float(angular_weight)
        self.exclude_self_interaction = bool(exclude_self_interaction)
        self.num_threads = int(num_threads)
        self.dtype = str(dtype)
        self.sparse = bool(sparse)
        if self.r_cut <= 0.0 or self.n_radial <= 0 or self.l_max < 0:
            raise ValueError("C00PSMLFF requires r_cut > 0, n_radial > 0, and l_max >= 0")
        if self.cutoff not in _CUTOFF_FUNCTIONS:
            raise ValueError("cutoff_function must be 'bp', 'mo', 'rj', or 'wmc'")
        if self.radial_sigma < 0.0:
            raise ValueError("radial_sigma must be non-negative")
        if self.radial_weight < 0.0 or self.angular_weight < 0.0:
            raise ValueError("radial_weight and angular_weight must be non-negative")
        if self.dtype not in {"float32", "float64"}:
            raise ValueError("dtype must be 'float32' or 'float64'")
        if not self.include_radial and not self.include_angular:
            raise ValueError("at least one of include_radial/include_angular must be true")
        self._native = self._make_native()

    def _make_native(self) -> Any:
        assert self.species is not None
        options = _cpp.C00PSMlffOptions()
        options.species = list(self.species)
        options.r_cut = self.r_cut
        options.n_radial = self.n_radial
        options.l_max = self.l_max
        options.cutoff_function = _CUTOFF_FUNCTIONS[self.cutoff]
        options.radial_sigma = self.radial_sigma
        options.include_radial = self.include_radial
        options.include_angular = self.include_angular
        options.normalize_radial = self.normalize_radial
        options.normalize_angular = self.normalize_angular
        options.super_vector = self.super_vector
        options.radial_weight = self.radial_weight
        options.angular_weight = self.angular_weight
        options.exclude_self_interaction = self.exclude_self_interaction
        options.num_threads = self.num_threads
        return _cpp.C00PSMlffCalculator(options)

    @property
    def feature_count(self) -> int:
        if not self.species or self._native is None:
            return 0
        return int(self._native.feature_count)

    def compute(self, value: StructureBatch | Sequence[Any] | Any, control: Any = None) -> DescriptorResult:
        batch = _as_batch(value)
        self.species = validate_batch_species(batch, self.species, descriptor=self.name)
        assert self._native is not None
        if control is not None and bool(getattr(control, "cancelled", lambda: False)()):
            raise _cpp.CancelledError()
        values = self._native.compute(
            batch.numbers,
            batch.positions,
            batch.cells,
            batch.pbc,
            batch.offsets,
            control,
        )
        values = format_values(values, dtype=self.dtype, sparse=self.sparse)
        return DescriptorResult(
            values,
            "atom",
            batch.ids,
            batch.offsets.copy(),
            self._labels(),
            self._metadata(),
        )

    def _labels(self) -> tuple[str, ...]:
        if not self.species or self._native is None:
            return ()
        labels: list[str] = []
        radial_counts = [int(count) for count in self._native.radial_counts]
        radial_channels = [(z, n) for z in self.species for n in range(radial_counts[0])]
        if self.include_radial:
            labels.extend(f"c00ps_mlff:c00,z={z},n={n}" for z, n in radial_channels)
        if self.include_angular:
            for degree in range(self.l_max + 1):
                channels = [(z, n) for z in self.species for n in range(radial_counts[degree])]
                for first, left in enumerate(channels):
                    for right in channels[first:]:
                        labels.append(
                            f"c00ps_mlff:ps,z1={left[0]},n1={left[1]},z2={right[0]},n2={right[1]},l={degree}"
                        )
        return tuple(labels)

    def _metadata(self) -> dict[str, Any]:
        return {
            "backend": "mdescriptor-cpp",
            "descriptor": self.name,
            "source": "C00/PS radial-angular MLFF descriptor core",
            "species": self.species,
            "r_cut": self.r_cut,
            "n_radial": self.n_radial,
            "l_max": self.l_max,
            "cutoff_function": self.cutoff,
            "radial_sigma": self.radial_sigma,
            "exclude_self_interaction": self.exclude_self_interaction,
            "radial_weight": self.radial_weight,
            "angular_weight": self.angular_weight,
            "include_radial": self.include_radial,
            "include_angular": self.include_angular,
            "normalize_radial": self.normalize_radial,
            "normalize_angular": self.normalize_angular,
            "super_vector": self.super_vector,
            "num_threads": self.num_threads,
            "dtype": self.dtype,
            "sparse": self.sparse,
        }


__all__ = ["C00PSMlffKernel"]
