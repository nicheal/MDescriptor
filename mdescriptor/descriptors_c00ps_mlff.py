"""C00PS-MLFF radial/angular descriptor backed by the native C++ extension."""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np

from .descriptors import (
    DescriptorResult,
    StructureBatch,
    _as_batch,
    _cpp,
    _format_output,
    _merge_config,
    _normalise_species,
    _species_from_batch,
)


_CUTOFF_FUNCTIONS = {"bp": 0, "mo": 1, "rj": 2, "wmc": 3}


class C00PSMlffCalculator:
    """Native C++ implementation of the C00 + PS MLFF descriptor."""

    name = "C00PSMLFF"

    def __init__(
        self,
        species: Iterable[int] | None = None,
        config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        config = _merge_config(config, kwargs)
        self.species = _normalise_species(species)
        self.r_cut = float(config.get("r_cut", config.get("cutoff", 6.0)))
        self.n_radial = int(config.get("n_radial", config.get("n_max", 8)))
        self.l_max = int(config.get("l_max", 4))
        self.cutoff = str(config.get("cutoff_function", "bp")).lower()
        self.include_radial = bool(config.get("include_radial", True))
        self.include_angular = bool(config.get("include_angular", True))
        self.normalize_radial = bool(config.get("normalize_radial", False))
        self.normalize_angular = bool(config.get("normalize_angular", False))
        self.super_vector = bool(config.get("super_vector", False))
        self.radial_weight = float(config.get("radial_weight", 1.0))
        self.angular_weight = float(config.get("angular_weight", 1.0))
        self.exclude_self_interaction = bool(config.get("exclude_self_interaction", True))
        self.num_threads = int(config.get("num_threads", 0))
        self.dtype = str(config.get("dtype", "float64"))
        self.sparse = bool(config.get("sparse", False))
        if self.r_cut <= 0.0 or self.n_radial <= 0 or self.l_max < 0:
            raise ValueError("C00PSMLFF requires r_cut > 0, n_radial > 0, and l_max >= 0")
        if self.cutoff not in _CUTOFF_FUNCTIONS:
            raise ValueError("cutoff_function must be 'bp', 'mo', 'rj', or 'wmc'")
        if self.radial_weight < 0.0 or self.angular_weight < 0.0:
            raise ValueError("radial_weight and angular_weight must be non-negative")
        if self.dtype not in {"float32", "float64"}:
            raise ValueError("dtype must be 'float32' or 'float64'")
        if not self.include_radial and not self.include_angular:
            raise ValueError("at least one of include_radial/include_angular must be true")
        self._native = self._make_native() if self.species is not None else None

    def _make_native(self) -> Any:
        assert self.species is not None
        options = _cpp.C00PSMlffOptions()
        options.species = list(self.species)
        options.r_cut = self.r_cut
        options.n_radial = self.n_radial
        options.l_max = self.l_max
        options.cutoff_function = _CUTOFF_FUNCTIONS[self.cutoff]
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
        if self.species is None:
            self.species = _species_from_batch(batch)
            self._native = self._make_native()
        missing = set(np.unique(batch.numbers)) - set(self.species)
        if missing:
            raise ValueError(f"batch contains species not fixed in calculator: {sorted(missing)}")
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
        values = _format_output(values, self.dtype, self.sparse)
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

    def _labels(self) -> tuple[str, ...]:
        if not self.species or self._native is None:
            return ()
        labels = []
        radial_counts = [int(count) for count in self._native.radial_counts]
        radial_channels = [(z, n) for z in self.species for n in range(radial_counts[0])]
        if self.include_radial:
            labels.extend(f"c00ps_mlff:c00,z={z},n={n}" for z, n in radial_channels)
        if self.include_angular:
            for l in range(self.l_max + 1):
                channels = [(z, n) for z in self.species for n in range(radial_counts[l])]
                for first, left in enumerate(channels):
                    for right in channels[first:]:
                        labels.append(
                            f"c00ps_mlff:ps,z1={left[0]},n1={left[1]},z2={right[0]},n2={right[1]},l={l}"
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


C00PSMLFF = C00PSMlffCalculator

__all__ = ["C00PSMLFF", "C00PSMlffCalculator"]
