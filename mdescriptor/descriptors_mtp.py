"""Native MTP moment-tensor descriptor adapter."""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np

from .descriptors import (
    DescriptorResult,
    StructureBatch,
    _as_batch,
    _merge_config,
    _normalise_species,
    _format_output,
    _cpp,
)


class MtpCalculator:
    """Rotationally invariant moment-tensor basis for periodic structures.

    The radial channels use cutoff-squared Chebyshev functions.  Each channel
    contributes scalar traces and pairwise Frobenius contractions of the
    moment tensors, which is the compact invariant core used by MTP models.

    Passing ``potential_path`` (or the aliases ``potential``/``model``) loads
    either an MLIP-2 text ``.mtp`` potential or a native MLIP-4 JSON MTP.  The
    MLIP-2 path exposes the official constant/alpha-moment columns.  The
    MLIP-4 path exposes the native scalar ``mtp_basis`` outputs in their JSON
    basis order.
    """

    name = "MTP"

    def __init__(
        self,
        species: Iterable[int] | None = None,
        config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        config = _merge_config(config, kwargs)
        self.species = _normalise_species(species)
        potential_path = config.get("potential_path", config.get("potential", config.get("model")))
        self.potential_path = None if potential_path is None else str(potential_path)
        if self.potential_path == "":
            raise ValueError("potential_path must not be empty")
        self._official = self.potential_path is not None
        self.min_dist = float(config.get("min_dist", 0.0))
        self.max_dist = float(config.get("max_dist", config.get("r_cut", config.get("cutoff", 5.0))))
        self.radial_basis_size = int(config.get("radial_basis_size", 4))
        self.radial_funcs_count = int(config.get("radial_funcs_count", 1))
        rank = config.get("max_rank", config.get("l_max"))
        if rank is None:
            level = config.get("max_level", config.get("level"))
            rank = 2 if level is None else min(5, max(0, int(level) - 2))
        self.max_rank = int(rank)
        self.radial_basis_type = str(config.get("radial_basis_type", "RBChebyshev"))
        self.dtype = str(config.get("dtype", "float64"))
        self.sparse = bool(config.get("sparse", False))
        self.num_threads = config.get("num_threads")
        if self.dtype not in {"float32", "float64"}:
            raise ValueError("dtype must be 'float32' or 'float64'")
        if not self._official and self.radial_basis_type not in {"RBChebyshev", "Chebyshev", "polynomial"}:
            raise ValueError("unsupported MTP radial_basis_type")
        if not self._official and (self.min_dist < 0.0 or self.max_dist <= self.min_dist):
            raise ValueError("MTP requires 0 <= min_dist < max_dist")
        if not self._official and (self.radial_basis_size <= 0 or self.radial_funcs_count <= 0):
            raise ValueError("MTP radial basis sizes must be positive")
        if not self._official and (self.max_rank < 0 or self.max_rank > 5):
            raise ValueError("MTP max_rank must be between 0 and 5")
        if self.num_threads is not None and int(self.num_threads) <= 0:
            raise ValueError("num_threads must be a positive integer or None")
        self._native: Any = None
        self._closed = False
        self._feature_count = 0
        if self._official and self.species is None and self.potential_path and self.potential_path.lower().endswith(".json"):
            # Native MLIP-4 stores the species order in PairDescriptorPot.
            # This keeps the JSON potential self-describing while preserving
            # the explicit-species behavior for MLIP-2 text potentials.
            import json

            with open(self.potential_path, "r", encoding="utf-8") as handle:
                potential = json.load(handle)
            if isinstance(potential, list) and len(potential) == 2:
                potential = potential[1]
            pair = potential.get("PairDescriptorPot", potential.get("pair_descriptor_pot", {}))
            if pair.get("species_order") is not None and all(int(value) > 0 for value in pair["species_order"]):
                self.species = _normalise_species(pair["species_order"])
        if self._official and self.species is not None:
            self._create_native()

    @property
    def feature_count(self) -> int:
        if self._official:
            return self._feature_count
        if not self.species:
            return 0
        channels = len(self.species) * self.radial_funcs_count * self.radial_basis_size
        # ponytail: keep the basis at rank traces + pair contractions; full MLIP alpha_index recursion is the upgrade path.
        return channels * (self.max_rank // 2 + 1) + (self.max_rank + 1) * channels * (channels + 1) // 2

    def _create_native(self) -> None:
        if self._native is not None:
            return
        if self.species is None:
            raise RuntimeError("MTP species must be inferred from a batch before creating the native calculator")
        options = _cpp.MtpOptions()
        options.species = list(self.species)
        options.potential_path = self.potential_path or ""
        options.min_dist = self.min_dist
        options.max_dist = self.max_dist
        options.radial_basis_size = self.radial_basis_size
        options.radial_funcs_count = self.radial_funcs_count
        options.max_rank = self.max_rank
        options.num_threads = 0 if self.num_threads is None else int(self.num_threads)
        self._native = _cpp.MtpCalculator(options)
        self._feature_count = int(self._native.feature_count)

    def _ensure_native(self, batch: StructureBatch) -> None:
        if self._closed:
            raise RuntimeError("MTP calculator is closed")
        if self.species is None:
            self.species = _normalise_species(np.unique(batch.numbers))
        missing = set(np.unique(batch.numbers)) - set(self.species)
        if missing:
            raise ValueError(f"batch contains species not fixed in calculator: {sorted(missing)}")
        self._create_native()

    def compute(self, value: StructureBatch | Sequence[Any] | Any, control: Any = None) -> DescriptorResult:
        batch = _as_batch(value)
        self._ensure_native(batch)
        values = self._native.compute(
            batch.numbers, batch.positions, batch.cells, batch.pbc, batch.offsets, control
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
        if self._native is not None:
            self._native.close()

    def _labels(self) -> tuple[str, ...]:
        if not self.species:
            return ()
        if self._official:
            if self._native is None:
                return ()
            if bool(getattr(self._native, "official_mlip4", False)):
                return tuple(f"mlip4:basis={index}" for index in range(int(self._native.feature_count)))
            mapping = tuple(int(index) for index in self._native.official_alpha_moment_mapping)
            return ("mlip2:constant",) + tuple(f"mlip2:moment={index}" for index in mapping)
        labels = []
        channels = [
            (species, radial_function, radial)
            for species in self.species
            for radial_function in range(self.radial_funcs_count)
            for radial in range(self.radial_basis_size)
        ]
        for species, radial_function, radial in channels:
            for rank in range(0, self.max_rank + 1, 2):
                labels.append(
                    f"mtp:trace=z{species},rf={radial_function},n={radial},rank={rank}"
                )
        for rank in range(self.max_rank + 1):
            for first, left in enumerate(channels):
                for right in channels[first:]:
                    labels.append(
                        "mtp:contraction=dot,"
                        f"rank={rank},z1={left[0]},rf1={left[1]},n1={left[2]},"
                        f"z2={right[0]},rf2={right[1]},n2={right[2]}"
                    )
        return tuple(labels)

    def _metadata(self) -> dict[str, Any]:
        min_dist = self.min_dist
        max_dist = self.max_dist
        radial_basis_size = self.radial_basis_size
        radial_funcs_count = self.radial_funcs_count
        radial_basis_type = self.radial_basis_type
        if self._official and self._native is not None:
            min_dist = float(self._native.official_min_dist)
            max_dist = float(self._native.official_max_dist)
            radial_basis_size = int(self._native.official_radial_basis_size)
            radial_funcs_count = int(self._native.official_radial_funcs_count)
            radial_basis_type = str(self._native.official_radial_basis_type)
        official_format = None
        official_mlip4 = False
        if self._official and self._native is not None:
            official_format = str(getattr(self._native, "official_format", "MLIP-2"))
            official_mlip4 = bool(getattr(self._native, "official_mlip4", False))
        return {
            "backend": "mdescriptor-cpp",
            "descriptor": self.name,
            "species": self.species,
            "potential_path": self.potential_path,
            "official_model": self._official,
            "official_format": official_format,
            "official_mlip4": official_mlip4,
            "feature_count": self.feature_count,
            "min_dist": min_dist,
            "max_dist": max_dist,
            "radial_basis_type": radial_basis_type,
            "radial_basis_size": radial_basis_size,
            "radial_funcs_count": radial_funcs_count,
            "max_rank": self.max_rank,
            "dtype": self.dtype,
            "sparse": self.sparse,
        }


MTP = MtpCalculator

__all__ = ["MTP", "MtpCalculator"]
