"""Native MTP moment-tensor descriptor adapter."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np

from ...core.result import format_values
from ...core.species import require_species, validate_batch_species
from .core import (
    DescriptorResult,
    StructureBatch,
    _as_batch,
    _cpp,
)


class MtpKernel:
    """Rotationally invariant moment-tensor basis for periodic structures.

    The radial channels use cutoff-squared Chebyshev functions.  Each channel
    contributes scalar traces and pairwise Frobenius contractions of the
    moment tensors, which is the compact invariant core used by MTP models.

    Passing ``model=`` loads either an MLIP-2 text ``.mtp`` potential or a
    native MLIP-4 JSON MTP.  The MLIP-2 path exposes the official
    constant/alpha-moment columns.  The MLIP-4 path exposes the native scalar
    ``mtp_basis`` outputs in their JSON basis order.  Omitting ``model=``
    selects the standalone moment-tensor basis.
    """

    name = "MTP"

    def __init__(
        self,
        species: Iterable[int] | None = None,
        model_path: str | None = None,
        min_dist: float = 0.0,
        max_dist: float | None = None,
        r_cut: float | None = None,
        cutoff: float | None = None,
        radial_basis_size: int = 4,
        radial_funcs_count: int = 1,
        max_rank: int | None = None,
        l_max: int | None = None,
        max_level: int | None = None,
        level: int | None = None,
        radial_basis_type: str = "RBChebyshev",
        dtype: str = "float64",
        sparse: bool = False,
        num_threads: int | None = None,
    ) -> None:
        self.species = require_species(species, descriptor=self.name)
        self.model_path = None if model_path is None else str(model_path)
        if self.model_path == "":
            raise ValueError("model must not be empty")
        self._official = self.model_path is not None
        self.min_dist = float(min_dist)
        max_dist = max_dist if max_dist is not None else (r_cut if r_cut is not None else (cutoff if cutoff is not None else 5.0))
        self.max_dist = float(max_dist)
        self.radial_basis_size = int(radial_basis_size)
        self.radial_funcs_count = int(radial_funcs_count)
        rank = max_rank if max_rank is not None else l_max
        if rank is None:
            rank = 2 if (max_level if max_level is not None else level) is None else min(5, max(0, int(max_level if max_level is not None else level) - 2))
        self.max_rank = int(rank)
        self.radial_basis_type = str(radial_basis_type)
        self.dtype = str(dtype)
        self.sparse = bool(sparse)
        self.num_threads = num_threads
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
        options = _cpp.MtpOptions()
        options.species = list(self.species)
        options.potential_path = self.model_path or ""
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
        self.species = validate_batch_species(batch, self.species, descriptor=self.name)
        self._create_native()

    def compute(self, value: StructureBatch | Sequence[Any] | Any, control: Any = None) -> DescriptorResult:
        batch = _as_batch(value)
        self._ensure_native(batch)
        values = self._native.compute(
            batch.numbers, batch.positions, batch.cells, batch.pbc, batch.offsets, control
        )
        values = format_values(np.asarray(values), dtype=self.dtype, sparse=self.sparse)
        return DescriptorResult(
            values,
            "atom",
            batch.ids,
            batch.offsets.copy(),
            self._labels(),
            self._metadata(),
        )

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
            "model_path": self.model_path,
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


__all__ = ["MtpKernel"]
