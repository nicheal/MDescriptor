"""Native MTP moment-tensor descriptor adapter."""

from __future__ import annotations

import threading
from collections.abc import Iterable
from typing import Any

import numpy as np

from ...core.errors import ModelLoadError
from ...core.species import require_species, validate_batch_species
from ._base import _cpp_metadata, _Kernel, _optional_threads, _validate_dtype
from .core import (
    StructureBatch,
    _cpp,
)


class MtpKernel(_Kernel):
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
        model_digest: str | None = None,
        model_data: bytes | None = None,
        min_dist: float = 0.0,
        max_dist: float | None = None,
        radial_basis_size: int = 4,
        radial_funcs_count: int = 1,
        max_rank: int | None = None,
        radial_basis_type: str = "RBChebyshev",
        dtype: str = "float64",
        sparse: bool = False,
        num_threads: int | None = None,
    ) -> None:
        self.species = require_species(species, descriptor=self.name)
        self.model_path = None if model_path is None else str(model_path)
        self.model_digest = None if model_digest is None else str(model_digest)
        self.model_data = None if model_data is None else bytes(model_data)
        if self.model_path == "":
            raise ValueError("model must not be empty")
        self._official = self.model_path is not None
        self.min_dist = float(min_dist)
        self.max_dist = float(max_dist if max_dist is not None else 5.0)
        self.radial_basis_size = int(radial_basis_size)
        self.radial_funcs_count = int(radial_funcs_count)
        self.max_rank = int(2 if max_rank is None else max_rank)
        self.radial_basis_type = str(radial_basis_type)
        self.dtype = _validate_dtype(dtype)
        self.sparse = bool(sparse)
        self.num_threads = _optional_threads(num_threads)
        if not self._official and self.radial_basis_type not in {"RBChebyshev", "Chebyshev", "polynomial"}:
            raise ValueError("unsupported MTP radial_basis_type")
        if not self._official and (self.min_dist < 0.0 or self.max_dist <= self.min_dist):
            raise ValueError("MTP requires 0 <= min_dist < max_dist")
        if not self._official and (self.radial_basis_size <= 0 or self.radial_funcs_count <= 0):
            raise ValueError("MTP radial basis sizes must be positive")
        if not self._official and (self.max_rank < 0 or self.max_rank > 5):
            raise ValueError("MTP max_rank must be between 0 and 5")
        self._native: Any = None
        self._closed = False
        self._feature_count = 0
        self._init_lock = threading.Lock()
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
        with self._init_lock:
            if self._native is not None:
                return
            options = _cpp.MtpOptions()
            options.species = list(self.species)
            options.potential_path = self.model_path or ""
            if self.model_digest is not None:
                options.model_digest = self.model_digest
            if self.model_data is not None:
                try:
                    options.model_data = bytes(self.model_data)
                except AttributeError as exc:
                    raise ModelLoadError(
                        "native extension lacks immutable model snapshot support; rebuild MDescriptor"
                    ) from exc
            options.min_dist = self.min_dist
            options.max_dist = self.max_dist
            options.radial_basis_size = self.radial_basis_size
            options.radial_funcs_count = self.radial_funcs_count
            options.max_rank = self.max_rank
            options.num_threads = 0 if self.num_threads is None else int(self.num_threads)
            native = _cpp.MtpCalculator(options)
            self._native = native
            self._feature_count = int(native.feature_count)

    def _cuda_payload(self) -> dict[str, Any]:
        """Return the flattened official MLIP-4 evaluator for CUDA."""

        if not self._official or self._native is None:
            return {"feature_count": self.feature_count}
        if not bool(self._native.official_mlip4):
            return {
                "feature_count": self.feature_count,
                "format": "MLIP-2",
                "species_count": int(self._native.official_species_count),
                "scaling": float(self._native.official_scaling),
                "radial_basis_type": str(self._native.official_radial_basis_type),
                "radial_coefficients": np.asarray(self._native.official_radial_coefficients, dtype=np.float64),
                "alpha_index_basic": np.asarray(self._native.official_alpha_index_basic, dtype=np.int32),
                "alpha_index_times": np.asarray(self._native.official_alpha_index_times, dtype=np.int32),
                "alpha_moment_mapping": np.asarray(self._native.official_alpha_moment_mapping, dtype=np.int32),
                "alpha_moments_count": int(self._native.official_alpha_moments_count),
                "radial_basis_size": int(self._native.official_radial_basis_size),
                "radial_funcs_count": int(self._native.official_radial_funcs_count),
                "radial_min_dist": float(self._native.official_min_dist),
                "radial_max_dist": float(self._native.official_max_dist),
            }
        return {
            "feature_count": self.feature_count,
            "model_species": np.asarray(self._native.official_model_species, dtype=np.int32),
            "model_parameters": np.asarray(self._native.official_model_parameters, dtype=np.float64),
            "radial_scaling": float(self._native.official_radial_scaling),
            "radial_kind": int(self._native.official_radial_kind),
            "radial_basis_size": int(self._native.official_radial_basis_size),
            "radial_funcs_count": int(self._native.official_radial_funcs_count),
            "radial_min_dist": float(self._native.official_min_dist),
            "radial_max_dist": float(self._native.official_max_dist),
            "radial_recursive": np.asarray(self._native.official_radial_recursive, dtype=np.float64),
            "radial_zeroth": float(self._native.official_radial_zeroth),
            "radial_exp_ratio": float(self._native.official_radial_exp_ratio),
            "radial_maxdist_sq": float(self._native.official_radial_maxdist_sq),
            "radial_maxdist_sq_minus_eps": float(self._native.official_radial_maxdist_sq_minus_eps),
            "radial_vdw_params": np.asarray(self._native.official_radial_vdw_params, dtype=np.float64),
            "moments": np.asarray(self._native.official_moments, dtype=np.int32),
            "eval_kinds": np.asarray(self._native.official_eval_kinds, dtype=np.int32),
            "eval_linear_ids": np.asarray(self._native.official_eval_linear_ids, dtype=np.int32),
            "eval_linear_coefficients": np.asarray(self._native.official_eval_linear_coefficients, dtype=np.float64),
            "eval_product_offsets": np.asarray(self._native.official_eval_product_offsets, dtype=np.int64),
            "eval_product_left": np.asarray(self._native.official_eval_product_left, dtype=np.int32),
            "eval_product_right": np.asarray(self._native.official_eval_product_right, dtype=np.int32),
            "eval_product_coefficients": np.asarray(self._native.official_eval_product_coefficients, dtype=np.float64),
            "scalar_output_ids": np.asarray(self._native.official_scalar_output_ids, dtype=np.int32),
        }

    def _ensure_native(self, batch: StructureBatch) -> None:
        if self._closed:
            raise RuntimeError("MTP calculator is closed")
        self.species = validate_batch_species(batch, self.species, descriptor=self.name)
        self._create_native()

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
        return _cpp_metadata(
            self.name,
            species=self.species,
            model_path=self.model_path,
            official_model=self._official,
            official_format=official_format,
            official_mlip4=official_mlip4,
            feature_count=self.feature_count,
            min_dist=min_dist,
            max_dist=max_dist,
            radial_basis_type=radial_basis_type,
            radial_basis_size=radial_basis_size,
            radial_funcs_count=radial_funcs_count,
            max_rank=self.max_rank,
            dtype=self.dtype,
            sparse=self.sparse,
        )


__all__ = ["MtpKernel"]
