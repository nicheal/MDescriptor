"""Native core implementation of the soap_turbo-style power spectrum."""

from __future__ import annotations

from typing import Any

import numpy as np

from .descriptors import (
    DescriptorResult,
    StructureBatch,
    _as_batch,
    _cpp,
    _merge_config,
    _normalise_species,
    _species_from_batch,
    _format_output,
)


def _per_species(value: Any, count: int, name: str, default: float, *, integer: bool = False) -> list[Any]:
    if value is None:
        values = np.full(count, default, dtype=np.float64)
    else:
        array = np.asarray(value)
        if array.ndim == 0:
            values = np.full(count, array.item())
        else:
            values = array.ravel()
        if len(values) != count:
            raise ValueError(f"{name} must be a scalar or have one value per species")
    if integer:
        numeric = np.asarray(values, dtype=np.float64)
        if not np.isfinite(numeric).all() or not np.equal(numeric, np.floor(numeric)).all():
            raise ValueError(f"{name} must contain finite integers")
        return [int(item) for item in numeric]
    numeric = np.asarray(values, dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise ValueError(f"{name} must contain finite values")
    return [float(item) for item in numeric]


class SoapTurboCalculator:
    """Batch-first native core of the soap_turbo atomic power spectrum.

    Values follow the upstream SOAPTurbo radial/angular conventions and are
    normalized per atom. The upstream sparse compression recipes are exposed
    through ``compression``.
    """

    name = "SOAPTurbo"

    def __init__(self, species=None, config: dict[str, Any] | None = None, **kwargs: Any):
        config = _merge_config(config, kwargs)
        self.species = _normalise_species(species)
        self._alpha_max_config = config.get("alpha_max", 8)
        self.l_max = int(config.get("l_max", 6))
        self.rcut_hard = float(config.get("rcut_hard", 5.0))
        self.rcut_soft = float(config.get("rcut_soft", self.rcut_hard))
        self.nf = float(config.get("nf", 1.0))
        self.radial_enhancement = int(config.get("radial_enhancement", 0))
        self.basis = str(config.get("basis", "poly3")).lower()
        compression = config.get("compression", config.get("compress_mode", ""))
        self.compression = "" if compression is None else str(compression).lower()
        if self.compression in {"none", "off"}:
            self.compression = ""
        self.dtype = str(config.get("dtype", "float64"))
        self.sparse = bool(config.get("sparse", False))
        self.num_threads = config.get("num_threads")
        self._atom_sigma_r_config = config.get("atom_sigma_r", 0.5)
        self._atom_sigma_r_scaling_config = config.get("atom_sigma_r_scaling", 0.0)
        self._atom_sigma_t_config = config.get("atom_sigma_t", 0.5)
        self._atom_sigma_t_scaling_config = config.get("atom_sigma_t_scaling", 0.0)
        self._amplitude_scaling_config = config.get("amplitude_scaling", 0.0)
        self._central_weight_config = config.get("central_weight", 1.0)
        central_species = config.get("central_species")
        if central_species is None:
            self.central_species = None
        elif np.isscalar(central_species):
            self.central_species = _normalise_species([int(central_species)])
        else:
            self.central_species = _normalise_species(central_species)
        if self.dtype not in {"float32", "float64"}:
            raise ValueError("dtype must be 'float32' or 'float64'")
        if self.l_max < 0 or self.l_max > 20 or self.rcut_hard <= 0 or self.rcut_soft <= 0:
            raise ValueError("invalid SOAPTurbo cutoff or angular parameters")
        if self.rcut_soft > self.rcut_hard or self.nf <= 0 or self.radial_enhancement not in {0, 1, 2}:
            raise ValueError("invalid SOAPTurbo cutoff smoothing parameters")
        if self.basis not in {"poly3", "poly3gauss"}:
            raise ValueError("basis must be 'poly3' or 'poly3gauss'")
        if self.num_threads is not None and int(self.num_threads) <= 0:
            raise ValueError("num_threads must be a positive integer or None")
        self._alpha_max: list[int] | None = None
        self._atom_sigma_r: list[float] | None = None
        self._atom_sigma_r_scaling: list[float] | None = None
        self._atom_sigma_t: list[float] | None = None
        self._atom_sigma_t_scaling: list[float] | None = None
        self._amplitude_scaling: list[float] | None = None
        self._central_weight: list[float] | None = None
        self._native: Any = None
        self._closed = False

    def _ensure_native(self, batch: StructureBatch) -> None:
        if self._closed:
            raise RuntimeError("SOAPTurbo calculator is closed")
        if self.species is None:
            self.species = _species_from_batch(batch)
        missing = set(np.unique(batch.numbers)) - set(self.species)
        if missing:
            raise ValueError(f"batch contains species not fixed in calculator: {sorted(missing)}")
        if self._native is not None:
            return
        count = len(self.species)
        self._alpha_max = _per_species(self._alpha_max_config, count, "alpha_max", 8.0, integer=True)
        self._atom_sigma_r = _per_species(self._atom_sigma_r_config, count, "atom_sigma_r", 0.5)
        self._atom_sigma_r_scaling = _per_species(self._atom_sigma_r_scaling_config, count, "atom_sigma_r_scaling", 0.0)
        self._atom_sigma_t = _per_species(self._atom_sigma_t_config, count, "atom_sigma_t", 0.5)
        self._atom_sigma_t_scaling = _per_species(self._atom_sigma_t_scaling_config, count, "atom_sigma_t_scaling", 0.0)
        self._amplitude_scaling = _per_species(self._amplitude_scaling_config, count, "amplitude_scaling", 0.0)
        self._central_weight = _per_species(self._central_weight_config, count, "central_weight", 1.0)
        if self.central_species is not None and not set(self.central_species).issubset(self.species):
            raise ValueError("central_species must be contained in species")
        options = _cpp.SoapTurboOptions()
        options.species = list(self.species)
        options.alpha_max = self._alpha_max
        options.central_species = list(self.central_species or ())
        options.atom_sigma_r = self._atom_sigma_r
        options.atom_sigma_r_scaling = self._atom_sigma_r_scaling
        options.atom_sigma_t = self._atom_sigma_t
        options.atom_sigma_t_scaling = self._atom_sigma_t_scaling
        options.amplitude_scaling = self._amplitude_scaling
        options.central_weight = self._central_weight
        options.l_max = self.l_max
        options.rcut_hard = self.rcut_hard
        options.rcut_soft = self.rcut_soft
        options.nf = self.nf
        options.radial_enhancement = self.radial_enhancement
        options.basis = {"poly3": 0, "poly3gauss": 1}[self.basis]
        options.compression = self.compression
        options.num_threads = 0 if self.num_threads is None else int(self.num_threads)
        self._native = _cpp.SoapTurboCalculator(options)

    @property
    def feature_count(self) -> int:
        alpha_max = self._alpha_max
        if alpha_max is None and self.species is not None:
            alpha_max = _per_species(self._alpha_max_config, len(self.species), "alpha_max", 8.0, integer=True)
        if not alpha_max:
            return 0
        channels = sum(alpha_max)
        if not self.compression:
            return channels * (channels + 1) // 2 * (self.l_max + 1)
        if self.compression == "trivial":
            pivots = []
            pivot = 0
            for count in alpha_max:
                pivots.append(pivot)
                pivot += count
            retained = sum(
                any(first == pivot or second == pivot for pivot in pivots)
                for first in range(channels)
                for second in range(first, channels)
            )
            # Each retained pair contributes one value for every l.
            return retained * (self.l_max + 1)
        if len(self.compression) != 3 or self.compression[1] != "_":
            raise ValueError("compression must be empty, trivial, or 0_0 through 2_2")
        nu_r, nu_s = int(self.compression[0]), int(self.compression[2])
        if nu_r not in range(3) or nu_s not in range(3):
            raise ValueError("compression must be empty, trivial, or 0_0 through 2_2")
        if len(set(alpha_max)) != 1:
            raise ValueError("0_0 through 2_2 compression requires equal alpha_max")
        n1 = alpha_max[0] if nu_r > 0 else 1
        n2 = alpha_max[0] if nu_r == 2 else 1
        s1 = len(alpha_max) if nu_s > 0 else 1
        s2 = len(alpha_max) if nu_s == 2 else 1
        if nu_r % 2 == 0 and nu_s % 2 == 0:
            channels = n1 * s1
            return channels * (channels + 1) // 2 * (self.l_max + 1)
        return n1 * s1 * n2 * s2 * (self.l_max + 1)

    def compute(self, batch: StructureBatch | Any, control: Any = None) -> DescriptorResult:
        batch = _as_batch(batch)
        self._ensure_native(batch)
        values = self._native.compute(
            batch.numbers, batch.positions, batch.cells, batch.pbc, batch.offsets, control)
        values = _format_output(values, self.dtype, self.sparse)
        return DescriptorResult(
            values, "atom", batch.ids, batch.offsets.copy(), self._labels(), self._metadata())

    def create(self, batch: StructureBatch | Any, control: Any = None) -> Any:
        return self.compute(batch, control).values

    def close(self) -> None:
        self._closed = True
        if self._native is not None:
            self._native.close()

    def _labels(self) -> tuple[str, ...]:
        if self._alpha_max is None or self.species is None:
            return ()
        if self.compression:
            return tuple(
                f"soapturbo:compression={self.compression},index={index}"
                for index in range(self.feature_count)
            )
        channels = [
            (species, radial)
            for species, count in zip(self.species, self._alpha_max)
            for radial in range(count)
        ]
        return tuple(
            f"soapturbo:z1={first[0]},n1={first[1]},z2={second[0]},n2={second[1]},l={l}"
            for first_index, first in enumerate(channels)
            for second in channels[first_index:]
            for l in range(self.l_max + 1)
        )

    def _metadata(self) -> dict[str, Any]:
        return {
            "backend": "mdescriptor-cpp",
            "descriptor": self.name,
            "species": self.species,
            "alpha_max": tuple(self._alpha_max or ()),
            "l_max": self.l_max,
            "rcut_hard": self.rcut_hard,
            "rcut_soft": self.rcut_soft,
            "basis": self.basis,
            "compression": self.compression or None,
            "central_species": self.central_species,
            "radial_enhancement": self.radial_enhancement,
            "dtype": self.dtype,
            "sparse": self.sparse,
        }


__all__ = ["SoapTurboCalculator"]
