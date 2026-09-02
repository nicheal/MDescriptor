"""Canonical configuration shared by the MBTR-family adapters.

The public descriptors intentionally expose slightly different constructors,
but their native implementations consume the same small set of normalized
controls.  Keeping that translation here gives the CPU and CUDA adapters one
configuration seam instead of making each backend interpret public options on
its own.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import log
from typing import Any

_GEOMETRY_CODES = {
    "atomic_number": 0,
    "distance": 1,
    "inverse_distance": 2,
    "angle": 3,
    "cosine": 4,
}
_WEIGHTING_CODES = {
    "unity": 0,
    "none": 0,
    "exp": 1,
    "inverse_square": 2,
    "smooth_cutoff": 3,
}
_NORMALIZATION_CODES = {
    "none": 0,
    "l2": 1,
    "n_atoms": 2,
    "valle_oganov": 3,
}


@dataclass(frozen=True, slots=True)
class MBTRConfig:
    """Backend-neutral, validated MBTR controls.

    ``r_cut`` is the effective neighbor cutoff.  For inverse-square and
    smooth-cutoff weighting it is also the weighting cutoff.  Exponential and
    unity weighting do not read it when evaluating a contribution, so using
    the effective value there is compatible with both the CPU and CUDA graph
    builders.
    """

    species: tuple[int, ...]
    geometry: int
    weighting: int
    normalization: int
    grid_min: float
    grid_max: float
    grid_sigma: float
    grid_n: int
    normalize_gaussians: bool
    scale: float
    threshold: float
    r_cut: float
    sharpness: float
    local: bool
    num_threads: int

    @property
    def feature_count(self) -> int:
        """Return the canonical dense width for this layout."""

        species_count = len(self.species)
        pair_count = species_count * (species_count + 1) // 2
        if self.geometry == _GEOMETRY_CODES["atomic_number"]:
            channels = species_count
        elif self.local:
            channels = (
                species_count + 1
                if self.geometry in {
                    _GEOMETRY_CODES["distance"],
                    _GEOMETRY_CODES["inverse_distance"],
                }
                else (species_count + 1) * (3 * (species_count + 1) - 1) // 2
            )
        else:
            channels = (
                pair_count
                if self.geometry in {
                    _GEOMETRY_CODES["distance"],
                    _GEOMETRY_CODES["inverse_distance"],
                }
                else species_count * pair_count
            )
        return channels * self.grid_n

    def native_kwargs(self) -> dict[str, Any]:
        """Return named values for the private CPU binding."""

        return {
            "geometry": self.geometry,
            "weighting": self.weighting,
            "normalization": self.normalization,
            "grid_min": self.grid_min,
            "grid_max": self.grid_max,
            "grid_sigma": self.grid_sigma,
            "grid_n": self.grid_n,
            "normalize_gaussians": self.normalize_gaussians,
            "scale": self.scale,
            "threshold": self.threshold,
            "r_cut": self.r_cut,
            "sharpness": self.sharpness,
            "local": self.local,
            "num_threads": self.num_threads,
        }

    def cuda_payload(self) -> dict[str, Any]:
        """Return named primitive values for the private CUDA seam."""

        return {
            "schema_version": 1,
            "species": list(self.species),
            "geometry": self.geometry,
            "weighting": self.weighting,
            "normalization": self.normalization,
            "grid_min": self.grid_min,
            "grid_max": self.grid_max,
            "grid_sigma": self.grid_sigma,
            "grid_n": self.grid_n,
            "normalize_gaussians": self.normalize_gaussians,
            "scale": self.scale,
            "threshold": self.threshold,
            "r_cut": self.r_cut,
            "sharpness": self.sharpness,
            "local": self.local,
        }


def _code(
    values: Mapping[str, int], function: str, *, description: str
) -> int:
    try:
        return values[function]
    except KeyError as exc:
        raise ValueError(f"unsupported MBTR {description}: {function}") from exc


def resolve_mbtr_config(
    *,
    species: tuple[int, ...],
    geometry: Mapping[str, Any],
    grid: Mapping[str, Any],
    weighting: Mapping[str, Any],
    periodic: bool,
    normalize_gaussians: bool,
    normalization: str,
    local: bool,
    num_threads: int,
) -> MBTRConfig:
    """Resolve one public constructor state into canonical native controls."""

    if not periodic:
        raise ValueError("only periodic MBTR is supported")
    geometry_code = _code(
        _GEOMETRY_CODES,
        str(geometry.get("function", "distance")),
        description="geometry",
    )
    weighting_code = _code(
        _WEIGHTING_CODES,
        str(weighting.get("function", "unity")),
        description="weighting",
    )
    if normalization not in _NORMALIZATION_CODES:
        raise ValueError("unsupported MBTR normalization")
    if num_threads < 0:
        raise ValueError("num_threads must be non-negative")

    grid_min = float(grid.get("min", 0.0))
    grid_max = float(grid.get("max", 6.0))
    grid_sigma = float(grid.get("sigma", 0.1))
    grid_n = int(grid.get("n", 50))
    if grid_n < 2 or grid_max <= grid_min or grid_sigma <= 0.0:
        raise ValueError("invalid MBTR grid")

    scale = float(weighting.get("scale", 0.5))
    threshold = float(weighting.get("threshold", 1e-3))
    default_cutoff = 0.0 if weighting_code in {0, 1} else grid_max
    requested_cutoff = float(weighting.get("r_cut", default_cutoff))
    sharpness = float(weighting.get("sharpness", 2.0))
    if weighting_code == 1 and (scale <= 0.0 or not 0.0 < threshold < 1.0):
        raise ValueError("exponential MBTR weighting needs positive scale and threshold in (0, 1)")
    if weighting_code in {2, 3} and requested_cutoff <= 0.0:
        raise ValueError("cutoff weighting needs a positive r_cut")

    effective_cutoff = requested_cutoff if requested_cutoff > 0.0 else 0.0
    if weighting_code == 1:
        multiplier = 0.5 if geometry_code in {3, 4} else 1.0
        effective_cutoff = max(
            effective_cutoff,
            multiplier * -log(threshold) / scale,
        )
    if effective_cutoff <= 0.0:
        effective_cutoff = grid_max

    return MBTRConfig(
        species=species,
        geometry=geometry_code,
        weighting=weighting_code,
        normalization=_NORMALIZATION_CODES[normalization],
        grid_min=grid_min,
        grid_max=grid_max,
        grid_sigma=grid_sigma,
        grid_n=grid_n,
        normalize_gaussians=bool(normalize_gaussians),
        scale=scale,
        threshold=threshold,
        r_cut=effective_cutoff,
        sharpness=sharpness,
        local=bool(local),
        num_threads=num_threads,
    )


__all__ = ["MBTRConfig", "resolve_mbtr_config"]
