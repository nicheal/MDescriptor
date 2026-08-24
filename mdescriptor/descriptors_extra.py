"""C++-backed matrix and histogram descriptors.

This module is deliberately a thin Python adapter. Descriptor kernels are
computed by ``mdescriptor._descriptor_cpp``; NumPy retains DScribe-compatible
sorting semantics and the public array contract, while ASE is used only by
:class:`StructureBatch` input packing.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np

from .descriptors import DescriptorResult, StructureBatch, _as_batch, _cpp


def _periodic_neighbors(
    batch: StructureBatch,
    structure: int,
    cutoff: float,
    *,
    include_self: bool = False,
) -> list[list[tuple[int, np.ndarray, float, tuple[int, int, int]]]]:
    """Compatibility view over the native neighbor graph used by tests/tools."""
    from math import sqrt

    start, stop = int(batch.offsets[structure]), int(batch.offsets[structure + 1])
    offsets, atoms, shifts, displacements, distance2 = _cpp.build_neighbor_graph(
        batch.numbers[start:stop], batch.positions[start:stop], batch.cells[structure], batch.pbc[structure], float(cutoff)
    )
    result = []
    for center in range(stop - start):
        neighbors = []
        for index in range(int(offsets[center]), int(offsets[center + 1])):
            shift = tuple(int(value) for value in shifts[index])
            atom = int(atoms[index])
            if not include_self and atom == center and shift == (0, 0, 0):
                continue
            neighbors.append((atom, np.asarray(displacements[index], dtype=np.float64), sqrt(max(float(distance2[index]), 0.0)), shift))
        result.append(neighbors)
    return result


class _StructureCalculator:
    name = "descriptor"
    level = "structure"

    @property
    def feature_count(self) -> int:
        return int(getattr(self, "_feature_count", 0))

    def create(self, value: StructureBatch | Sequence[Any] | Any, control: Any = None) -> np.ndarray:
        return self.compute(value, control).values


def _sort_matrix_values_like_dscribe(
    values: np.ndarray,
    batch: StructureBatch,
    counts: np.ndarray,
    n_atoms_max: int,
    *,
    sine_diagonal: bool = False,
    exponent: float = 2.4,
) -> np.ndarray:
    """Apply DescriptorMatrix.sort to native matrices with NumPy semantics."""
    for structure, count_value in enumerate(counts):
        count = int(count_value)
        matrix = values[structure].reshape(n_atoms_max, n_atoms_max)
        physical = matrix[:count, :count]
        if sine_diagonal:
            start, stop = int(batch.offsets[structure]), int(batch.offsets[structure + 1])
            diagonal = 0.5 * np.power(batch.numbers[start:stop].astype(np.float64), exponent)
            np.fill_diagonal(physical, diagonal)
        norms = np.linalg.norm(physical, axis=1)
        order = np.argsort(-norms, kind="stable", axis=0)
        sorted_matrix = physical[order][:, order].copy()
        matrix.fill(0.0)
        matrix[:count, :count] = sorted_matrix
    return values


class _MatrixCalculator(_StructureCalculator):
    kind = 0

    def __init__(self, n_atoms_max: int | None = None, permutation: str = "sorted_l2", exponent: float = 2.4):
        self.n_atoms_max = n_atoms_max
        self.permutation = str(permutation)
        self.exponent = float(exponent)
        if self.n_atoms_max is not None and int(self.n_atoms_max) <= 0:
            raise ValueError("n_atoms_max must be positive")
        if self.permutation not in {"none", "sorted_l2", "eigenspectrum"}:
            raise ValueError("permutation must be 'none', 'sorted_l2', or 'eigenspectrum'")

    def compute(self, value: StructureBatch | Sequence[Any] | Any, control: Any = None) -> DescriptorResult:
        batch = _as_batch(value)
        counts = np.diff(batch.offsets)
        max_atoms = int(self.n_atoms_max or (counts.max() if len(counts) else 0))
        columns = max_atoms if self.permutation == "eigenspectrum" else max_atoms * max_atoms
        if not len(counts):
            values = np.empty((0, columns), dtype=np.float64)
        else:
            python_sorted_l2 = self.permutation == "sorted_l2" and self.kind != 2
            native_permutation = "none" if python_sorted_l2 else self.permutation
            if self.kind == 2:
                values = _cpp.compute_coulomb_matrix(
                    batch.numbers, batch.positions, batch.cells, batch.pbc, batch.offsets,
                    max_atoms, native_permutation, self.exponent, control,
                )
            else:
                values = _cpp.compute_matrix(
                    batch.numbers, batch.positions, batch.cells, batch.pbc, batch.offsets,
                    max_atoms, native_permutation, self.exponent, self.kind,
                    getattr(self, "accuracy", 1e-5), getattr(self, "w", 1.0),
                    float(getattr(self, "r_cut", 0.0) or 0.0),
                    float(getattr(self, "g_cut", 0.0) or 0.0),
                    float(getattr(self, "a", 0.0) or 0.0), control,
                )
            if python_sorted_l2:
                values = _sort_matrix_values_like_dscribe(
                    np.asarray(values, dtype=np.float64), batch, counts, max_atoms,
                    sine_diagonal=self.kind == 0, exponent=self.exponent,
                )
            elif self.kind == 0 and self.permutation == "none":
                values = np.asarray(values, dtype=np.float64)
                for structure, count_value in enumerate(counts):
                    count = int(count_value)
                    start, stop = int(batch.offsets[structure]), int(batch.offsets[structure + 1])
                    diagonal = 0.5 * np.power(batch.numbers[start:stop].astype(np.float64), self.exponent)
                    np.fill_diagonal(values[structure].reshape(max_atoms, max_atoms)[:count, :count], diagonal)
        self._feature_count = int(values.shape[1])
        return DescriptorResult(
            np.asarray(values, dtype=np.float64), self.level, batch.ids, None,
            tuple(f"{self.name}:{index}" for index in range(values.shape[1])),
            {"backend": "mdescriptor-cpp", "descriptor": self.name},
        )


class CoulombMatrixCalculator(_MatrixCalculator):
    name = "CoulombMatrix"
    kind = 2


class SineMatrixCalculator(_MatrixCalculator):
    name = "SineMatrix"
    kind = 0


class EwaldSumMatrixCalculator(_MatrixCalculator):
    name = "EwaldSumMatrix"
    kind = 1

    def __init__(
        self, n_atoms_max: int | None = None, permutation: str = "sorted_l2",
        accuracy: float = 1e-5, w: float = 1.0, r_cut: float | None = None,
        g_cut: float | None = None, a: float | None = None,
    ):
        super().__init__(n_atoms_max=n_atoms_max, permutation=permutation)
        self.accuracy, self.w, self.r_cut, self.g_cut, self.a = float(accuracy), float(w), r_cut, g_cut, a
        if not 0.0 < self.accuracy < 1.0 or self.w <= 0.0:
            raise ValueError("accuracy must be between zero and one and w must be positive")
        if (r_cut is None) != (g_cut is None):
            raise ValueError("r_cut and g_cut must be provided together")


def _enum_geometry(function: str) -> int:
    try:
        return {"atomic_number": 0, "distance": 1, "inverse_distance": 2, "angle": 3, "cosine": 4}[function]
    except KeyError as exc:
        raise ValueError(f"unsupported MBTR geometry: {function}") from exc


def _enum_weighting(function: str) -> int:
    try:
        return {"unity": 0, "none": 0, "exp": 1, "inverse_square": 2, "smooth_cutoff": 3}[function]
    except KeyError as exc:
        raise ValueError(f"unsupported MBTR weighting: {function}") from exc


class MBTRCalculator(_StructureCalculator):
    name = "MBTR"

    def __init__(
        self, species: Iterable[int] | None = None, geometry: dict[str, Any] | None = None,
        grid: dict[str, Any] | None = None, weighting: dict[str, Any] | None = None,
        periodic: bool = True, normalize_gaussians: bool = True, normalization: str = "none",
    ):
        self.species = tuple(sorted(int(value) for value in species)) if species is not None else None
        self.geometry = dict(geometry or {"function": "distance"})
        self.grid = dict(grid or {"min": 0.0, "max": 6.0, "n": 50, "sigma": 0.1})
        self.weighting = dict(weighting or {"function": "exp", "scale": 0.5, "threshold": 1e-3})
        if not periodic:
            raise ValueError("only periodic MBTR is supported")
        if normalization not in {"none", "l2", "n_atoms", "valle_oganov"}:
            raise ValueError("unsupported MBTR normalization")
        self.normalize_gaussians, self.normalization = bool(normalize_gaussians), normalization

    def _options(self, batch: StructureBatch, *, local: bool = False) -> tuple[tuple[int, ...], tuple[Any, ...]]:
        species = self.species or tuple(int(value) for value in np.unique(batch.numbers))
        self.species = species
        geometry = _enum_geometry(str(self.geometry.get("function", "distance")))
        weighting = _enum_weighting(str(self.weighting.get("function", "unity")))
        grid = (float(self.grid.get("min", 0.0)), float(self.grid.get("max", 6.0)), float(self.grid.get("sigma", 0.1)), int(self.grid.get("n", 50)))
        scale, threshold = float(self.weighting.get("scale", 0.5)), float(self.weighting.get("threshold", 1e-3))
        default_cutoff = 0.0 if weighting in {0, 1} else grid[1]
        r_cut, sharpness = float(self.weighting.get("r_cut", default_cutoff)), float(self.weighting.get("sharpness", 2.0))
        if weighting == 1 and (scale <= 0.0 or not 0.0 < threshold < 1.0):
            raise ValueError("exponential MBTR weighting needs positive scale and threshold in (0, 1)")
        if weighting in {2, 3} and r_cut <= 0.0:
            raise ValueError("cutoff weighting needs a positive r_cut")
        if grid[3] < 2 or grid[1] <= grid[0] or grid[2] <= 0.0:
            raise ValueError("invalid MBTR grid")
        normalization = {"none": 0, "l2": 1, "n_atoms": 2, "valle_oganov": 3}[self.normalization]
        return species, (geometry, weighting, normalization, *grid, self.normalize_gaussians, scale, threshold, r_cut, sharpness, local)

    def compute(self, value: StructureBatch | Sequence[Any] | Any, control: Any = None) -> DescriptorResult:
        batch = _as_batch(value)
        species, options = self._options(batch, local=self.name == "LMBTR")
        values = np.asarray(_cpp.compute_mbtr(batch.numbers, batch.positions, batch.cells, batch.pbc, batch.offsets, list(species), *options, control), dtype=np.float64)
        self._feature_count = int(values.shape[1])
        return DescriptorResult(
            values, "atom" if self.name == "LMBTR" else "structure", batch.ids,
            batch.offsets.copy() if self.name == "LMBTR" else None,
            tuple(f"{self.name}:{index}" for index in range(values.shape[1])),
            {"backend": "mdescriptor-cpp", "descriptor": self.name, "species": species},
        )


class LMBTRCalculator(MBTRCalculator):
    name = "LMBTR"


class ValleOganovCalculator(MBTRCalculator):
    name = "ValleOganov"

    def __init__(self, species: Iterable[int] | None = None, function: str = "distance", n: int = 50, sigma: float = 0.1, r_cut: float = 6.0, **kwargs: Any):
        if function == "distance":
            kwargs.setdefault("geometry", {"function": "distance"})
            kwargs.setdefault("grid", {"min": 0.0, "max": r_cut, "n": n, "sigma": sigma})
            kwargs.setdefault("weighting", {"function": "inverse_square", "r_cut": r_cut})
        elif function == "angle":
            kwargs.setdefault("geometry", {"function": "angle"})
            kwargs.setdefault("grid", {"min": 0.0, "max": 180.0, "n": n, "sigma": sigma})
            kwargs.setdefault("weighting", {"function": "smooth_cutoff", "r_cut": r_cut})
        else:
            raise ValueError("function must be 'distance' or 'angle'")
        kwargs.setdefault("normalization", "valle_oganov")
        super().__init__(species=species, **kwargs)


__all__ = ["CoulombMatrixCalculator", "SineMatrixCalculator", "EwaldSumMatrixCalculator", "MBTRCalculator", "LMBTRCalculator", "ValleOganovCalculator"]
