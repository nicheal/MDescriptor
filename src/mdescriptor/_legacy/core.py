"""Python/ASE front-end for the MDescriptor C++ descriptors."""

from __future__ import annotations

from pathlib import Path
import os
import shutil
import sys
import threading
from typing import Any, Iterable, Sequence

import numpy as np

from ..core.input import StructureBatch, StructureInput
from ..core.result import DescriptorResult

# MinGW builds need their runtime DLL directory registered before importing the
# extension. Release MSVC wheels do not enter this branch.
_dll_directories = []
if os.name == "nt":
    candidates = {Path(sys.executable).parent}
    gxx = shutil.which("g++")
    if gxx:
        candidates.add(Path(gxx).resolve().parent)
    for directory in candidates:
        if directory.is_dir() and hasattr(os, "add_dll_directory"):
            _dll_directories.append(os.add_dll_directory(str(directory)))

try:
    from .. import _native as _cpp
except ImportError as exc:  # pragma: no cover - exercised only before a build
    raise ImportError(
        "MDescriptor's native descriptor module is not built; install the project with `python -m pip install -e .`."
    ) from exc


ComputeControl = _cpp.ComputeControl
CancelledError = _cpp.CancelledError
_build_neighbor_graph = _cpp.build_neighbor_graph
_compute_coulomb_matrix = _cpp.compute_coulomb_matrix
_compute_atomic_composition = _cpp.compute_atomic_composition
_compute_sorted_distances = _cpp.compute_sorted_distances
_compute_neighbor_list = _cpp.compute_neighbor_list
_compute_spherical_expansion = _cpp.compute_spherical_expansion
_compute_spherical_expansion_by_pair = _cpp.compute_spherical_expansion_by_pair


def _as_batch(value: StructureInput) -> StructureBatch:
    """Normalize all descriptor inputs through the canonical core batch."""

    if isinstance(value, StructureBatch):
        return value
    return StructureBatch.from_ase(value)


batch_from_ase = StructureBatch.from_ase


def _species_from_batch(batch: StructureBatch) -> tuple[int, ...]:
    return tuple(int(value) for value in np.unique(batch.numbers))


def _normalise_species(species: Iterable[int] | None) -> tuple[int, ...] | None:
    if species is None:
        return None
    result = tuple(int(value) for value in species)
    if not result or len(set(result)) != len(result):
        raise ValueError("species must be a non-empty sequence of unique atomic numbers")
    if any(value <= 0 for value in result):
        raise ValueError("species must contain positive atomic numbers")
    return result


def _basis_gto(r_cut: float, n_max: int, l_max: int) -> tuple[np.ndarray, np.ndarray]:
    """Construct reference implementation's orthonormalized GTO radial basis."""

    from math import gamma

    radii = np.linspace(1.0, r_cut, n_max)
    alphas = np.empty((l_max + 1, n_max), dtype=np.float64)
    betas = np.empty((l_max + 1, n_max, n_max), dtype=np.float64)
    for l in range(l_max + 1):
        alpha = -np.log(1e-3 / radii**l) / radii**2
        overlap = 0.5 * gamma(l + 1.5) * (alpha[:, None] + alpha[None, :]) ** (-l - 1.5)
        eigenvalues, eigenvectors = np.linalg.eigh(overlap)
        if np.any(eigenvalues <= 0):
            raise ValueError("could not normalize the SOAP GTO radial basis")
        betas[l] = (eigenvectors * (1.0 / np.sqrt(eigenvalues))) @ eigenvectors.T
        alphas[l] = alpha
    return alphas, betas


def _basis_polynomial(r_cut: float, n_max: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct reference implementation's orthonormalized polynomial radial basis and quadrature."""
    nodes, weights = np.polynomial.legendre.leggauss(100)
    grid = 0.5 * r_cut * (nodes + 1.0)
    quadrature_weights = 0.5 * r_cut * weights
    overlap = np.empty((n_max, n_max), dtype=np.float64)
    for i in range(1, n_max + 1):
        for j in range(1, n_max + 1):
            overlap[i - 1, j - 1] = (2.0 * r_cut ** (7 + i + j)) / (
                (5 + i + j) * (6 + i + j) * (7 + i + j)
            )
    eigenvalues, eigenvectors = np.linalg.eigh(overlap)
    if np.any(eigenvalues <= 0.0):
        raise ValueError("could not normalize the SOAP polynomial radial basis")
    betas = (eigenvectors * (1.0 / np.sqrt(eigenvalues))) @ eigenvectors.T
    powers = np.asarray([(r_cut - np.clip(grid, 0.0, r_cut)) ** (n + 3) for n in range(n_max)])
    return grid, quadrature_weights, betas @ powers


def _soap_weighting_config(
    weighting: dict[str, Any] | None,
    r_cut: float | None,
) -> tuple[dict[str, Any], float]:
    weighting = dict(weighting or {})
    function = weighting.get("function")
    if function is not None:
        function = str(function)
        if function not in {"poly", "pow", "exp"}:
            raise ValueError("SOAP weighting function must be 'poly', 'pow', or 'exp'")
        r0 = float(weighting.get("r0", 0.0))
        c = float(weighting.get("c", -1.0))
        if r0 <= 0.0 or c < 0.0:
            raise ValueError("SOAP weighting requires r0 > 0 and c >= 0")
        weighting["r0"], weighting["c"] = r0, c
        if function == "poly":
            m = float(weighting.get("m", -1.0))
            if m < 0.0:
                raise ValueError("SOAP poly weighting requires m >= 0")
            weighting["m"] = m
        else:
            d = float(weighting.get("d", -1.0))
            if d < 0.0:
                raise ValueError("SOAP pow/exp weighting requires d >= 0")
            weighting["d"] = d
            if function == "pow":
                m = float(weighting.get("m", -1.0))
                if m < 0.0:
                    raise ValueError("SOAP pow weighting requires m >= 0")
                weighting["m"] = m
                weighting["threshold"] = float(weighting.get("threshold", 1e-2))
            else:
                weighting["threshold"] = float(weighting.get("threshold", 1e-2))
            if weighting["threshold"] <= 0.0:
                raise ValueError("SOAP weighting threshold must be positive")
    if "w0" in weighting:
        weighting["w0"] = float(weighting["w0"])
        if weighting["w0"] < 0.0:
            raise ValueError("SOAP weighting w0 must be non-negative")
    if r_cut is not None:
        resolved = float(r_cut)
    elif function == "poly":
        resolved = weighting["r0"]
    elif function == "pow":
        argument = weighting["c"] / weighting["threshold"] - weighting["d"]
        if weighting["m"] <= 0.0 or argument <= 0.0:
            raise ValueError("cannot infer SOAP pow weighting cutoff")
        resolved = weighting["r0"] * argument ** (1.0 / weighting["m"])
    elif function == "exp":
        argument = weighting["c"] / weighting["threshold"] - weighting["d"]
        if argument <= 1.0:
            raise ValueError("cannot infer SOAP exp weighting cutoff")
        resolved = weighting["r0"] * np.log(argument)
    else:
        resolved = 6.0
    return weighting, resolved


def _merge_config(config: dict[str, Any] | None, kwargs: dict[str, Any]) -> dict[str, Any]:
    merged = dict(config or {})
    merged.update({key: value for key, value in kwargs.items() if value is not None})
    return merged


def _format_output(values: np.ndarray, dtype: str, sparse: bool) -> Any:
    values = values.astype(dtype, copy=False)
    if not sparse:
        return values
    try:
        import sparse as sparse_module
    except ImportError as exc:  # pragma: no cover - depends on optional output support
        raise ImportError("sparse=True requires the optional 'sparse' package") from exc
    return sparse_module.COO.from_numpy(values)


class SoapCalculator:
    """Stateful periodic SOAP calculator backed by the C++ batch kernel."""

    def __init__(self, species: Iterable[int] | None = None, config: dict[str, Any] | None = None, **kwargs: Any):
        config = _merge_config(config, kwargs)
        self.rbf = str(config.get("rbf", "gto"))
        self.n_max = int(config.get("n_max", 8))
        self.l_max = int(config.get("l_max", 6))
        self.sigma = float(config.get("sigma", 1.0))
        self.average = str(config.get("average", "inner"))
        self.dtype = str(config.get("dtype", "float64"))
        self.sparse = bool(config.get("sparse", False))
        if self.dtype not in {"float32", "float64"}:
            raise ValueError("dtype must be 'float32' or 'float64'")
        self.weighting, self.r_cut = _soap_weighting_config(config.get("weighting"), config.get("r_cut"))
        compression = config.get("compression") or {"mode": "off", "species_weighting": None}
        if not isinstance(compression, dict):
            raise ValueError("compression must be a dictionary")
        self.compression = dict(compression)
        self.compression_mode = str(self.compression.get("mode", "off"))
        if self.compression_mode not in {"off", "mu2", "mu1nu1", "crossover"}:
            raise ValueError("invalid SOAP compression mode")
        self.num_threads = config.get("num_threads")
        if self.num_threads is not None and int(self.num_threads) <= 0:
            raise ValueError("num_threads must be a positive integer or None")
        if self.rbf not in {"gto", "polynomial"}:
            raise ValueError("rbf must be 'gto' or 'polynomial'")
        if self.r_cut <= 0 or self.n_max < 1 or self.l_max < 0 or self.l_max > 20 or self.sigma <= 0:
            raise ValueError("invalid SOAP parameters")
        if not np.isfinite([self.r_cut, self.sigma]).all():
            raise ValueError("SOAP parameters must be finite")
        if self.rbf == "gto" and self.r_cut <= 1.0:
            raise ValueError("SOAP GTO radial basis requires r_cut > 1")
        if self.average not in {"off", "inner", "outer"}:
            raise ValueError("average must be 'off', 'inner', or 'outer'")
        self.species = _normalise_species(species)
        species_weighting = self.compression.get("species_weighting")
        if species_weighting is not None and not isinstance(species_weighting, dict):
            raise ValueError("species_weighting must be a dictionary or None")
        self.species_weighting = species_weighting
        self._native: Any = None
        self._init_lock = threading.Lock()
        self._closed = False

    def _ensure_native(self, batch: StructureBatch) -> None:
        with self._init_lock:
            if self._closed:
                raise RuntimeError("SOAP calculator is closed")
            if self.species is None:
                self.species = _species_from_batch(batch)
            missing = set(np.unique(batch.numbers)) - set(self.species)
            if missing:
                raise ValueError(f"batch contains species not fixed in calculator: {sorted(missing)}")
            if self._native is None:
                options = _cpp.SoapOptions()
                options.species = list(self.species)
                options.r_cut = self.r_cut
                options.n_max = self.n_max
                options.l_max = self.l_max
                options.sigma = self.sigma
                options.radial_basis = 0 if self.rbf == "gto" else 1
                if self.rbf == "gto":
                    alphas, betas = _basis_gto(self.r_cut, self.n_max, self.l_max)
                    options.alphas = alphas.ravel().tolist()
                    options.betas = betas.ravel().tolist()
                else:
                    grid, weights, values = _basis_polynomial(self.r_cut, self.n_max)
                    options.radial_grid = grid.tolist()
                    options.radial_weights = weights.tolist()
                    options.radial_values = values.ravel().tolist()
                options.weighting_has_function = "function" in self.weighting
                options.weighting_function = {"poly": 1, "pow": 2, "exp": 3}.get(self.weighting.get("function"), 0)
                options.weighting_has_w0 = "w0" in self.weighting
                options.weighting_r0 = float(self.weighting.get("r0", 1.0))
                options.weighting_c = float(self.weighting.get("c", 1.0))
                options.weighting_d = float(self.weighting.get("d", 0.0))
                options.weighting_m = float(self.weighting.get("m", 1.0))
                options.weighting_threshold = float(self.weighting.get("threshold", 1e-2))
                options.weighting_w0 = float(self.weighting.get("w0", 1.0))
                if self.species_weighting is None:
                    options.species_weights = [1.0] * len(self.species)
                else:
                    if set(self.species_weighting) != set(self.species):
                        raise ValueError("species_weighting must contain exactly the configured species")
                    options.species_weights = [float(self.species_weighting[value]) for value in self.species]
                options.compression = {"off": 0, "mu2": 1, "mu1nu1": 2, "crossover": 3}[self.compression_mode]
                options.inner_average = self.average == "inner"
                options.outer_average = self.average == "outer"
                options.num_threads = 0 if self.num_threads is None else int(self.num_threads)
                self._native = _cpp.SoapCalculator(options)

    @property
    def feature_count(self) -> int:
        if not self.species:
            return 0
        species = len(self.species)
        if self.compression_mode == "mu2":
            return self.n_max * (self.n_max + 1) // 2 * (self.l_max + 1)
        if self.compression_mode == "mu1nu1":
            return species * self.n_max * self.n_max * (self.l_max + 1)
        if self.compression_mode == "crossover":
            return species * self.n_max * (self.n_max + 1) // 2 * (self.l_max + 1)
        width = species * self.n_max
        return width * (width + 1) // 2 * (self.l_max + 1)

    def compute(self, batch: StructureBatch | Sequence[Any] | Any, control: Any = None) -> DescriptorResult:
        batch = _as_batch(batch)
        self._ensure_native(batch)
        native_inner_average = self.average == "inner"
        native_outer_average = self.average == "outer"
        values = self._native.compute(
            batch.numbers,
            batch.positions,
            batch.cells,
            batch.pbc,
            batch.offsets,
            control,
            0 if self.num_threads is None else int(self.num_threads),
            native_inner_average,
            native_outer_average,
        )
        if self.average == "outer":
            values = _format_output(values, self.dtype, self.sparse)
            return DescriptorResult(values, "structure", batch.ids, None, self._labels(), self._metadata())
        if self.average == "inner":
            values = _format_output(values, self.dtype, self.sparse)
            return DescriptorResult(values, "structure", batch.ids, None, self._labels(), self._metadata())
        values = _format_output(values, self.dtype, self.sparse)
        return DescriptorResult(values, "atom", batch.ids, batch.offsets.copy(), self._labels(), self._metadata())

    def create(self, batch: StructureBatch | Sequence[Any] | Any, control: Any = None) -> Any:
        return self.compute(batch, control).values

    def close(self) -> None:
        self._closed = True
        if self._native is not None:
            self._native.close()

    def _labels(self) -> tuple[str, ...]:
        labels = []
        if self.compression_mode == "mu2":
            for l in range(self.l_max + 1):
                for n1 in range(self.n_max):
                    for n2 in range(n1, self.n_max):
                        labels.append(f"soap:compression=mu2,l={l},n1={n1},n2={n2}")
            return tuple(labels)
        if self.compression_mode == "mu1nu1":
            for first in self.species:
                for l in range(self.l_max + 1):
                    for n1 in range(self.n_max):
                        for n2 in range(self.n_max):
                            labels.append(f"soap:compression=mu1nu1,z1={first},l={l},n1={n1},n2={n2}")
            return tuple(labels)
        species_pairs = ((first, first) for first in self.species) if self.compression_mode == "crossover" else (
            (first, second) for i, first in enumerate(self.species) for second in self.species[i:]
        )
        for first, second in species_pairs:
            for l in range(self.l_max + 1):
                for n1 in range(self.n_max):
                    for n2 in range(n1 if first == second else 0, self.n_max):
                        labels.append(f"soap:z1={first},z2={second},l={l},n1={n1},n2={n2}")
        return tuple(labels)

    def _metadata(self) -> dict[str, Any]:
        return {
            "backend": "mdescriptor-cpp", "descriptor": "SOAP", "species": self.species,
            "average": self.average, "r_cut": self.r_cut, "n_max": self.n_max,
            "l_max": self.l_max, "sigma": self.sigma, "rbf": self.rbf,
            "weighting": dict(self.weighting), "compression": dict(self.compression),
            "dtype": self.dtype, "sparse": self.sparse,
        }


class AcsfCalculator:
    """Stateful periodic ACSF calculator supporting G1, G2, G3, G4 and G5."""

    def __init__(self, species: Iterable[int] | None = None, config: dict[str, Any] | None = None, **kwargs: Any):
        config = _merge_config(config, kwargs)
        self.r_cut = float(config.get("r_cut", 6.0))
        self.dtype = str(config.get("dtype", "float64"))
        self.sparse = bool(config.get("sparse", False))
        if self.dtype not in {"float32", "float64"}:
            raise ValueError("dtype must be 'float32' or 'float64'")
        self.num_threads = config.get("num_threads")
        if self.num_threads is not None and int(self.num_threads) <= 0:
            raise ValueError("num_threads must be a positive integer or None")
        self.species = _normalise_species(species)
        self.g2_params = self._parse_g2(config.get("g2_params", config.get("G2", config.get("g2", []))))
        self.g3_params = self._parse_g3(config.get("g3_params", config.get("G3", config.get("g3", []))))
        self.g4_params = self._parse_g4(config.get("g4_params", config.get("G4", config.get("g4", []))))
        self.g5_params = self._parse_g5(config.get("g5_params", config.get("G5", config.get("g5", []))))
        if self.r_cut <= 0 or not np.isfinite(self.r_cut):
            raise ValueError("r_cut must be positive")
        self._native: Any = None
        self._init_lock = threading.Lock()
        self._closed = False

    @staticmethod
    def _parse_g2(value: Any) -> np.ndarray:
        if value is None:
            return np.empty((0, 2), dtype=np.float64)
        if isinstance(value, dict):
            eta = np.asarray(value.get("eta", []), dtype=np.float64).ravel()
            rs = np.asarray(value.get("Rs", value.get("rs", [])), dtype=np.float64).ravel()
            value = [(float(e), float(r)) for e in eta for r in rs]
        array = np.asarray(value, dtype=np.float64)
        if array.size == 0:
            return np.empty((0, 2), dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != 2 or not np.isfinite(array).all() or np.any(array[:, 0] <= 0):
            raise ValueError("g2_params must be an (n, 2) array of (eta, Rs)")
        return array

    @staticmethod
    def _parse_g3(value: Any) -> np.ndarray:
        if value is None:
            return np.empty(0, dtype=np.float64)
        array = np.asarray(value, dtype=np.float64)
        if array.size == 0:
            return np.empty(0, dtype=np.float64)
        if array.ndim != 1 or not np.isfinite(array).all():
            raise ValueError("g3_params must be a one-dimensional finite array of kappa")
        return array

    @staticmethod
    def _parse_g4(value: Any) -> np.ndarray:
        if value is None:
            return np.empty((0, 3), dtype=np.float64)
        if isinstance(value, dict):
            eta = np.asarray(value.get("eta", []), dtype=np.float64).ravel()
            zeta = np.asarray(value.get("zeta", []), dtype=np.float64).ravel()
            lambdas = np.asarray(value.get("lambda", value.get("lambdas", [])), dtype=np.float64).ravel()
            value = [(float(e), float(z), float(lam)) for e in eta for z in zeta for lam in lambdas]
        array = np.asarray(value, dtype=np.float64)
        if array.size == 0:
            return np.empty((0, 3), dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != 3 or not np.isfinite(array).all() or np.any(array[:, 0] <= 0) or np.any(array[:, 1] <= 0):
            raise ValueError("g4_params must be an (n, 3) array of (eta, zeta, lambda)")
        return array

    @staticmethod
    def _parse_g5(value: Any) -> np.ndarray:
        if value is None:
            return np.empty((0, 3), dtype=np.float64)
        if isinstance(value, dict):
            eta = np.asarray(value.get("eta", []), dtype=np.float64).ravel()
            zeta = np.asarray(value.get("zeta", []), dtype=np.float64).ravel()
            lambdas = np.asarray(value.get("lambda", value.get("lambdas", [])), dtype=np.float64).ravel()
            value = [(float(e), float(z), float(lam)) for e in eta for z in zeta for lam in lambdas]
        array = np.asarray(value, dtype=np.float64)
        if array.size == 0:
            return np.empty((0, 3), dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != 3 or not np.isfinite(array).all() or np.any(array[:, 0] <= 0) or np.any(array[:, 1] <= 0):
            raise ValueError("g5_params must be an (n, 3) array of (eta, zeta, lambda)")
        return array

    def _ensure_native(self, batch: StructureBatch) -> None:
        with self._init_lock:
            if self._closed:
                raise RuntimeError("ACSF calculator is closed")
            if self.species is None:
                self.species = _species_from_batch(batch)
            missing = set(np.unique(batch.numbers)) - set(self.species)
            if missing:
                raise ValueError(f"batch contains species not fixed in calculator: {sorted(missing)}")
            if self._native is None:
                options = _cpp.AcsfOptions()
                options.species = list(self.species)
                options.r_cut = self.r_cut
                options.g2_params = self.g2_params.ravel().tolist()
                options.g3_params = self.g3_params.ravel().tolist()
                options.g4_params = self.g4_params.ravel().tolist()
                options.g5_params = self.g5_params.ravel().tolist()
                options.n_g2 = len(self.g2_params)
                options.n_g3 = len(self.g3_params)
                options.n_g4 = len(self.g4_params)
                options.n_g5 = len(self.g5_params)
                options.num_threads = 0 if self.num_threads is None else int(self.num_threads)
                self._native = _cpp.AcsfCalculator(options)

    @property
    def feature_count(self) -> int:
        types = len(self.species)
        return ((1 + len(self.g2_params) + len(self.g3_params)) * types
                + (len(self.g4_params) + len(self.g5_params)) * types * (types + 1) // 2
                if self.species else 0)

    def compute(self, batch: StructureBatch | Sequence[Any] | Any, control: Any = None) -> DescriptorResult:
        batch = _as_batch(batch)
        self._ensure_native(batch)
        values = self._native.compute(batch.numbers, batch.positions, batch.cells, batch.pbc, batch.offsets, control)
        values = _format_output(values, self.dtype, self.sparse)
        return DescriptorResult(values, "atom", batch.ids, batch.offsets.copy(), self._labels(), self._metadata())

    def create(self, batch: StructureBatch | Sequence[Any] | Any, control: Any = None) -> Any:
        return self.compute(batch, control).values

    def close(self) -> None:
        self._closed = True
        if self._native is not None:
            self._native.close()

    def _labels(self) -> tuple[str, ...]:
        labels = []
        for species in self.species:
            labels.append(f"acsf:G1:z={species}")
            labels.extend(f"acsf:G2:z={species},eta={eta:g},Rs={rs:g}" for eta, rs in self.g2_params)
            labels.extend(f"acsf:G3:z={species},kappa={kappa:g}" for kappa in self.g3_params)
        for i, first in enumerate(self.species):
            for second in self.species[i:]:
                labels.extend(f"acsf:G4:z1={first},z2={second},eta={eta:g},zeta={zeta:g},lambda={lam:g}" for eta, zeta, lam in self.g4_params)
                labels.extend(f"acsf:G5:z1={first},z2={second},eta={eta:g},zeta={zeta:g},lambda={lam:g}" for eta, zeta, lam in self.g5_params)
        return tuple(labels)

    def _metadata(self) -> dict[str, Any]:
        return {"backend": "mdescriptor-cpp", "descriptor": "ACSF", "species": self.species, "r_cut": self.r_cut, "g2_params": self.g2_params.copy(), "g3_params": self.g3_params.copy(), "g4_params": self.g4_params.copy(), "g5_params": self.g5_params.copy(), "dtype": self.dtype, "sparse": self.sparse}
