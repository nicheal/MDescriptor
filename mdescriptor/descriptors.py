"""Python/ASE front-end for the MDescriptor C++ descriptors."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import shutil
import sys
import threading
from typing import Any, Iterable, Literal, Sequence

import numpy as np

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
    from . import _descriptor_cpp as _cpp
except ImportError as exc:  # pragma: no cover - exercised only before a build
    raise ImportError(
        "MDescriptor's native descriptor module is not built; install the project with `python -m pip install -e .`."
    ) from exc


ComputeControl = _cpp.ComputeControl
CancelledError = _cpp.CancelledError
_build_neighbor_graph = _cpp.build_neighbor_graph
_compute_coulomb_matrix = _cpp.compute_coulomb_matrix
_compute_featomic_atomic_composition = _cpp.compute_featomic_atomic_composition
_compute_featomic_sorted_distances = _cpp.compute_featomic_sorted_distances
_compute_featomic_neighbor_list = _cpp.compute_featomic_neighbor_list
_compute_featomic_spherical = _cpp.compute_featomic_spherical
_compute_featomic_spherical_by_pair = _cpp.compute_featomic_spherical_by_pair

@dataclass(frozen=True)
class StructureBatch:
    """A contiguous, reusable snapshot of fully periodic structures.

    ``spins`` and ``charge_spin`` are optional inputs for descriptors such as
    DPA4C: the former has shape ``(total_atoms, 3)`` and the latter has one
    ``[charge, multiplicity]`` pair per structure.
    """

    numbers: np.ndarray
    positions: np.ndarray
    cells: np.ndarray
    pbc: np.ndarray
    offsets: np.ndarray
    ids: tuple[str, ...]
    spins: np.ndarray | None = None
    charge_spin: np.ndarray | None = None

    def __post_init__(self) -> None:
        numbers = np.ascontiguousarray(self.numbers, dtype=np.int32)
        positions = np.ascontiguousarray(self.positions, dtype=np.float64)
        cells = np.ascontiguousarray(self.cells, dtype=np.float64)
        pbc = np.ascontiguousarray(self.pbc, dtype=np.int32)
        offsets = np.ascontiguousarray(self.offsets, dtype=np.int64)
        ids = tuple(str(value) for value in self.ids)

        spins = None
        if self.spins is not None:
            spins = np.ascontiguousarray(self.spins, dtype=np.float64)
        charge_spin = None
        if self.charge_spin is not None:
            charge_spin = np.ascontiguousarray(self.charge_spin, dtype=np.float64)

        if numbers.ndim != 1:
            raise ValueError("numbers must be a one-dimensional int32 array")
        if np.any(numbers <= 0):
            raise ValueError("atomic numbers must be positive")
        if positions.shape != (len(numbers), 3):
            raise ValueError("positions must have shape (total_atoms, 3)")
        if cells.ndim != 3 or cells.shape[1:] != (3, 3):
            raise ValueError("cells must have shape (structures, 3, 3)")
        if pbc.ndim != 2 or pbc.shape[1:] != (3,):
            raise ValueError("pbc must have shape (structures, 3)")
        if offsets.ndim != 1 or len(offsets) != len(ids) + 1:
            raise ValueError("offsets must have one entry per structure plus a sentinel")
        if len(cells) != len(ids) or len(pbc) != len(ids):
            raise ValueError("structure arrays and ids have inconsistent lengths")
        if len(offsets) and (offsets[0] != 0 or offsets[-1] != len(numbers)):
            raise ValueError("offsets must start at zero and end at total_atoms")
        if np.any(offsets[1:] < offsets[:-1]):
            raise ValueError("offsets must be monotonic")
        if not np.isfinite(positions).all() or not np.isfinite(cells).all():
            raise ValueError("positions and cells must be finite")
        if spins is not None and (spins.ndim != 2 or spins.shape != (len(numbers), 3)):
            raise ValueError("spins must have shape (total_atoms, 3)")
        if charge_spin is not None and (
            charge_spin.ndim != 2 or charge_spin.shape != (len(ids), 2)
        ):
            raise ValueError("charge_spin must have shape (structures, 2)")
        if spins is not None and not np.isfinite(spins).all():
            raise ValueError("spins must be finite")
        if charge_spin is not None and not np.isfinite(charge_spin).all():
            raise ValueError("charge_spin must be finite")
        if np.any(pbc != 1):
            raise ValueError("only fully periodic structures (pbc=(1, 1, 1)) are supported")
        for matrix in cells:
            if abs(float(np.linalg.det(matrix))) < 1e-14:
                raise ValueError("cells must be nonsingular")

        object.__setattr__(self, "numbers", numbers)
        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "cells", cells)
        object.__setattr__(self, "pbc", pbc)
        object.__setattr__(self, "offsets", offsets)
        object.__setattr__(self, "ids", ids)
        object.__setattr__(self, "spins", spins)
        object.__setattr__(self, "charge_spin", charge_spin)

    @property
    def structures(self) -> int:
        return len(self.ids)

    @property
    def atoms(self) -> int:
        return len(self.numbers)

    @classmethod
    def from_ase(
        cls,
        structures: Sequence[Any] | Any,
        ids: Sequence[str] | None = None,
    ) -> "StructureBatch":
        """Pack an ASE ``Atoms`` object or sequence into the native batch layout."""

        try:
            from ase import Atoms
        except ImportError as exc:  # pragma: no cover
            raise ImportError("ASE is required to build a StructureBatch") from exc

        if isinstance(structures, Atoms):
            structures = [structures]
        else:
            structures = list(structures)
        if ids is not None and len(ids) != len(structures):
            raise ValueError("ids must have one entry per structure")

        number_parts: list[np.ndarray] = []
        position_parts: list[np.ndarray] = []
        cell_parts: list[np.ndarray] = []
        pbc_parts: list[np.ndarray] = []
        spin_parts: list[np.ndarray] = []
        frame_charge_spin: list[np.ndarray] = []
        have_spins = False
        have_charge_spin = False
        offsets = [0]
        generated_ids: list[str] = []
        for index, atoms in enumerate(structures):
            if not isinstance(atoms, Atoms):
                raise TypeError("structures must contain ASE Atoms objects")
            number_parts.append(np.asarray(atoms.get_atomic_numbers(), dtype=np.int32))
            position_parts.append(np.asarray(atoms.get_positions(), dtype=np.float64))
            cell_parts.append(np.asarray(atoms.cell.array, dtype=np.float64))
            pbc_parts.append(np.asarray(atoms.get_pbc(), dtype=np.int32))
            atom_spin = atoms.arrays.get("spin", atoms.arrays.get("spins"))
            if atom_spin is None:
                atom_spin = atoms.info.get("spin", atoms.info.get("spins"))
            if atom_spin is not None:
                have_spins = True
                spin_parts.append(np.asarray(atom_spin, dtype=np.float64))
            else:
                spin_parts.append(np.zeros((len(atoms), 3), dtype=np.float64))
            frame_state = atoms.info.get("charge_spin")
            if frame_state is None:
                charge = atoms.info.get("charge")
                multiplicity = atoms.info.get("spin_multiplicity")
                if charge is not None or multiplicity is not None:
                    frame_state = (0.0 if charge is None else charge, 0.0 if multiplicity is None else multiplicity)
            if frame_state is not None:
                have_charge_spin = True
                frame_charge_spin.append(np.asarray(frame_state, dtype=np.float64))
            else:
                frame_charge_spin.append(np.zeros(2, dtype=np.float64))
            offsets.append(offsets[-1] + len(atoms))
            if ids is not None:
                generated_ids.append(str(ids[index]))
            else:
                source = atoms.info.get("source_path", atoms.info.get("_source_path"))
                frame = atoms.info.get("frame", index)
                generated_ids.append(
                    f"{Path(source).resolve()}#{frame}" if source else str(index)
                )

        numbers = np.concatenate(number_parts) if number_parts else np.empty(0, dtype=np.int32)
        positions = (
            np.concatenate(position_parts, axis=0)
            if position_parts
            else np.empty((0, 3), dtype=np.float64)
        )
        cells = np.stack(cell_parts) if cell_parts else np.empty((0, 3, 3), dtype=np.float64)
        pbc = np.stack(pbc_parts) if pbc_parts else np.empty((0, 3), dtype=np.int32)
        spins = np.concatenate(spin_parts, axis=0) if have_spins else None
        charge_spin = np.stack(frame_charge_spin) if have_charge_spin else None
        return cls(
            numbers,
            positions,
            cells,
            pbc,
            np.asarray(offsets),
            tuple(generated_ids),
            spins,
            charge_spin,
        )


batch_from_ase = StructureBatch.from_ase


@dataclass(frozen=True)
class DescriptorResult:
    values: Any
    level: Literal["atom", "structure", "pair"]
    structure_ids: tuple[str, ...]
    atom_offsets: np.ndarray | None
    labels: tuple[str, ...]
    metadata: dict[str, Any]

    @property
    def shape(self) -> tuple[int, ...]:
        return self.values.shape

    def __array__(self, dtype: Any = None) -> np.ndarray:
        values = self.values.todense() if hasattr(self.values, "todense") else self.values
        return np.asarray(values, dtype=dtype)


def _as_batch(value: StructureBatch | Sequence[Any] | Any) -> StructureBatch:
    return value if isinstance(value, StructureBatch) else StructureBatch.from_ase(value)


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
    """Construct DScribe's orthonormalized GTO radial basis."""

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
    """Construct DScribe's orthonormalized polynomial radial basis and quadrature."""
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
