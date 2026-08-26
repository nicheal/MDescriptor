"""C++-backed rotational descriptor adapters.

The public names are retained for API compatibility. No external package, SciPy,
Torch, or Python neighbor-loop is used by the calculation path.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from .core import DescriptorResult, StructureBatch, _as_batch, _cpp


class _AtomKernel:
    name = "descriptor"

    @property
    def feature_count(self) -> int:
        return int(getattr(self, "_feature_count", 0))


class EadKernel(_AtomKernel):
    name = "EAD"

    def __init__(
        self,
        parameters: dict[str, Any] | None = None,
        Rc: float = 6.0,
        cutoff: str = "cosine",
        num_threads: int | None = None,
    ):
        parameters = parameters or {"L": 3, "eta": [0.05, 0.1, 0.5], "Rs": [0.0]}
        self.L = int(parameters.get("L", 3))
        self.eta = np.asarray(parameters.get("eta", [0.05]), dtype=np.float64).ravel()
        self.Rs = np.asarray(parameters.get("Rs", [0.0]), dtype=np.float64).ravel()
        self.Rc = float(Rc)
        self.num_threads = 0 if num_threads is None else int(num_threads)
        if cutoff != "cosine" or self.L < 0 or self.Rc <= 0.0 or np.any(self.eta < 0.0):
            raise ValueError("invalid EAD parameters")
        if self.num_threads < 0:
            raise ValueError("num_threads must be non-negative")

    @property
    def feature_count(self) -> int:
        return (self.L + 1) * len(self.eta) * len(self.Rs)

    def compute(self, value: StructureBatch | Sequence[Any] | Any, control: Any = None) -> DescriptorResult:
        batch = _as_batch(value)
        values = np.asarray(
            _cpp.compute_ead(
                batch.numbers, batch.positions, batch.cells, batch.pbc, batch.offsets,
                self.L, self.Rc, self.eta.tolist(), self.Rs.tolist(), self.num_threads, control,
            ),
            dtype=np.float64,
        )
        return DescriptorResult(
            values, "atom", batch.ids, batch.offsets.copy(),
            tuple(f"{self.name}:{index}" for index in range(values.shape[1])),
            {"backend": "mdescriptor-cpp", "descriptor": self.name},
        )


class So3Kernel(_AtomKernel):
    name = "SO3"

    def __init__(
        self,
        nmax: int = 3,
        lmax: int = 3,
        rcut: float = 3.5,
        alpha: float = 2.0,
        weight_on: bool = False,
        num_threads: int | None = None,
    ):
        self.nmax, self.lmax, self.rcut = int(nmax), int(lmax), float(rcut)
        self.alpha, self.weight_on = float(alpha), bool(weight_on)
        self.num_threads = 0 if num_threads is None else int(num_threads)
        if self.nmax < 1 or self.lmax < 0 or self.rcut <= 0.0 or self.alpha <= 0.0:
            raise ValueError("invalid SO3 parameters")
        if self.num_threads < 0:
            raise ValueError("num_threads must be non-negative")

    @property
    def feature_count(self) -> int:
        return (self.lmax + 1) * self.nmax * (self.nmax + 1) // 2

    def compute(self, value: StructureBatch | Sequence[Any] | Any, control: Any = None) -> DescriptorResult:
        batch = _as_batch(value)
        values = np.asarray(_cpp.compute_rotational_descriptors(
            batch.numbers, batch.positions, batch.cells, batch.pbc, batch.offsets,
            0, self.nmax, self.lmax, self.rcut, self.alpha, self.weight_on,
            False, 1.0, 3, 3, self.num_threads, control, 1.0,
        ), dtype=np.float64)
        return DescriptorResult(values, "atom", batch.ids, batch.offsets.copy(), tuple(f"{self.name}:{i}" for i in range(values.shape[1])), {"backend": "mdescriptor-cpp", "descriptor": self.name})


class So4Kernel(_AtomKernel):
    name = "SO4"

    def __init__(
        self,
        lmax: int = 3,
        rcut: float = 3.5,
        normalize_U: bool = False,
        num_threads: int | None = None,
    ):
        self.lmax, self.rcut, self.normalize_U = int(lmax), float(rcut), bool(normalize_U)
        self.num_threads = 0 if num_threads is None else int(num_threads)
        if self.lmax < 0 or self.rcut <= 0.0:
            raise ValueError("invalid SO4 parameters")
        if self.num_threads < 0:
            raise ValueError("num_threads must be non-negative")

    def compute(self, value: StructureBatch | Sequence[Any] | Any, control: Any = None) -> DescriptorResult:
        batch = _as_batch(value)
        values = np.asarray(_cpp.compute_rotational_descriptors(
            batch.numbers, batch.positions, batch.cells, batch.pbc, batch.offsets,
            1, self.lmax + 1, self.lmax, self.rcut, 2.0, False, self.normalize_U,
            1.0, 3, 3, self.num_threads, control, 1.0,
        ), dtype=np.float64)
        self._feature_count = int(values.shape[1])
        return DescriptorResult(values, "atom", batch.ids, batch.offsets.copy(), tuple(f"{self.name}:{i}" for i in range(values.shape[1])), {"backend": "mdescriptor-cpp", "descriptor": self.name})


class SnapKernel(So4Kernel):
    name = "SNAP"

    def __init__(
        self,
        weights: dict[Any, float] | None = None,
        lmax: int = 3,
        rcut: float = 3.5,
        normalize_U: bool = False,
        num_threads: int | None = None,
    ):
        super().__init__(lmax=lmax, rcut=rcut, normalize_U=normalize_U, num_threads=num_threads)
        self.weights = weights or {}

    def _neighbor_weights(self, batch: StructureBatch) -> list[float]:
        if not self.weights:
            return []
        numeric = {}
        symbol_numbers = None
        for key, value in self.weights.items():
            if isinstance(key, (int, np.integer)):
                number = int(key)
            elif isinstance(key, str):
                if symbol_numbers is None:
                    from ase.data import atomic_numbers
                    symbol_numbers = atomic_numbers
                try:
                    number = int(symbol_numbers[key])
                except KeyError as exc:
                    raise ValueError(f"unknown chemical symbol in weights: {key!r}") from exc
            else:
                raise TypeError("weights keys must be atomic numbers or chemical symbols")
            weight = float(value)
            if not np.isfinite(weight):
                raise ValueError("weights must be finite")
            numeric[number] = weight
        return [numeric.get(int(number), 1.0) for number in batch.numbers]

    def compute(self, value: StructureBatch | Sequence[Any] | Any, control: Any = None) -> DescriptorResult:
        batch = _as_batch(value)
        values = np.asarray(_cpp.compute_rotational_descriptors(
            batch.numbers, batch.positions, batch.cells, batch.pbc, batch.offsets,
            2, self.lmax + 1, self.lmax, self.rcut, 2.0, False, self.normalize_U,
            1.0, 3, 3, self.num_threads, control, 0.99363,
            self._neighbor_weights(batch),
        ), dtype=np.float64)
        self._feature_count = int(values.shape[1])
        return DescriptorResult(values, "atom", batch.ids, batch.offsets.copy(), tuple(f"{self.name}:{i}" for i in range(values.shape[1])), {"backend": "mdescriptor-cpp", "descriptor": self.name})


class LbispectrumKernel(SnapKernel):
    name = "LBispectrum"

    def __init__(
        self,
        twojmax: int = 3,
        diagonal: int = 3,
        rfac0: float = 0.99363,
        rmin0: float = 0.0,
        rcutfac: float = 1.0,
        element_profile: dict[Any, dict[str, float]] | None = None,
        element_radii: dict[Any, float] | None = None,
        weights: dict[Any, float] | None = None,
        lmax: int | None = None,
        rcut: float = 3.5,
        normalize_U: bool = False,
        num_threads: int | None = None,
    ):
        self.twojmax, self.diagonal = int(twojmax), int(diagonal)
        self.rfac0, self.rmin0, self.rcutfac = float(rfac0), float(rmin0), float(rcutfac)
        if self.diagonal not in range(4):
            raise ValueError("diagonal must be 0, 1, 2 or 3")
        if element_profile is not None:
            if element_radii is not None or weights is not None:
                raise ValueError("element_profile cannot be combined with weights or element_radii")
            element_radii = {key: float(value["r"]) for key, value in element_profile.items()}
            weights = {key: float(value["w"]) for key, value in element_profile.items()}
        self.element_radii = element_radii or {}
        super().__init__(
            lmax=max(0, self.twojmax // 2) if lmax is None else int(lmax),
            rcut=rcut,
            normalize_U=normalize_U,
            weights=weights,
            num_threads=num_threads,
        )

    @staticmethod
    def _element_values(batch: StructureBatch, values: dict[Any, float], label: str) -> list[float]:
        if not values:
            return []
        numeric: dict[int, float] = {}
        symbol_numbers = None
        for key, value in values.items():
            if isinstance(key, (int, np.integer)):
                number = int(key)
            elif isinstance(key, str):
                if symbol_numbers is None:
                    from ase.data import atomic_numbers
                    symbol_numbers = atomic_numbers
                try:
                    number = int(symbol_numbers[key])
                except KeyError as exc:
                    raise ValueError(f"unknown chemical symbol in {label}: {key!r}") from exc
            else:
                raise TypeError(f"{label} keys must be atomic numbers or chemical symbols")
            numeric[number] = float(value)
        missing = sorted(set(map(int, batch.numbers)) - set(numeric))
        if missing:
            raise ValueError(f"{label} is missing atomic numbers: {missing}")
        return [numeric[int(number)] for number in batch.numbers]

    def compute(self, value: StructureBatch | Sequence[Any] | Any, control: Any = None) -> DescriptorResult:
        batch = _as_batch(value)
        values = np.asarray(_cpp.compute_rotational_descriptors(
            batch.numbers, batch.positions, batch.cells, batch.pbc, batch.offsets,
            3, self.lmax + 1, self.lmax, self.rcut, 2.0, False, self.normalize_U,
            1.0, self.twojmax, self.diagonal, self.num_threads, control, self.rfac0,
            self._neighbor_weights(batch), self.rmin0, self.rcutfac,
            self._element_values(batch, self.element_radii, "element_radii")), dtype=np.float64)
        self._feature_count = int(values.shape[1])
        return DescriptorResult(values, "atom", batch.ids, batch.offsets.copy(), tuple(f"{self.name}:{i}" for i in range(values.shape[1])), {"backend": "mdescriptor-cpp", "descriptor": self.name})


__all__ = ["EadKernel", "So3Kernel", "So4Kernel", "SnapKernel", "LbispectrumKernel"]
