"""Official PyTorch DPA4C descriptor adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .descriptors import DescriptorResult, StructureBatch, _as_batch, _merge_config, _build_neighbor_graph
from .models import DPA4C_MODEL


_ELEMENT_SYMBOLS = (
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
    "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr",
    "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe",
    "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf",
    "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs",
    "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
)
_ATOMIC_SYMBOLS = {index + 1: symbol for index, symbol in enumerate(_ELEMENT_SYMBOLS)}


def _graph_from_batch(batch: StructureBatch, cutoff: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    src_parts: list[np.ndarray] = []
    dst_parts: list[np.ndarray] = []
    vector_parts: list[np.ndarray] = []
    for frame in range(batch.structures):
        begin, end = int(batch.offsets[frame]), int(batch.offsets[frame + 1])
        offsets, atoms, shifts, displacements, _distance2 = _build_neighbor_graph(
            batch.numbers[begin:end],
            batch.positions[begin:end],
            batch.cells[frame],
            batch.pbc[frame],
            float(cutoff),
        )
        for center in range(end - begin):
            first, last = int(offsets[center]), int(offsets[center + 1])
            if last <= first:
                continue
            local_atoms = np.asarray(atoms[first:last], dtype=np.int64)
            local_shifts = np.asarray(shifts[first:last], dtype=np.int32)
            keep = ~(
                (local_atoms == center)
                & np.all(local_shifts == 0, axis=1)
            )
            local_atoms = local_atoms[keep]
            local_vectors = np.asarray(displacements[first:last], dtype=np.float64)[keep]
            if local_atoms.size == 0:
                continue
            dst_parts.append(np.full(local_atoms.size, begin + center, dtype=np.int64))
            src_parts.append(begin + local_atoms)
            vector_parts.append(local_vectors)
    if not src_parts:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty((0, 3), dtype=np.float64),
        )
    return np.concatenate(src_parts), np.concatenate(dst_parts), np.concatenate(vector_parts, axis=0)


def _frame_index(batch: StructureBatch) -> np.ndarray:
    return np.repeat(
        np.arange(batch.structures, dtype=np.int64),
        np.diff(batch.offsets),
    )


def _validate_charge_spin(value: np.ndarray) -> np.ndarray:
    states = np.asarray(value, dtype=np.float64).reshape(-1, 2)
    if not np.isfinite(states).all() or not np.equal(states, np.floor(states)).all():
        raise ValueError("charge_spin must contain finite integer [charge, multiplicity] pairs")
    if np.any(states[:, 0] < -100) or np.any(states[:, 0] >= 100):
        raise ValueError("charge must be an integer in [-100, 100)")
    if np.any(states[:, 1] < 0) or np.any(states[:, 1] >= 100):
        raise ValueError("multiplicity must be an integer in [0, 100)")
    return np.ascontiguousarray(states, dtype=np.float64)


class Dpa4cCalculator:
    """Compute a frozen DPA4C descriptor from a PyTorch checkpoint.

    The bundled official ``.pt`` checkpoint is used when ``model_path`` is
    omitted. A user-supplied official checkpoint may replace it.
    """

    name = "DPA4C"

    def __init__(
        self,
        model_path: str | Path | None = None,
        config: dict[str, Any] | None = None,
        calibrate: bool = True,
        device: str = "cpu",
        **kwargs: Any,
    ) -> None:
        config = _merge_config(config, kwargs)
        if model_path is None:
            model_path = config.get("model_path", config.get("model_file", config.get("model")))
        if model_path is None:
            model_path = DPA4C_MODEL
        if str(model_path) == "":
            raise ValueError("Dpa4cCalculator model_path cannot be empty")
        self.model_path = str(Path(model_path).expanduser())
        self.calibrate = bool(config.get("calibrate", calibrate))
        self.device_name = str(config.get("device", device))
        self._closed = False
        try:
            import torch
            from .dpa4c.model import load_dpa4c_checkpoint
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise ImportError(
                "Dpa4cCalculator requires the project's PyTorch dependency"
            ) from exc
        if self.device_name.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"requested DPA4C device {self.device_name!r}, but CUDA is unavailable")
        self._torch = torch
        self._native, self._config = load_dpa4c_checkpoint(self.model_path, device=self.device_name)
        self.use_amp = bool(config.get("use_amp", self._config.use_amp))
        self.type_map = self._config.type_map
        self.species = tuple(index + 1 for index, symbol in enumerate(_ELEMENT_SYMBOLS) if symbol in self.type_map)
        self._validate_overrides(config)

    def _validate_overrides(self, config: dict[str, Any]) -> None:
        for key in (
            "rcut", "channels", "lmax", "basis_type", "n_radial", "radial_modes",
            "precision", "use_spin", "add_chg_spin_ebd", "default_chg_spin", "exclude_types",
        ):
            if key not in config:
                continue
            actual = getattr(self._config, key)
            expected = config[key]
            if key == "exclude_types" and expected is not None:
                expected = tuple(tuple(pair) for pair in expected)
            elif isinstance(actual, tuple):
                expected = tuple(expected) if expected is not None else None
            if actual != expected:
                raise ValueError(f"DPA4C config {key!r}={expected!r} disagrees with checkpoint value {actual!r}")

    @property
    def feature_count(self) -> int:
        return int(self._native.feature_count)

    @property
    def descriptor_dim(self) -> int:
        return self.feature_count

    def _type_indices(self, numbers: np.ndarray) -> np.ndarray:
        mapping = {symbol: index for index, symbol in enumerate(self.type_map)}
        result = np.empty(numbers.shape, dtype=np.int64)
        for index, number in enumerate(numbers.tolist()):
            symbol = _ATOMIC_SYMBOLS.get(int(number))
            if symbol not in mapping:
                raise ValueError(f"atomic number {number} ({symbol or 'unknown'}) is absent from the checkpoint type_map")
            result[index] = mapping[symbol]
        return result

    def compute(
        self,
        value: StructureBatch | Sequence[Any] | Any,
        control: Any = None,
        *,
        spin: np.ndarray | None = None,
        charge_spin: np.ndarray | None = None,
    ) -> DescriptorResult:
        del control
        if self._closed:
            raise RuntimeError("DPA4C calculator is closed")
        batch = _as_batch(value)
        spin_values = spin if spin is not None else batch.spins
        if self._config.use_spin is not None:
            if spin_values is None:
                raise ValueError(
                    "a spin-conditioned DPA4C requires StructureBatch.spins or spin=..."
                )
            spin_values = np.ascontiguousarray(spin_values, dtype=np.float64)
            if spin_values.shape != (batch.atoms, 3):
                raise ValueError(f"spin must have shape ({batch.atoms}, 3)")

        charge_values = charge_spin if charge_spin is not None else batch.charge_spin
        if self._config.add_chg_spin_ebd:
            if charge_values is None:
                charge_values = self._config.default_chg_spin
            if charge_values is None:
                raise ValueError(
                    "a charge-conditioned DPA4C requires charge_spin or default_chg_spin"
                )
            charge_values = _validate_charge_spin(charge_values)
            if charge_values.shape[0] == 1 and batch.structures != 1:
                charge_values = np.broadcast_to(
                    charge_values, (batch.structures, 2)
                ).copy()
            if charge_values.shape != (batch.structures, 2):
                raise ValueError(
                    "charge_spin must have one [charge, multiplicity] pair per structure"
                )
        src, dst, vectors = _graph_from_batch(batch, self._config.rcut)
        atype = self._type_indices(batch.numbers)
        frame_index = _frame_index(batch)
        with self._torch.no_grad():
            values = self._native(
                self._torch.as_tensor(vectors, dtype=self._native.compute_dtype, device=self.device_name),
                self._torch.as_tensor(src, dtype=self._torch.long, device=self.device_name),
                self._torch.as_tensor(dst, dtype=self._torch.long, device=self.device_name),
                self._torch.as_tensor(atype, dtype=self._torch.long, device=self.device_name),
                batch.atoms,
                calibrate=self.calibrate,
                spin=None
                if spin_values is None
                else self._torch.as_tensor(
                    spin_values, dtype=self._native.compute_dtype, device=self.device_name
                ),
                charge_spin=None
                if charge_values is None
                else self._torch.as_tensor(
                    charge_values, dtype=self._native.compute_dtype, device=self.device_name
                ),
                frame_index=self._torch.as_tensor(
                    frame_index, dtype=self._torch.long, device=self.device_name
                ),
            )
        values = values.detach().to(device="cpu", dtype=self._torch.float64).numpy()
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
        self._native = None

    def _labels(self) -> tuple[str, ...]:
        labels = [f"dpa4c:scalar,channel={index}" for index in range(self._config.channels)]
        for degree, width in enumerate(self._native.widths[1:], start=1):
            for row in range(width):
                for column in range(row, width):
                    labels.append(f"dpa4c:gram,degree={degree},row={row},column={column}")
        labels.extend(f"dpa4c:bispectrum,index={index}" for index in range(int(self._native.probe_index.numel())))
        labels.extend(f"dpa4c:quartic,index={index}" for index in range(self._native.widths[2] * 2))
        if self._native.spin_channels:
            labels.extend(
                f"dpa4c:spin,vector_gram,index={index}"
                for index in range(int(self._native.spin_vector_index.numel()))
            )
            labels.extend(
                f"dpa4c:spin,quadrupole_gram,index={index}"
                for index in range(int(self._native.spin_quadrupole_index.numel()))
            )
            labels.extend(
                f"dpa4c:spin,cross_degree2,index={index}"
                for index in range(2 * self._native.widths[2])
            )
            labels.extend(
                f"dpa4c:spin,magnitude,index={index}"
                for index in range(self._native.spin_channels)
            )
            labels.extend(
                f"dpa4c:spin,coordination,index={index}"
                for index in range(self._native.spin_channels)
            )
        labels.extend(["dpa4c:divisor=scalar", "dpa4c:divisor=angular"])
        labels.extend(f"dpa4c:type_embedding,index={index}" for index in range(self._config.channels))
        return tuple(labels)

    def _metadata(self) -> dict[str, Any]:
        gram_width = sum(
            width * (width + 1) // 2 for width in self._native.widths[1:]
        )
        scalar_width = self._config.channels
        bispectrum_width = int(self._native.probe_index.numel())
        quartic_width = self._native.widths[2] * 2
        geometry_width = scalar_width + gram_width + bispectrum_width + quartic_width
        spin_width = 0
        if self._native.spin_channels:
            spin_width = self.feature_count - geometry_width - 2 - scalar_width
        offsets = {
            "scalar": (0, scalar_width),
            "gram": (scalar_width, scalar_width + gram_width),
            "bispectrum": (
                scalar_width + gram_width,
                scalar_width + gram_width + bispectrum_width,
            ),
            "quartic": (
                scalar_width + gram_width + bispectrum_width,
                geometry_width,
            ),
        }
        if spin_width:
            offsets["spin"] = (geometry_width, geometry_width + spin_width)
        divisor_start = geometry_width + spin_width
        offsets["divisor"] = (divisor_start, divisor_start + 2)
        offsets["type_embedding"] = (
            divisor_start + 2,
            divisor_start + 2 + scalar_width,
        )
        return {
            "backend": "mdescriptor-torch",
            "descriptor": self.name,
            "model_path": self.model_path,
            "type_map": self.type_map,
            "rcut": self._config.rcut,
            "channels": self._config.channels,
            "lmax": self._config.lmax,
            "basis_type": self._config.basis_type,
            "n_radial": self._config.n_radial,
            "radial_modes": self._config.radial_modes,
            "use_amp": self.use_amp,
            "precision": self._config.precision,
            "use_spin": self._config.use_spin,
            "add_chg_spin_ebd": self._config.add_chg_spin_ebd,
            "calibrated": self.calibrate,
            "device": self.device_name,
            "feature_count": self.feature_count,
            "block_offsets": offsets,
        }


DPA4C = Dpa4cCalculator
DPA4CCalculator = Dpa4cCalculator

__all__ = ["DPA4C", "DPA4CCalculator", "Dpa4cCalculator"]
