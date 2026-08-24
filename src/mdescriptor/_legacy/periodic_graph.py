"""Shared periodic graph and atomic-number adapters for model-backed descriptors."""

from __future__ import annotations

import numpy as np

from .core import StructureBatch, _build_neighbor_graph


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


def graph_from_batch(batch: StructureBatch, cutoff: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
            keep = ~((local_atoms == center) & np.all(local_shifts == 0, axis=1))
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


def frame_index(batch: StructureBatch) -> np.ndarray:
    return np.repeat(np.arange(batch.structures, dtype=np.int64), np.diff(batch.offsets))


def validate_charge_spin(value: np.ndarray) -> np.ndarray:
    states = np.asarray(value, dtype=np.float64).reshape(-1, 2)
    if not np.isfinite(states).all() or not np.equal(states, np.floor(states)).all():
        raise ValueError("charge_spin must contain finite integer [charge, multiplicity] pairs")
    if np.any(states[:, 0] < -100) or np.any(states[:, 0] >= 100):
        raise ValueError("charge must be an integer in [-100, 100)")
    if np.any(states[:, 1] < 0) or np.any(states[:, 1] >= 100):
        raise ValueError("multiplicity must be an integer in [0, 100)")
    return np.ascontiguousarray(states, dtype=np.float64)


__all__ = [
    "_ATOMIC_SYMBOLS",
    "_ELEMENT_SYMBOLS",
    "frame_index",
    "graph_from_batch",
    "validate_charge_spin",
]
