"""Canonical, validated structure input boundary."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np


@dataclass(frozen=True)
class StructureBatch:
    """A contiguous snapshot of periodic or isolated structures.

    Fully periodic structures carry a nonsingular cell and ``pbc=(1, 1, 1)``.
    Isolated structures use ``pbc=(0, 0, 0)`` and may carry ASE's zero cell.
    Mixed periodicity is intentionally rejected until the native kernels have
    an explicit partial-periodicity contract.
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
        spins = None if self.spins is None else np.ascontiguousarray(self.spins, dtype=np.float64)
        charge_spin = None if self.charge_spin is None else np.ascontiguousarray(self.charge_spin, dtype=np.float64)

        if numbers.ndim != 1 or np.any(numbers <= 0):
            raise ValueError("numbers must be a one-dimensional array of positive atomic numbers")
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
        if np.any((pbc != 0) & (pbc != 1)):
            raise ValueError("pbc must contain only 0 or 1")
        for index, matrix in enumerate(cells):
            flags = pbc[index]
            if bool(np.all(flags == 1)):
                if abs(float(np.linalg.det(matrix))) < 1e-14:
                    raise ValueError("periodic cells must be nonsingular")
            elif not bool(np.all(flags == 0)):
                raise ValueError("mixed periodicity is not supported; use all-zero or all-one pbc")
        if spins is not None and (spins.ndim != 2 or spins.shape != (len(numbers), 3)):
            raise ValueError("spins must have shape (total_atoms, 3)")
        if charge_spin is not None and (charge_spin.ndim != 2 or charge_spin.shape != (len(ids), 2)):
            raise ValueError("charge_spin must have shape (structures, 2)")
        if spins is not None and not np.isfinite(spins).all():
            raise ValueError("spins must be finite")
        if charge_spin is not None and not np.isfinite(charge_spin).all():
            raise ValueError("charge_spin must be finite")

        for name, value in {
            "numbers": numbers,
            "positions": positions,
            "cells": cells,
            "pbc": pbc,
            "offsets": offsets,
            "ids": ids,
            "spins": spins,
            "charge_spin": charge_spin,
        }.items():
            object.__setattr__(self, name, value)

    @property
    def structures(self) -> int:
        return len(self.ids)

    @property
    def atoms(self) -> int:
        return len(self.numbers)

    @classmethod
    def from_ase(cls, structures: Sequence[Any] | Any, ids: Sequence[str] | None = None) -> StructureBatch:
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
                generated_ids.append(f"{Path(source).resolve()}#{frame}" if source else str(index))

        return cls(
            np.concatenate(number_parts) if number_parts else np.empty(0, dtype=np.int32),
            np.concatenate(position_parts, axis=0) if position_parts else np.empty((0, 3), dtype=np.float64),
            np.stack(cell_parts) if cell_parts else np.empty((0, 3, 3), dtype=np.float64),
            np.stack(pbc_parts) if pbc_parts else np.empty((0, 3), dtype=np.int32),
            np.asarray(offsets),
            tuple(generated_ids),
            np.concatenate(spin_parts, axis=0) if have_spins else None,
            np.stack(frame_charge_spin) if have_charge_spin else None,
        )


batch_from_ase = StructureBatch.from_ase


def coerce_batch(value: StructureInput) -> StructureBatch:
    """Return an existing batch unchanged or pack ASE structures once."""

    if isinstance(value, StructureBatch):
        return value
    return StructureBatch.from_ase(value)


StructureInput: TypeAlias = StructureBatch | Sequence[Any] | Any

__all__ = ["StructureBatch", "StructureInput", "batch_from_ase", "coerce_batch"]
