"""Canonical, validated structure input boundary."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np

from .errors import _InputValidationError

# Per-frame fields handed to the shared packer: ``(numbers, positions, cell,
# pbc, id, spins, charge_spin)``.  ``None`` marks an absent optional spin field.
_FrameFields = tuple[Any, Any, Any, Any, Any, Any, Any]


@dataclass(frozen=True)
class StructureBatch:
    """An owned, read-only snapshot of periodic or isolated structures.

    Fully periodic structures carry a nonsingular cell and ``pbc=(1, 1, 1)``.
    Isolated structures use ``pbc=(0, 0, 0)`` and may carry ASE's zero cell.
    Partial periodicity within one frame is rejected until the native kernels
    have an explicit partial-periodicity contract. A batch may contain both
    isolated and fully periodic frames.
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
        # ``StructureBatch`` is a value object at the public seam.  Always
        # copy before validation so caller-owned arrays cannot mutate a batch
        # while a worker or native kernel is consuming it.  Integer arrays are
        # checked before narrowing; NumPy's direct cast would silently turn
        # values such as 1.5 into 1.
        numbers = _integer_snapshot(self.numbers, np.int32, "numbers")
        positions = _array_snapshot(self.positions, np.float64, "positions")
        cells = _array_snapshot(self.cells, np.float64, "cells")
        pbc = _integer_snapshot(self.pbc, np.int32, "pbc", allow_bool=True)
        offsets = _integer_snapshot(self.offsets, np.int64, "offsets")
        ids = tuple(str(value) for value in self.ids)
        spins = (
            None
            if self.spins is None
            else _array_snapshot(self.spins, np.float64, "spins")
        )
        charge_spin = (
            None
            if self.charge_spin is None
            else _array_snapshot(self.charge_spin, np.float64, "charge_spin")
        )

        if numbers.ndim != 1 or np.any(numbers <= 0):
            raise _input_error(
                "numbers must be a one-dimensional array of positive atomic numbers",
                "numbers",
            )
        if positions.shape != (len(numbers), 3):
            raise _input_error("positions must have shape (total_atoms, 3)", "positions")
        if cells.ndim != 3 or cells.shape[1:] != (3, 3):
            raise _input_error("cells must have shape (structures, 3, 3)", "cells")
        if pbc.ndim != 2 or pbc.shape[1:] != (3,):
            raise _input_error("pbc must have shape (structures, 3)", "pbc")
        if offsets.ndim != 1 or len(offsets) != len(ids) + 1:
            raise _input_error(
                "offsets must have one entry per structure plus a sentinel", "offsets"
            )
        if len(cells) != len(ids) or len(pbc) != len(ids):
            raise _input_error(
                "structure arrays and ids have inconsistent lengths", "structures"
            )
        if len(offsets) and (offsets[0] != 0 or offsets[-1] != len(numbers)):
            raise _input_error("offsets must start at zero and end at total_atoms", "offsets")
        if np.any(offsets[1:] < offsets[:-1]):
            raise _input_error("offsets must be monotonic", "offsets")
        if not np.isfinite(positions).all():
            raise _input_error("positions and cells must be finite", "positions")
        if not np.isfinite(cells).all():
            raise _input_error("positions and cells must be finite", "cells")
        if np.any((pbc != 0) & (pbc != 1)):
            raise _input_error("pbc must contain only 0 or 1", "pbc")
        for index, matrix in enumerate(cells):
            flags = pbc[index]
            if bool(np.all(flags == 1)):
                if abs(float(np.linalg.det(matrix))) < 1e-14:
                    raise _input_error("periodic cells must be nonsingular", "cells")
            elif not bool(np.all(flags == 0)):
                raise _input_error(
                    "mixed periodicity is not supported; use all-zero or all-one pbc",
                    "pbc",
                )
        if spins is not None and (spins.ndim != 2 or spins.shape != (len(numbers), 3)):
            raise _input_error("spins must have shape (total_atoms, 3)", "spins")
        if charge_spin is not None and (charge_spin.ndim != 2 or charge_spin.shape != (len(ids), 2)):
            raise _input_error("charge_spin must have shape (structures, 2)", "charge_spin")
        if spins is not None and not np.isfinite(spins).all():
            raise _input_error("spins must be finite", "spins")
        if charge_spin is not None and not np.isfinite(charge_spin).all():
            raise _input_error("charge_spin must be finite", "charge_spin")

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
            if isinstance(value, np.ndarray):
                value.setflags(write=False)
            object.__setattr__(self, name, value)

    @property
    def structures(self) -> int:
        return len(self.ids)

    @property
    def atoms(self) -> int:
        return len(self.numbers)

    def _slice_view(self, start: int, stop: int) -> StructureBatch:
        """Return a read-only view for a complete structure range.

        This private path is only for already validated batches. Public
        construction keeps its copy-and-validate boundary; block slicing
        reuses the validated buffers and owns only the rebased offsets.
        """

        if start < 0 or stop < start or stop > self.structures:
            raise ValueError("structure slice is outside the batch")
        atom_start = int(self.offsets[start])
        atom_stop = int(self.offsets[stop])
        numbers = self.numbers[atom_start:atom_stop]
        positions = self.positions[atom_start:atom_stop]
        cells = self.cells[start:stop]
        pbc = self.pbc[start:stop]
        offsets = np.asarray(
            self.offsets[start : stop + 1] - atom_start,
            dtype=np.int64,
            order="C",
        )
        spins = None if self.spins is None else self.spins[atom_start:atom_stop]
        charge_spin = None if self.charge_spin is None else self.charge_spin[start:stop]
        for value in (numbers, positions, cells, pbc, offsets, spins, charge_spin):
            if value is not None:
                value.setflags(write=False)

        # Block slicing has always returned the base value object.  Keep that
        # contract instead of manufacturing a potentially incomplete subclass.
        view = object.__new__(StructureBatch)
        for name, field in {
            "numbers": numbers,
            "positions": positions,
            "cells": cells,
            "pbc": pbc,
            "offsets": offsets,
            "ids": self.ids[start:stop],
            "spins": spins,
            "charge_spin": charge_spin,
        }.items():
            object.__setattr__(view, name, field)
        return view

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
            raise _input_error("ids must have one entry per structure", "ids")

        def read_ase_frame(atoms: Any, index: int) -> _FrameFields:
            if not isinstance(atoms, Atoms):
                raise TypeError("structures must contain ASE Atoms objects")
            atom_spin = atoms.arrays.get("spin", atoms.arrays.get("spins"))
            if atom_spin is None:
                atom_spin = atoms.info.get("spin", atoms.info.get("spins"))
            spin = (
                None
                if atom_spin is None
                else np.asarray(atom_spin, dtype=np.float64)
            )
            frame_state = atoms.info.get("charge_spin")
            if frame_state is None:
                charge = atoms.info.get("charge")
                multiplicity = atoms.info.get("spin_multiplicity")
                if charge is not None or multiplicity is not None:
                    frame_state = (0.0 if charge is None else charge, 0.0 if multiplicity is None else multiplicity)
            charge_spin = (
                None
                if frame_state is None
                else np.asarray(frame_state, dtype=np.float64)
            )
            if ids is not None:
                identifier: Any = str(ids[index])
            else:
                source = atoms.info.get("source_path", atoms.info.get("_source_path"))
                frame = atoms.info.get("frame", index)
                identifier = f"{Path(source).resolve()}#{frame}" if source else str(index)
            return (
                np.asarray(atoms.get_atomic_numbers(), dtype=np.int32),
                np.asarray(atoms.get_positions(), dtype=np.float64),
                np.asarray(atoms.cell.array, dtype=np.float64),
                np.asarray(atoms.get_pbc(), dtype=np.int32),
                identifier,
                spin,
                charge_spin,
            )

        return _pack_frames(structures, read_ase_frame)

    @classmethod
    def from_frames(cls, frames: Iterable[Any] | Any) -> StructureBatch:
        """Pack GUI-style frame records into one validated batch.

        A frame may be a mapping or an object exposing ``numbers``,
        ``positions``, ``cell``, ``pbc`` and ``id``.  ``spins`` and
        ``charge_spin`` are optional and follow the corresponding batch
        fields.  The cumulative ``offsets`` sentinel array is generated here
        so callers do not need to flatten structures themselves.
        """

        if isinstance(frames, Mapping) or not isinstance(frames, Iterable):
            frame_values = [frames]
        else:
            frame_values = list(frames)

        def read_frame_frame(frame: Any, index: int) -> _FrameFields:
            numbers = _frame_array(frame, "numbers", index=index)
            assert numbers is not None
            try:
                len(numbers)
            except TypeError as exc:
                raise _input_error(
                    f"frame {index} numbers must be one-dimensional", "numbers"
                ) from exc
            spin = _frame_array(
                frame, "spins", aliases=("spin",), index=index, default=None
            )
            charge_spin = _frame_array(frame, "charge_spin", index=index, default=None)
            return (
                numbers,
                _frame_array(frame, "positions", index=index),
                _frame_array(frame, "cell", aliases=("cells",), index=index),
                _frame_array(frame, "pbc", index=index),
                _frame_field(frame, "id", index=index),
                spin,
                charge_spin,
            )

        return _pack_frames(frame_values, read_frame_frame)


def _pack_frames(
    frames: Iterable[Any],
    read_frame: Callable[[Any, int], _FrameFields],
) -> StructureBatch:
    """Assemble per-frame fields into one validated batch.

    ``read_frame`` extracts one frame's fields; this builder owns the shared
    concatenation, cumulative atom offsets, absent-spin placeholders, and
    empty-batch fallbacks so ASE and GUI frame inputs pack identically.
    """

    number_parts: list[np.ndarray] = []
    position_parts: list[np.ndarray] = []
    cell_parts: list[np.ndarray] = []
    pbc_parts: list[np.ndarray] = []
    spin_parts: list[np.ndarray] = []
    frame_charge_spin: list[np.ndarray] = []
    have_spins = False
    have_charge_spin = False
    offsets = [0]
    generated_ids: list[Any] = []

    for index, frame in enumerate(frames):
        numbers, positions, cell, pbc, identifier, spin, charge_spin = read_frame(
            frame, index
        )
        number_parts.append(np.asarray(numbers))
        position_parts.append(np.asarray(positions))
        cell_parts.append(np.asarray(cell))
        pbc_parts.append(np.asarray(pbc))
        generated_ids.append(identifier)
        if spin is None:
            spin_parts.append(np.zeros((len(numbers), 3), dtype=np.float64))
        else:
            have_spins = True
            spin_parts.append(np.asarray(spin))
        if charge_spin is None:
            frame_charge_spin.append(np.zeros(2, dtype=np.float64))
        else:
            have_charge_spin = True
            frame_charge_spin.append(np.asarray(charge_spin))
        offsets.append(offsets[-1] + len(numbers))

    return StructureBatch(
        _join_frame_arrays(number_parts, "numbers")
        if number_parts
        else np.empty(0, dtype=np.int32),
        _join_frame_arrays(position_parts, "positions")
        if position_parts
        else np.empty((0, 3), dtype=np.float64),
        _join_frame_arrays(cell_parts, "cells", stack=True)
        if cell_parts
        else np.empty((0, 3, 3), dtype=np.float64),
        _join_frame_arrays(pbc_parts, "pbc", stack=True)
        if pbc_parts
        else np.empty((0, 3), dtype=np.int32),
        np.asarray(offsets),
        tuple(generated_ids),
        _join_frame_arrays(spin_parts, "spins") if have_spins else None,
        _join_frame_arrays(frame_charge_spin, "charge_spin", stack=True)
        if have_charge_spin
        else None,
    )


def coerce_batch(value: StructureInput) -> StructureBatch:
    """Return an existing batch unchanged or pack ASE structures once."""

    if isinstance(value, StructureBatch):
        return value
    return StructureBatch.from_ase(value)


# Honest alias: ``compute`` accepts an already-built ``StructureBatch`` or any
# ASE structure/sequence of structures that ``StructureBatch.from_ase`` can
# adapt; anything else is rejected at the input boundary.
StructureInput: TypeAlias = Any


def _array_snapshot(value: Any, dtype: Any, name: str) -> np.ndarray:
    try:
        array = np.array(value, dtype=dtype, order="C", copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise _input_error(f"{name} must be a numeric array", name) from exc
    if not np.isfinite(array).all():
        raise _input_error(f"{name} must be finite", name)
    return array


def _frame_field(
    frame: Any,
    name: str,
    *,
    index: int | None = None,
    aliases: Sequence[str] = (),
    default: Any = ...,
) -> Any:
    """Read one required or optional field from a frame record."""

    names = (name, *aliases)
    if isinstance(frame, Mapping):
        for candidate in names:
            if candidate in frame:
                return frame[candidate]
    else:
        for candidate in names:
            try:
                return getattr(frame, candidate)
            except AttributeError:
                continue
    if default is not ...:
        return default
    location = "" if index is None else f" {index}"
    field = {"cell": "cells", "id": "ids"}.get(name, name)
    raise _input_error(
        f"frame{location} is missing required field {name!r}", field
    )


def _frame_array(
    frame: Any,
    name: str,
    *,
    index: int,
    aliases: Sequence[str] = (),
    default: Any = ...,
) -> np.ndarray | None:
    value = _frame_field(
        frame, name, index=index, aliases=aliases, default=default
    )
    if value is None and default is None:
        return None
    field = "cells" if name == "cell" else name
    try:
        return np.asarray(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise _input_error(f"frame {index} {field} must be an array", field) from exc


def _join_frame_arrays(
    parts: list[np.ndarray], name: str, *, stack: bool = False
) -> np.ndarray:
    try:
        return np.stack(parts) if stack else np.concatenate(parts, axis=0)
    except (TypeError, ValueError) as exc:
        raise _input_error(str(exc), name) from exc


def _integer_snapshot(
    value: Any,
    dtype: Any,
    name: str,
    *,
    allow_bool: bool = False,
) -> np.ndarray:
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise _input_error(f"{name} must be an integer array", name) from exc
    if raw.dtype.kind == "b":
        if not allow_bool:
            raise _input_error(f"{name} must contain integers, not booleans", name)
    elif raw.dtype.kind in "iu":
        pass
    elif raw.dtype.kind == "f":
        if not np.isfinite(raw).all() or not np.equal(raw, np.trunc(raw)).all():
            raise _input_error(f"{name} must contain finite integers", name)
    else:
        raise _input_error(f"{name} must contain integers", name)

    limits = np.iinfo(dtype)
    if raw.size and (np.any(raw < limits.min) or np.any(raw > limits.max)):
        raise _input_error(f"{name} contains values outside {dtype} range", name)
    try:
        return np.array(raw, dtype=dtype, order="C", copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise _input_error(f"{name} must be an integer array", name) from exc


def _input_error(message: str, field: str) -> _InputValidationError:
    return _InputValidationError(message, ("input", field))

__all__ = ["StructureBatch", "StructureInput", "coerce_batch"]
