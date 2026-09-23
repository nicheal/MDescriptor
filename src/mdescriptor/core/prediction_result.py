"""Immutable energy, per-atom energy, and force predictions."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .json_value import freeze_json


@dataclass(frozen=True, slots=True)
class PredictionResult:
    """A detached prediction snapshot in eV and eV/angstrom."""

    energy: np.ndarray
    atom_energy: np.ndarray
    forces: np.ndarray
    structure_ids: tuple[str, ...]
    offsets: np.ndarray
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        ids = tuple(str(value) for value in self.structure_ids)
        offsets = np.asarray(self.offsets)
        if offsets.ndim != 1 or offsets.dtype.kind not in "iu":
            raise ValueError("prediction offsets must be a one-dimensional integer array")
        offsets = np.array(offsets, dtype=np.int64, order="C", copy=True)
        if offsets.shape != (len(ids) + 1,) or offsets[0] != 0 or np.any(np.diff(offsets) < 0):
            raise ValueError("prediction offsets must delimit every structure")

        energy = _snapshot(self.energy, (len(ids),), "energy")
        atom_count = int(offsets[-1])
        atom_energy = _snapshot(self.atom_energy, (atom_count,), "atom_energy")
        forces = _snapshot(self.forces, (atom_count, 3), "forces")
        offsets.setflags(write=False)

        object.__setattr__(self, "energy", energy)
        object.__setattr__(self, "atom_energy", atom_energy)
        object.__setattr__(self, "forces", forces)
        object.__setattr__(self, "structure_ids", ids)
        object.__setattr__(self, "offsets", offsets)
        object.__setattr__(self, "metadata", freeze_json(dict(self.metadata)))


def _snapshot(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    result = np.array(value, dtype=np.float64, order="C", copy=True)
    if result.shape != shape:
        raise ValueError(f"prediction {name} must have shape {shape}")
    if not np.isfinite(result).all():
        raise ValueError(f"prediction {name} must contain finite values")
    result.setflags(write=False)
    return result


__all__ = ["PredictionResult"]
