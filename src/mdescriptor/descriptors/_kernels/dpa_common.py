"""Shared dispatch for the native DPA kernel adapters."""

from __future__ import annotations

from typing import Any

import numpy as np

from ...core.input import StructureBatch
from ..model_backed.graph import _ATOMIC_SYMBOLS


def compute_native_batch(
    calculator: Any,
    type_mapper: Any,
    batch: StructureBatch,
    control: Any,
) -> np.ndarray:
    """Initialize progress, map a validated batch, and call either DPA calculator."""

    if control is not None:
        reset = getattr(control, "reset", None)
        if callable(reset):
            reset(batch.structures)

    symbols: list[str] = []
    for number in batch.numbers.tolist():
        try:
            symbols.append(_ATOMIC_SYMBOLS[int(number)])
        except KeyError as exc:
            raise ValueError(
                f"atomic number {number} is absent from the checkpoint type_map"
            ) from exc
    try:
        type_indices = type_mapper.symbols_to_atype(symbols).astype(np.int32, copy=False)
    except KeyError as exc:
        raise ValueError(
            f"element {exc.args[0]!r} is absent from the checkpoint type_map"
        ) from exc
    return calculator.compute(
        batch.numbers,
        batch.positions,
        batch.cells,
        batch.pbc,
        batch.offsets,
        type_indices,
        control,
    )


__all__ = ["compute_native_batch"]
