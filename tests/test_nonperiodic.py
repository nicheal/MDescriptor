"""Contracts for isolated structures with a zero cell."""

from __future__ import annotations

import numpy as np
import pytest

from mdescriptor import StructureBatch, get_descriptor
from mdescriptor._native import build_neighbor_graph


def _isolated_batch(shift: float = 0.0) -> StructureBatch:
    return StructureBatch(
        np.asarray([6, 6], dtype=np.int32),
        np.asarray([[shift, 0.0, 0.0], [shift + 1.2, 0.0, 0.0]], dtype=np.float64),
        np.zeros((1, 3, 3), dtype=np.float64),
        np.zeros((1, 3), dtype=np.int32),
        np.asarray([0, 2], dtype=np.int64),
        (f"isolated-{shift}",),
    )


def test_structure_batch_accepts_zero_cell_isolated_structures() -> None:
    batch = _isolated_batch()
    assert np.array_equal(batch.pbc, np.zeros((1, 3), dtype=np.int32))
    assert np.allclose(batch.cells, 0.0)


def test_nonperiodic_neighbor_graph_has_no_periodic_images() -> None:
    batch = _isolated_batch()
    offsets, atoms, shifts, _displacements, _distance2 = build_neighbor_graph(
        batch.numbers[0:2],
        batch.positions[0:2],
        batch.cells[0],
        batch.pbc[0],
        2.0,
    )
    assert offsets.tolist() == [0, 2, 4]
    assert np.all(np.asarray(shifts) == 0)
    assert sorted(np.asarray(atoms).tolist()) == [0, 0, 1, 1]


def test_nonperiodic_soap_is_translation_invariant() -> None:
    descriptor = get_descriptor("SOAP")(
        species=[6],
        r_cut=3.5,
        n_max=2,
        l_max=2,
        sigma=0.5,
        average="off",
    )
    first = descriptor.compute(_isolated_batch(0.0)).values
    second = descriptor.compute(_isolated_batch(10.0)).values
    assert np.allclose(first, second, rtol=1e-12, atol=1e-12)
    descriptor.close()


def test_batch_can_mix_isolated_and_fully_periodic_structures() -> None:
    isolated = _isolated_batch()
    mixed = StructureBatch(
        np.concatenate((isolated.numbers, isolated.numbers)),
        np.concatenate((isolated.positions, isolated.positions + 5.0)),
        np.concatenate((np.eye(3, dtype=np.float64)[None] * 8.0, isolated.cells)),
        np.concatenate((np.ones((1, 3), dtype=np.int32), isolated.pbc)),
        np.asarray([0, 2, 4], dtype=np.int64),
        ("periodic", "isolated"),
    )
    descriptor = get_descriptor("SOAP")(
        species=[6],
        r_cut=3.5,
        n_max=2,
        l_max=2,
        sigma=0.5,
        average="off",
    )
    try:
        result = descriptor.compute(mixed)
        assert result.values.shape[0] == 4
        assert result.row_offsets.tolist() == [0, 2, 4]
    finally:
        descriptor.close()


def test_partial_periodicity_is_rejected_at_both_boundaries() -> None:
    batch = _isolated_batch()
    with pytest.raises(ValueError, match="mixed periodicity"):
        StructureBatch(
            batch.numbers,
            batch.positions,
            batch.cells,
            np.asarray([[1, 0, 0]], dtype=np.int32),
            batch.offsets,
            batch.ids,
        )
    with pytest.raises(ValueError, match="mixed periodicity"):
        build_neighbor_graph(
            batch.numbers,
            batch.positions,
            batch.cells[0],
            np.asarray([1, 0, 0], dtype=np.int32),
            2.0,
        )
