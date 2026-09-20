"""Regression coverage for batch neighbor graph exceptions and ordering."""

from __future__ import annotations

import numpy as np
import pytest

from tests._public import ExecutionOptions, NeighborList, SortedDistances, StructureBatch


def _two_structure_batch() -> StructureBatch:
    return StructureBatch(
        numbers=np.asarray([1, 8, 1, 8], dtype=np.int32),
        positions=np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
             [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
            dtype=np.float64,
        ),
        cells=np.zeros((2, 3, 3), dtype=np.float64),
        pbc=np.zeros((2, 3), dtype=np.int32),
        offsets=np.asarray([0, 2, 4], dtype=np.int64),
        ids=("first", "second"),
    )


@pytest.mark.parametrize("num_threads", [1, 2])
def test_multistructure_neighbor_bounds_raise_value_error(num_threads: int) -> None:
    descriptor = NeighborList(
        cutoff=1e-12, execution=ExecutionOptions(num_threads=num_threads),
    )
    try:
        with pytest.raises(ValueError, match="cell-grid dimension exceeds integer range"):
            descriptor.compute(_two_structure_batch())
    finally:
        descriptor.close()


def test_multistructure_sorted_distances_keep_numeric_order() -> None:
    batch = _two_structure_batch()
    serial = SortedDistances(
        species=[1, 8], cutoff=1.5, max_neighbors=2,
        execution=ExecutionOptions(num_threads=1),
    )
    threaded = SortedDistances(
        species=[1, 8], cutoff=1.5, max_neighbors=2,
        execution=ExecutionOptions(num_threads=2),
    )
    try:
        serial_result = serial.compute(batch)
        threaded_result = threaded.compute(batch)
        expected = np.asarray(
            [[0.0, 0.0, 1.0, 1.5],
             [1.0, 1.5, 0.0, 0.0],
             [0.0, 0.0, 1.0, 1.5],
             [1.0, 1.5, 0.0, 0.0]],
        )
        np.testing.assert_allclose(serial_result.values, expected)
        np.testing.assert_allclose(threaded_result.values, serial_result.values)
        np.testing.assert_array_equal(threaded_result.row_offsets, serial_result.row_offsets)
    finally:
        serial.close()
        threaded.close()
