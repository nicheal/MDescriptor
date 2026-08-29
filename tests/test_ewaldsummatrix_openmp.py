"""Numerical stability and timing checks for the native Ewald OpenMP path."""

from __future__ import annotations

import time

import numpy as np
import pytest

from tests._public import EwaldSumMatrix, ExecutionOptions, StructureBatch


def _batch(count: int = 64) -> StructureBatch:
    """Build one non-orthogonal periodic structure with useful Ewald work."""

    indices = np.arange(count, dtype=np.float64)
    positions = np.column_stack(
        (
            np.mod(0.37 * indices + 0.11, 11.5),
            np.mod(0.61 * indices + 0.23, 12.5),
            np.mod(0.83 * indices + 0.31, 13.5),
        )
    )
    cell = np.asarray(
        [[14.0, 0.2, 0.1], [0.3, 15.0, 0.4], [0.1, 0.2, 16.0]],
        dtype=np.float64,
    )
    return StructureBatch(
        np.resize(np.asarray([1, 6, 8, 14], dtype=np.int32), count),
        positions,
        cell[None, :, :],
        np.ones((1, 3), dtype=np.int32),
        np.asarray([0, count], dtype=np.int64),
        ("ewaldsummatrix-openmp",),
    )


def _descriptor(num_threads: int) -> EwaldSumMatrix:
    return EwaldSumMatrix(
        n_atoms_max=64,
        permutation="none",
        accuracy=1e-5,
        w=1.0,
        r_cut=6.0,
        g_cut=3.0,
        a=0.3,
        execution=ExecutionOptions(num_threads=num_threads),
    )


def test_ewaldsummatrix_openmp_matches_single_thread_bitwise() -> None:
    batch = _batch()
    serial = _descriptor(1)
    parallel = _descriptor(4)
    try:
        serial_result = serial.compute(batch)
        parallel_result = parallel.compute(batch)
        np.testing.assert_array_equal(parallel_result.values, serial_result.values)
        assert np.isfinite(parallel_result.values).all()
    finally:
        serial.close()
        parallel.close()


def _median_runtime(descriptor: EwaldSumMatrix, batch: StructureBatch) -> float:
    descriptor.compute(batch)
    elapsed = []
    for _ in range(3):
        started = time.perf_counter()
        descriptor.compute(batch)
        elapsed.append(time.perf_counter() - started)
    return float(np.median(elapsed))


@pytest.mark.timing
def test_ewaldsummatrix_openmp_small_batch_timing_smoke() -> None:
    batch = _batch()
    serial = _descriptor(1)
    parallel = _descriptor(4)
    try:
        serial_seconds = _median_runtime(serial, batch)
        parallel_seconds = _median_runtime(parallel, batch)
        assert serial_seconds > 0.0
        assert parallel_seconds > 0.0
        print(
            "EwaldSumMatrix small batch: "
            f"serial={serial_seconds:.6f}s, "
            f"threads=4={parallel_seconds:.6f}s, "
            f"speedup={serial_seconds / parallel_seconds:.2f}x"
        )
    finally:
        serial.close()
        parallel.close()
