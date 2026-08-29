"""OpenMP contract and small-batch timing checks for the local MBTR path."""

from __future__ import annotations

import time

import numpy as np
import pytest
from ase import Atoms

from tests._public import LMBTR, ExecutionOptions, StructureBatch


def _batch() -> StructureBatch:
    """Build a small periodic batch with enough local work for a timing sample."""

    cell = np.diag([28.0, 28.0, 28.0])
    positions = np.asarray(
        [
            [10.0 + 1.5 * x, 9.0 + 1.5 * y, 9.0 + 1.5 * z]
            for z in range(2)
            for y in range(4)
            for x in range(4)
        ],
        dtype=np.float64,
    )
    numbers = np.tile(np.asarray([1, 6, 8, 14], dtype=np.int32), 8)
    systems = [
        Atoms(
            numbers=numbers,
            positions=positions + shift,
            cell=cell,
            pbc=True,
        )
        for shift in (
            np.asarray([0.0, 0.0, 0.0]),
            np.asarray([0.2, -0.1, 0.15]),
            np.asarray([-0.15, 0.25, -0.2]),
            np.asarray([0.35, 0.1, -0.1]),
        )
    ]
    return StructureBatch.from_ase(systems)


@pytest.fixture(scope="module")
def lmbtr_batch() -> StructureBatch:
    return _batch()


def _single_structure(batch: StructureBatch, index: int = 0) -> StructureBatch:
    begin = int(batch.offsets[index])
    end = int(batch.offsets[index + 1])
    return StructureBatch(
        batch.numbers[begin:end],
        batch.positions[begin:end],
        batch.cells[index : index + 1],
        batch.pbc[index : index + 1],
        np.asarray([0, end - begin], dtype=np.int64),
        (batch.ids[index],),
    )


@pytest.fixture(scope="module")
def lmbtr_single_batch(lmbtr_batch: StructureBatch) -> StructureBatch:
    return _single_structure(lmbtr_batch)


def _parameters(function: str) -> dict[str, object]:
    grid = (
        {"min": 0.0, "max": 5.0, "n": 32, "sigma": 0.1}
        if function == "distance"
        else {"min": 0.0, "max": 180.0, "n": 32, "sigma": 1.5}
    )
    return {
        "species": [1, 6, 8, 14],
        "geometry": {"function": function},
        "grid": grid,
        "weighting": {"function": "smooth_cutoff", "r_cut": 5.0, "sharpness": 2.0},
    }


def _compute(
    batch: StructureBatch,
    function: str,
    num_threads: int,
):
    descriptor = LMBTR(
        **_parameters(function),
        execution=ExecutionOptions(num_threads=num_threads),
    )
    try:
        return descriptor.compute(batch)
    finally:
        descriptor.close()


@pytest.mark.parametrize("function", ["distance", "angle"])
@pytest.mark.parametrize("single_structure", [False, True], ids=["batch", "single"])
def test_lmbtr_openmp_matches_serial_output(
    lmbtr_batch: StructureBatch,
    lmbtr_single_batch: StructureBatch,
    function: str,
    single_structure: bool,
) -> None:
    batch = lmbtr_single_batch if single_structure else lmbtr_batch
    serial = _compute(batch, function, num_threads=1)
    parallel = _compute(batch, function, num_threads=2)

    assert serial.level == parallel.level == "atom"
    assert serial.labels == parallel.labels
    np.testing.assert_array_equal(serial.samples, parallel.samples)
    np.testing.assert_array_equal(serial.row_offsets, parallel.row_offsets)
    np.testing.assert_allclose(
        serial.values,
        parallel.values,
        rtol=1e-9,
        atol=1e-11,
    )


def _median_runtime(
    descriptor: LMBTR,
    batch: StructureBatch,
    *,
    warmup: int = 2,
    repeat: int = 5,
) -> float:
    for _ in range(warmup):
        descriptor.compute(batch)
    elapsed = []
    for _ in range(repeat):
        started = time.perf_counter()
        descriptor.compute(batch)
        elapsed.append(time.perf_counter() - started)
    return float(np.median(elapsed))


@pytest.mark.timing
def test_lmbtr_openmp_small_batch_timing_smoke(
    lmbtr_batch: StructureBatch,
    record_property,
) -> None:
    parameters = _parameters("angle")
    serial = LMBTR(
        **parameters,
        execution=ExecutionOptions(num_threads=1),
    )
    parallel = LMBTR(
        **parameters,
        execution=ExecutionOptions(num_threads=2),
    )
    try:
        serial_seconds = _median_runtime(serial, lmbtr_batch)
        parallel_seconds = _median_runtime(parallel, lmbtr_batch)
    finally:
        serial.close()
        parallel.close()

    assert np.isfinite(serial_seconds) and serial_seconds > 0.0
    assert np.isfinite(parallel_seconds) and parallel_seconds > 0.0
    record_property("serial_seconds", serial_seconds)
    record_property("two_thread_seconds", parallel_seconds)
    record_property("speedup", serial_seconds / parallel_seconds)
