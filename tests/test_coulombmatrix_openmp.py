"""Precision and timing checks for the native CoulombMatrix OpenMP path."""

from __future__ import annotations

import time

import numpy as np
from ase import Atoms

from tests._public import CoulombMatrix, ExecutionOptions, StructureBatch


def _system(seed: int, count: int) -> Atoms:
    rng = np.random.default_rng(seed)
    cell = np.diag([18.0, 19.0, 20.0])
    positions = rng.random((count, 3)) * np.diag(cell)
    numbers = np.resize(np.asarray([1, 6, 8, 14], dtype=np.int32), count)
    return Atoms(numbers=numbers, positions=positions, cell=cell, pbc=True)


def _batch() -> StructureBatch:
    systems = [_system(1729 + index, 16 + 4 * (index % 3)) for index in range(8)]
    return StructureBatch.from_ase(
        systems, ids=[f"coulomb-{index}" for index in range(len(systems))]
    )


def _descriptor(permutation: str, num_threads: int) -> CoulombMatrix:
    return CoulombMatrix(
        n_atoms_max=24,
        permutation=permutation,
        execution=ExecutionOptions(num_threads=num_threads),
    )


def test_coulombmatrix_openmp_matches_single_thread_bitwise() -> None:
    batch = _batch()
    for permutation in ("none", "sorted_l2", "eigenspectrum"):
        serial = _descriptor(permutation, 1)
        parallel = _descriptor(permutation, 4)
        try:
            serial_result = serial.compute(batch)
            parallel_result = parallel.compute(batch)
            assert serial_result.metadata["execution"]["num_threads"] == 1
            assert parallel_result.metadata["execution"]["num_threads"] == 4
            np.testing.assert_array_equal(parallel_result.values, serial_result.values)
        finally:
            serial.close()
            parallel.close()


def _median_runtime(descriptor: CoulombMatrix, batch: StructureBatch) -> float:
    descriptor.compute(batch)
    elapsed = []
    for _ in range(3):
        started = time.perf_counter()
        descriptor.compute(batch)
        elapsed.append(time.perf_counter() - started)
    return float(np.median(elapsed))


def test_coulombmatrix_openmp_small_batch_speed() -> None:
    batch = StructureBatch.from_ase(
        [_system(4000 + index, 96) for index in range(16)],
        ids=[f"coulomb-speed-{index}" for index in range(16)],
    )
    serial = _descriptor("none", 1)
    parallel = _descriptor("none", 4)
    try:
        serial_seconds = _median_runtime(serial, batch)
        parallel_seconds = _median_runtime(parallel, batch)
        assert serial_seconds > 0.0
        assert parallel_seconds > 0.0
        print(
            "CoulombMatrix small batch: "
            f"serial={serial_seconds:.6f}s, "
            f"threads=4={parallel_seconds:.6f}s, "
            f"speedup={serial_seconds / parallel_seconds:.2f}x"
        )
    finally:
        serial.close()
        parallel.close()
