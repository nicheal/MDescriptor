"""OpenMP accuracy contract for the local MBTR path."""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms

from tests._golden import _single_structure
from tests._public import LMBTR, ExecutionOptions, StructureBatch


def _batch() -> StructureBatch:
    """Build a small periodic batch with enough local MBTR work."""

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


@pytest.fixture(scope="module")
def lmbtr_single_batch(lmbtr_batch: StructureBatch) -> StructureBatch:
    return _single_structure(lmbtr_batch, 0)


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
