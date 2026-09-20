"""CUDA MBTR channel decomposition parity and boundary coverage."""

from __future__ import annotations

import numpy as np
import pytest
from tests._cuda import load_cuda_for_tests

from mdescriptor import ExecutionOptions, StructureBatch
from mdescriptor.descriptors import LMBTR, MBTR, ValleOganov


def _batch(*, permute_first: bool = False) -> StructureBatch:
    numbers = np.asarray([1, 8, 6, 1, 14, 8, 1], dtype=np.int32)
    positions = np.asarray(
        [
            [0.1, 0.1, 0.1],
            [5.7, 0.1, 0.1],
            [0.1, 5.7, 0.1],
            [2.0, 2.0, 2.0],
            [1.0, 1.0, 1.0],
            [2.2, 1.0, 1.0],
            [1.0, 2.2, 1.0],
        ],
        dtype=np.float64,
    )
    if permute_first:
        order = np.asarray([2, 0, 3, 1], dtype=np.int64)
        numbers = np.concatenate((numbers[:4][order], numbers[4:]))
        positions = np.concatenate((positions[:4][order], positions[4:]))
    return StructureBatch(
        numbers,
        positions,
        np.tile(np.eye(3, dtype=np.float64) * 6.0, (3, 1, 1)),
        np.ones((3, 3), dtype=np.int32),
        np.asarray([0, 4, 4, 7], dtype=np.int64),
        ("periodic", "empty", "second"),
    )


def _batch_for_species(species: list[int]) -> StructureBatch:
    base = _batch()
    numbers = np.resize(np.asarray(species, dtype=np.int32), base.numbers.size)
    return StructureBatch(
        numbers,
        base.positions,
        base.cells,
        base.pbc,
        base.offsets,
        base.ids,
    )


@pytest.mark.gpu
@pytest.mark.parametrize(
    ("descriptor_type", "parameters"),
    [
        (
            MBTR,
            {
                "species": [1, 6, 8, 14],
                "geometry": {"function": "distance"},
                "grid": {"min": 0.0, "max": 6.0, "n": 20, "sigma": 0.2},
                "weighting": {"function": "smooth_cutoff", "r_cut": 3.0},
                "normalization": "valle_oganov",
            },
        ),
        (
            MBTR,
            {
                "species": [1, 6, 8, 14],
                "geometry": {"function": "angle"},
                "grid": {"min": 0.0, "max": 180.0, "n": 20, "sigma": 0.5},
                "weighting": {"function": "smooth_cutoff", "r_cut": 3.0},
                "normalization": "n_atoms",
            },
        ),
        (
            MBTR,
            {
                "species": [1],
                "geometry": {"function": "distance"},
                "grid": {"min": 0.0, "max": 6.0, "n": 20, "sigma": 0.2},
                "weighting": {"function": "smooth_cutoff", "r_cut": 3.0},
                "normalization": "none",
            },
        ),
        (
            ValleOganov,
            {"species": [1, 6], "function": "angle", "n": 20, "sigma": 0.5, "r_cut": 3.0},
        ),
        (
            MBTR,
            {
                "species": [1, 6, 8, 14],
                "geometry": {"function": "atomic_number"},
                "grid": {"min": 1.0, "max": 16.0, "n": 20, "sigma": 0.2},
                "weighting": {"function": "unity"},
                "normalization": "none",
            },
        ),
        (
            MBTR,
            {
                "species": [1, 6, 8, 14],
                "geometry": {"function": "inverse_distance"},
                "grid": {"min": 0.0, "max": 2.0, "n": 20, "sigma": 0.2},
                "weighting": {"function": "inverse_square", "r_cut": 3.0},
                "normalization": "none",
            },
        ),
        (
            MBTR,
            {
                "species": [1, 6, 8, 14],
                "geometry": {"function": "cosine"},
                "grid": {"min": -1.0, "max": 1.0, "n": 20, "sigma": 0.05},
                "weighting": {"function": "smooth_cutoff", "r_cut": 3.0},
                "normalization": "l2",
            },
        ),
        (
            ValleOganov,
            {"species": [1, 6, 8, 14], "function": "angle", "n": 20, "sigma": 0.5, "r_cut": 3.0},
        ),
    ],
)
def test_cuda_mbtr_channel_split_matches_cpu_and_repeats(
    descriptor_type: type[object], parameters: dict[str, object]
) -> None:
    load_cuda_for_tests()
    declared_species = parameters["species"]
    assert isinstance(declared_species, list)
    base = _batch()
    batch = (
        base
        if set(np.unique(base.numbers)).issubset(declared_species)
        else _batch_for_species(declared_species)
    )
    cpu = descriptor_type(**parameters, execution=ExecutionOptions(device="cpu", num_threads=1))
    gpu = descriptor_type(**parameters, execution=ExecutionOptions(device="cuda"))
    try:
        expected = cpu.compute(batch)
        actual = gpu.compute(batch)
        repeated = gpu.compute(batch)
        np.testing.assert_allclose(actual.values, expected.values, rtol=2e-10, atol=2e-11)
        np.testing.assert_array_equal(actual.values, repeated.values)
        np.testing.assert_array_equal(actual.row_offsets, expected.row_offsets)
        assert actual.values.shape == expected.values.shape
        assert np.isfinite(actual.values).all()
        empty_start = int(actual.row_offsets[1]) if actual.row_offsets is not None else 1
        if actual.level == "structure":
            np.testing.assert_array_equal(actual.values[1], 0.0)
        else:
            assert empty_start == 4
    finally:
        cpu.close()
        gpu.close()


@pytest.mark.gpu
def test_cuda_lmbtr_empty_rows_and_atom_permutation_match_cpu() -> None:
    load_cuda_for_tests()
    parameters = {
        "species": [1, 6, 8, 14],
        "geometry": {"function": "angle"},
        "grid": {"min": 0.0, "max": 180.0, "n": 20, "sigma": 0.5},
        "weighting": {"function": "smooth_cutoff", "r_cut": 3.0},
    }
    batch = _batch()
    permuted = _batch(permute_first=True)
    cpu = LMBTR(**parameters, execution=ExecutionOptions(device="cpu", num_threads=1))
    gpu = LMBTR(**parameters, execution=ExecutionOptions(device="cuda"))
    try:
        expected = cpu.compute(batch)
        actual = gpu.compute(batch)
        permuted_result = gpu.compute(permuted)
        np.testing.assert_allclose(actual.values, expected.values, rtol=2e-10, atol=2e-11)
        np.testing.assert_array_equal(actual.row_offsets, np.asarray([0, 4, 4, 7]))
        np.testing.assert_array_equal(actual.values[4:4], np.empty((0, actual.values.shape[1])))
        np.testing.assert_allclose(
            actual.values[:4][[2, 0, 3, 1]],
            permuted_result.values[:4],
            rtol=2e-10,
            atol=2e-11,
        )
    finally:
        cpu.close()
        gpu.close()
