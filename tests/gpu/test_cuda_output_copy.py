"""CUDA extended-descriptor output ownership regressions."""

from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from tests._cuda import load_cuda_for_tests

from mdescriptor import ComputeControl, ExecutionOptions, OutputOptions, StructureBatch
from mdescriptor.descriptors import AtomicComposition


def _module_path(module_name: str, environment_name: str) -> Path:
    module = importlib.import_module(module_name)
    location = getattr(module, "__file__", None)
    assert location
    actual = Path(location).resolve()
    configured = os.environ.get(environment_name)
    if configured:
        assert actual.parent == Path(configured).expanduser().resolve()
    print(f"{module_name}={actual}")
    return actual


def _loaded_cuda() -> Any:
    load_cuda_for_tests()
    cuda_path = _module_path("mdescriptor._cuda", "MDESCRIPTOR_CUDA_PLUGIN_DIR")
    native_environment = "MDESCRIPTOR_EXPECTED_NATIVE_PLUGIN_DIR"
    if native_environment not in os.environ:
        native_environment = "MDESCRIPTOR_NATIVE_PLUGIN_DIR"
    native_path = _module_path("mdescriptor._native", native_environment)
    assert cuda_path.suffix in {".so", ".pyd", ".dylib"}
    assert native_path.suffix in {".so", ".pyd", ".dylib"}
    return importlib.import_module("mdescriptor._cuda")


def _batch(numbers: list[int], identifier: str, spacing: float = 0.8) -> StructureBatch:
    count = len(numbers)
    positions = np.zeros((count, 3), dtype=np.float64)
    positions[:, 0] = np.arange(count, dtype=np.float64) * spacing
    return StructureBatch(
        np.asarray(numbers, dtype=np.int32),
        positions,
        np.eye(3, dtype=np.float64)[None, :, :] * 20.0,
        np.zeros((1, 3), dtype=np.int32),
        np.asarray([0, count], dtype=np.int64),
        (identifier,),
    )


def _empty_batch() -> StructureBatch:
    return StructureBatch(
        np.empty(0, dtype=np.int32),
        np.empty((0, 3), dtype=np.float64),
        np.tile(np.eye(3, dtype=np.float64), (2, 1, 1)),
        np.zeros((2, 3), dtype=np.int32),
        np.asarray([0, 0, 0], dtype=np.int64),
        ("empty-0", "empty-1"),
    )


def _many_structure_batch(structures: int) -> StructureBatch:
    return StructureBatch(
        np.ones(structures, dtype=np.int32),
        np.zeros((structures, 3), dtype=np.float64),
        np.tile(np.eye(3, dtype=np.float64) * 20.0, (structures, 1, 1)),
        np.zeros((structures, 3), dtype=np.int32),
        np.arange(structures + 1, dtype=np.int64),
        tuple(str(index) for index in range(structures)),
    )


def _assert_reuse_and_close(
    backend: Any,
    first_batch: StructureBatch,
    second_batch: StructureBatch,
) -> tuple[np.ndarray, np.ndarray]:
    first_values: np.ndarray
    second_values: np.ndarray
    first_snapshot: np.ndarray
    try:
        first_result = backend.compute(first_batch, ComputeControl())
        first_values = np.asarray(first_result["values"])
        first_snapshot = first_values.copy()
        second_result = backend.compute(second_batch, ComputeControl())
        second_values = np.asarray(second_result["values"])
        np.testing.assert_array_equal(first_values, first_snapshot)
    finally:
        backend.close()
    np.testing.assert_array_equal(first_values, first_snapshot)
    return first_values, second_values


@pytest.mark.gpu
def test_cuda_atom_output_owns_large_values_across_reuse_and_close() -> None:
    """Atom-level values stay detached from the reusable device buffer."""

    cuda = _loaded_cuda()
    numbers = [1, 8, 14, 1] * 1024
    second_numbers = [14, 8, 1, 8] * 1024
    backend = cuda.CudaBackend(
        "AtomicComposition", {"species": [1, 8, 14], "per_system": False}
    )
    first_values, second_values = _assert_reuse_and_close(
        backend, _batch(numbers, "first"), _batch(second_numbers, "second")
    )
    expected = (np.asarray(numbers)[:, None] == np.asarray([1, 8, 14])).astype(
        np.float64
    )
    np.testing.assert_array_equal(first_values, expected)
    assert second_values.shape == first_values.shape
    assert first_values.nbytes > 64 * 1024


@pytest.mark.gpu
def test_cuda_structure_output_owns_values_across_reuse_and_close() -> None:
    """Structure-level matrix values remain valid after a later compute."""

    cuda = _loaded_cuda()
    backend = cuda.CudaBackend(
        "CoulombMatrix", {"n_atoms_max": 8, "permutation": "none"}
    )
    first_values, second_values = _assert_reuse_and_close(
        backend,
        _batch([1] * 8, "first"),
        _batch([8] * 8, "second"),
    )
    assert first_values.shape == second_values.shape == (1, 64)
    assert np.isfinite(first_values).all()
    assert np.isfinite(second_values).all()
    assert not np.array_equal(first_values, second_values)


@pytest.mark.gpu
def test_cuda_pair_output_owns_values_across_reuse_and_close() -> None:
    """Pair-level features retain their NumPy storage after context reuse."""

    cuda = _loaded_cuda()
    backend = cuda.CudaBackend(
        "SphericalExpansionByPair",
        {
            "cutoff": 3.1,
            "density_width": 0.3,
            "max_radial": 2,
            "max_angular": 2,
            "_cuda_feature_count": 27,
        },
    )
    first_values, second_values = _assert_reuse_and_close(
        backend,
        _batch([1, 1, 1, 1], "first", spacing=0.7),
        _batch([1, 1, 1, 1], "second", spacing=0.75),
    )
    assert first_values.shape[1] == second_values.shape[1] == 27
    assert first_values.shape[0] > 0
    assert np.isfinite(first_values).all()
    assert np.isfinite(second_values).all()


@pytest.mark.gpu
def test_cuda_extended_output_empty_array_owns_its_shape() -> None:
    """The zero-width matrix fast path returns a valid owned NumPy array."""

    cuda = _loaded_cuda()
    backend = cuda.CudaBackend("CoulombMatrix", {"permutation": "none"})
    try:
        result = backend.compute(_empty_batch(), ComputeControl())
        values = np.asarray(result["values"])
        assert values.shape == (2, 0)
        assert values.size == 0
    finally:
        backend.close()


@pytest.mark.gpu
def test_public_cuda_multiblock_result_survives_reuse_and_close() -> None:
    """The public snapshot remains valid after combined CUDA blocks are reused."""

    load_cuda_for_tests()
    descriptor = AtomicComposition(
        species=[1],
        per_system=False,
        output=OutputOptions(dtype="float32"),
        execution=ExecutionOptions(device="cuda"),
    )
    first_batch = _many_structure_batch(33)
    second_batch = _many_structure_batch(34)
    try:
        first = descriptor.compute(first_batch)
        snapshot = np.asarray(first.values).copy()
        assert snapshot.shape == (33, 1)
        assert first.values.dtype == np.float32
        assert first.values.flags.writeable is False
        second = descriptor.compute(second_batch)
        assert second.values.shape == (34, 1)
    finally:
        descriptor.close()

    np.testing.assert_array_equal(first.values, snapshot)
    np.testing.assert_array_equal(first.values, 1.0)
