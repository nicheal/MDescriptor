"""Public contract tests shared by the structure-level matrix descriptors."""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms

from tests._public import (
    ComputeControl,
    CoulombMatrix,
    EwaldSumMatrix,
    SineMatrix,
    StructureBatch,
)

MATRIX_DESCRIPTORS = (
    ("CoulombMatrix", CoulombMatrix, {}),
    ("SineMatrix", SineMatrix, {}),
    (
        "EwaldSumMatrix",
        EwaldSumMatrix,
        {"accuracy": 1e-5, "w": 1.0, "r_cut": 4.0, "g_cut": 3.0, "a": 0.3},
    ),
)


def _batch() -> StructureBatch:
    systems = [
        Atoms(
            "NaCl",
            positions=[[0.0, 0.0, 0.0], [1.2, 0.1, 0.0]],
            cell=np.diag([8.0, 8.0, 8.0]),
            pbc=True,
        ),
        Atoms(
            "Si3",
            positions=[[0.0, 0.0, 0.0], [1.4, 0.2, 0.0], [0.1, 1.3, 0.4]],
            cell=np.diag([9.0, 9.0, 9.0]),
            pbc=True,
        ),
    ]
    return StructureBatch.from_ase(systems)


def _batch_with_empty_frame() -> StructureBatch:
    return StructureBatch(
        np.array([11, 17], dtype=np.int32),
        np.array([[0.0, 0.0, 0.0], [1.2, 0.1, 0.0]], dtype=np.float64),
        np.stack([np.diag([8.0, 8.0, 8.0]), np.diag([8.0, 8.0, 8.0])]),
        np.ones((2, 3), dtype=np.int32),
        np.array([0, 2, 2], dtype=np.int64),
        ("full", "empty"),
    )


def _empty_batch() -> StructureBatch:
    return StructureBatch(
        np.empty(0, dtype=np.int32),
        np.empty((0, 3), dtype=np.float64),
        np.stack([np.eye(3), np.eye(3)]),
        np.ones((2, 3), dtype=np.int32),
        np.array([0, 0, 0], dtype=np.int64),
        ("empty-0", "empty-1"),
    )


def _descriptor(descriptor_type, parameters, permutation: str):
    return descriptor_type(n_atoms_max=4, permutation=permutation, **parameters)


@pytest.mark.parametrize("name, descriptor_type, parameters", MATRIX_DESCRIPTORS)
def test_matrix_descriptors_share_shape_and_padding_contract(
    name, descriptor_type, parameters
) -> None:
    descriptor = _descriptor(descriptor_type, parameters, "none")
    try:
        result = descriptor.compute(_batch())
    finally:
        descriptor.close()

    assert result.values.shape == (2, 16), name
    assert result.values.ndim == 2, name
    assert np.isfinite(result.values).all(), name
    np.testing.assert_array_equal(result.values[0].reshape(4, 4)[2:, :], 0.0)
    np.testing.assert_array_equal(result.values[0].reshape(4, 4)[:, 2:], 0.0)
    np.testing.assert_array_equal(result.values[1].reshape(4, 4)[3:, :], 0.0)
    np.testing.assert_array_equal(result.values[1].reshape(4, 4)[:, 3:], 0.0)


@pytest.mark.parametrize("name, descriptor_type, parameters", MATRIX_DESCRIPTORS)
def test_matrix_descriptors_share_sorted_l2_contract(
    name, descriptor_type, parameters
) -> None:
    batch = _batch()
    raw_descriptor = _descriptor(descriptor_type, parameters, "none")
    sorted_descriptor = _descriptor(descriptor_type, parameters, "sorted_l2")
    try:
        raw = raw_descriptor.compute(batch).values
        actual = sorted_descriptor.compute(batch).values
    finally:
        raw_descriptor.close()
        sorted_descriptor.close()

    for index, count in enumerate((2, 3)):
        matrix = raw[index].reshape(4, 4)[:count, :count]
        order = np.argsort(-np.linalg.norm(matrix, axis=1), kind="stable")
        expected = np.zeros((4, 4))
        expected[:count, :count] = matrix[order][:, order]
        np.testing.assert_allclose(actual[index].reshape(4, 4), expected, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("name, descriptor_type, parameters", MATRIX_DESCRIPTORS)
def test_matrix_descriptors_share_eigenspectrum_contract(
    name, descriptor_type, parameters
) -> None:
    batch = _batch()
    raw_descriptor = _descriptor(descriptor_type, parameters, "none")
    spectrum_descriptor = _descriptor(descriptor_type, parameters, "eigenspectrum")
    try:
        raw = raw_descriptor.compute(batch).values
        actual = spectrum_descriptor.compute(batch).values
    finally:
        raw_descriptor.close()
        spectrum_descriptor.close()

    assert actual.shape == (2, 4), name
    for index, count in enumerate((2, 3)):
        matrix = raw[index].reshape(4, 4)[:count, :count]
        expected = np.linalg.eigvalsh(matrix)
        expected = expected[np.argsort(np.abs(expected))[::-1]]
        np.testing.assert_allclose(actual[index, :count], expected, rtol=1e-8, atol=1e-10)
        np.testing.assert_array_equal(actual[index, count:], 0.0)


@pytest.mark.parametrize("name, descriptor_type, parameters", MATRIX_DESCRIPTORS)
def test_matrix_descriptors_accept_empty_frames_and_report_progress(
    name, descriptor_type, parameters
) -> None:
    descriptor = _descriptor(descriptor_type, parameters, "none")
    control = ComputeControl()
    try:
        result = descriptor.compute(_batch_with_empty_frame(), control=control)
    finally:
        descriptor.close()

    assert result.values.shape == (2, 16), name
    np.testing.assert_array_equal(result.values[1], 0.0)
    assert control.total() == 2
    assert control.completed() == 2


@pytest.mark.parametrize("name, descriptor_type, parameters", MATRIX_DESCRIPTORS)
def test_all_empty_matrix_batch_keeps_zero_width_compatibility_and_progress(
    name, descriptor_type, parameters
) -> None:
    descriptor = descriptor_type(permutation="none", **parameters)
    control = ComputeControl()
    try:
        result = descriptor.compute(_empty_batch(), control=control)
    finally:
        descriptor.close()

    assert result.values.shape == (2, 0), name
    assert control.total() == 2
    assert control.completed() == 2
