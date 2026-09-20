"""Focused tests for the private dense CUDA result handoff."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

import mdescriptor.core.adapter as adapter_module
from mdescriptor import StructureBatch
from mdescriptor.core.backends import _combine_cuda_block_results, _cuda_result
from mdescriptor.core.options import OutputOptions
from mdescriptor.core.result import DescriptorResult, _OwnedDenseValues


def _batch(structures: int = 2) -> StructureBatch:
    return StructureBatch(
        np.ones(structures, dtype=np.int32),
        np.zeros((structures, 3), dtype=np.float64),
        np.tile(np.eye(3, dtype=np.float64), (structures, 1, 1)),
        np.zeros((structures, 3), dtype=np.int32),
        np.arange(structures + 1, dtype=np.int64),
        tuple(str(index) for index in range(structures)),
    )


def test_public_result_still_snapshots_an_owning_array() -> None:
    values = np.empty((2, 2), dtype=np.float64)
    values[...] = np.arange(4.0).reshape(2, 2)
    assert values.flags.owndata is True
    result = DescriptorResult(values, "structure", ("a", "b"), None, ("x", "y"))

    values[0, 0] = 99.0

    assert result.values[0, 0] == 0.0
    assert result.values is not values
    assert result.values.flags.writeable is False


def test_cuda_block_combination_accepts_mixed_layout_and_empty_blocks() -> None:
    fortran = np.asfortranarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    strided = np.asarray(
        [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
        dtype=np.float32,
    )[:, ::-1]
    raw = _combine_cuda_block_results(
        [
            {"values": fortran, "level": "structure"},
            {"values": strided, "level": "structure"},
        ]
    )
    combined = raw["values"].array
    result = _cuda_result(raw, _batch(4), "fake")

    assert combined.flags.owndata is True
    assert combined.flags.c_contiguous is True
    np.testing.assert_array_equal(
        result.values,
        np.concatenate((fortran, strided), axis=0),
    )

    empty = _combine_cuda_block_results(
        [
            {"values": np.empty((0, 3), order="F"), "level": "structure"},
            {"values": np.empty((0, 3)), "level": "structure"},
        ]
    )
    empty_values = empty["values"].array
    assert empty_values.shape == (0, 3)
    assert empty_values.flags.owndata is True
    assert empty_values.flags.c_contiguous is True


def test_only_combined_cuda_values_use_the_private_snapshot_seam() -> None:
    first = np.asarray([[1.0, 2.0]])
    second = np.asarray([[3.0, 4.0]])
    raw = _combine_cuda_block_results(
        [
            {"values": first, "level": "structure", "labels": ("x", "y")},
            {"values": second, "level": "structure", "labels": ("x", "y")},
        ]
    )

    assert isinstance(raw["values"], _OwnedDenseValues)
    combined = raw["values"].array
    result = _cuda_result(raw, _batch(), "fake")

    assert result.values is combined
    assert result.values.flags.writeable is False
    np.testing.assert_array_equal(result.values, [[1.0, 2.0], [3.0, 4.0]])


def test_arbitrary_cuda_mapping_is_still_copied() -> None:
    values = np.asarray([[1.0, 2.0], [3.0, 4.0]])
    result = _cuda_result(
        {"values": values, "level": "structure", "labels": ("x", "y")},
        _batch(),
        "fake",
    )

    values[0, 0] = 99.0

    assert result.values is not values
    assert result.values[0, 0] == 1.0


def test_adapter_dense_and_sparse_output_keep_their_contracts() -> None:
    result = DescriptorResult(
        np.asarray([[1.0, 2.0], [3.0, 4.0]]),
        "structure",
        ("a", "b"),
        None,
        ("x", "y"),
    )

    dense = adapter_module._apply_output(result, OutputOptions(dtype="float32"))
    assert dense.values.dtype == np.float32
    assert dense.values.flags.writeable is False

    sparse_result = adapter_module._apply_output(
        result,
        OutputOptions(dtype="float32", sparse=True),
    )
    assert sparse.isspmatrix_csr(sparse_result.values)
    assert sparse_result.values.dtype == np.float32
    with pytest.raises(ValueError, match="read-only"):
        sparse_result.values[0, 0] = 9.0


def test_adapter_f_order_conversion_falls_back_to_the_normal_snapshot(monkeypatch) -> None:
    result = DescriptorResult(
        np.asarray([[1.0, 2.0], [3.0, 4.0]]),
        "structure",
        ("a", "b"),
        None,
        ("x", "y"),
    )

    def fortran_values(values, *, dtype, sparse):
        del sparse
        return np.asfortranarray(values, dtype=np.dtype(dtype))

    monkeypatch.setattr(adapter_module, "format_values", fortran_values)
    converted = adapter_module._apply_output(result, OutputOptions(dtype="float64"))

    assert converted.values.flags.c_contiguous is True
    assert converted.values.flags.writeable is False
    np.testing.assert_array_equal(converted.values, result.values)

    same = result._replace_output(
        result.values,
        {"dtype": "float64", "sparse": False},
    )
    assert same.values is result.values
