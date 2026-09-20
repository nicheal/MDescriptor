"""CSR result ownership and serialization contracts."""

from __future__ import annotations

import pickle

import numpy as np
import pytest
from scipy import sparse

import mdescriptor.core.result as result_module
from mdescriptor.core.result import DescriptorResult


def _noncanonical_values() -> sparse.csr_matrix:
    return sparse.csr_matrix(
        (
            np.asarray([2.0, 3.0, 4.0]),
            np.asarray([1, 1, 0], dtype=np.int32),
            np.asarray([0, 2, 3], dtype=np.int32),
        ),
        shape=(2, 2),
    )


def _result(values: sparse.csr_matrix) -> DescriptorResult:
    return DescriptorResult(
        values,
        "structure",
        ("a", "b"),
        None,
        ("x", "y"),
        {},
    )


def test_csr_snapshot_is_canonical_and_detached_from_input() -> None:
    source = _noncanonical_values()
    source_data = source.data.copy()
    source_indices = source.indices.copy()

    result = _result(source)

    np.testing.assert_array_equal(result.values.toarray(), [[0.0, 5.0], [4.0, 0.0]])
    assert result.values.has_canonical_format
    assert not source.has_canonical_format
    np.testing.assert_array_equal(source.data, source_data)
    np.testing.assert_array_equal(source.indices, source_indices)
    for name in ("data", "indices", "indptr"):
        assert getattr(source, name).flags.writeable
        assert not np.shares_memory(getattr(source, name), getattr(result.values, name))
        assert not getattr(result.values, name).flags.writeable

    source.data[0] = 99.0
    assert result.values[0, 1] == 5.0
    with pytest.raises(ValueError, match="read-only"):
        result.values.data = np.array([9.0])
    with pytest.raises(ValueError):
        result.values.indices[0] = 0
    with pytest.raises(ValueError):
        result.values[0, 0] = 9.0


def test_csr_snapshot_pickle_restores_read_only_values() -> None:
    result = _result(_noncanonical_values())
    restored = pickle.loads(pickle.dumps(result))

    np.testing.assert_array_equal(restored.values.toarray(), result.values.toarray())
    assert restored.values.__class__.__name__ == "csr_matrix"
    for name in ("data", "indices", "indptr"):
        assert not getattr(restored.values, name).flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        restored.values.indptr = np.array([0, 1, 2])


def test_csr_snapshot_reuses_the_owned_canonical_storage(monkeypatch) -> None:
    original = result_module._readonly_csr
    observed: dict[str, object] = {}

    def observe(values, scipy_sparse, *, copy_values=True):
        observed["owned"] = values
        observed["copy_values"] = copy_values
        return original(values, scipy_sparse, copy_values=copy_values)

    monkeypatch.setattr(result_module, "_readonly_csr", observe)
    source = _noncanonical_values()
    result = _result(source)
    owned = observed["owned"]

    assert observed["copy_values"] is False
    for name in ("data", "indices", "indptr"):
        assert not np.shares_memory(getattr(source, name), getattr(owned, name))
        assert np.shares_memory(getattr(owned, name), getattr(result.values, name))
