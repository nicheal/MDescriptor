"""Direct tests for the shared result and sample-index deep module."""

from __future__ import annotations

import numpy as np
import pytest

from mdescriptor.core.result import DescriptorResult, pair_samples


def test_empty_feature_labels_are_still_exactly_width_zero():
    result = DescriptorResult(
        np.empty((2, 0)),
        "structure",
        ("a", "b"),
        None,
        (),
        {"descriptor": "empty", "backend": "test"},
    )
    assert result.feature_count == 0
    assert result.samples.shape == (2, 1)
    assert result.samples.dtype == np.int64
    assert result.metadata["schema_version"] == 1


def test_atom_and_pair_samples_use_local_indices_and_are_contiguous():
    atom = DescriptorResult(
        np.zeros((3, 2)),
        "atom",
        ("a", "b"),
        np.asarray([0, 2, 3], dtype=np.int64),
        ("x", "y"),
        {"descriptor": "atom", "backend": "test"},
    )
    np.testing.assert_array_equal(atom.samples, [[0, 0], [0, 1], [1, 0]])
    assert atom.samples.flags.c_contiguous
    records = np.asarray([[0, 1, 2, 0, -1], [2, 2, 0, 0, 0]], dtype=np.int64)
    samples = pair_samples(records, np.asarray([0, 1, 2]), np.asarray([0, 2, 3]))
    pair = DescriptorResult(
        np.zeros((2, 1)),
        "pair",
        ("a", "b"),
        np.asarray([0, 1, 2]),
        ("x",),
        {"descriptor": "pair", "backend": "test"},
        samples=samples,
        _atom_row_offsets=np.asarray([0, 2, 3]),
    )
    np.testing.assert_array_equal(pair.samples, [[0, 0, 1, 2, 0, -1], [1, 0, 0, 0, 0, 0]])
    assert pair.samples.dtype == np.int64
    assert "pair_records" not in pair.metadata


def test_result_rejects_bad_labels_samples_and_live_metadata():
    with pytest.raises(ValueError, match="labels"):
        DescriptorResult(np.zeros((1, 1)), "structure", ("a",), None, (), {})
    with pytest.raises(ValueError, match="structure samples"):
        DescriptorResult(
            np.zeros((1, 1)),
            "structure",
            ("a",),
            None,
            ("x",),
            {},
            samples=np.asarray([[1]], dtype=np.int64),
        )
    with pytest.raises(TypeError, match="JSON-safe"):
        DescriptorResult(
            np.zeros((1, 1)), "structure", ("a",), None, ("x",), {"bad": object()}
        )
    with pytest.raises(ValueError, match="local atom"):
        pair_samples(
            np.asarray([[0, 2, 0, 0, 0]], dtype=np.int64),
            np.asarray([0, 1]),
            np.asarray([0, 2]),
        )
    with pytest.raises(ValueError, match="local atom"):
        DescriptorResult(
            np.zeros((1, 1)),
            "pair",
            ("a",),
            np.asarray([0, 1]),
            ("x",),
            {},
            samples=np.asarray([[0, 99, 0, 0, 0, 0]], dtype=np.int64),
            _atom_row_offsets=np.asarray([0, 2]),
        )
