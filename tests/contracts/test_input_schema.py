"""Direct tests for the immutable StructureBatch input seam."""

from __future__ import annotations

import numpy as np
import pytest

from mdescriptor import StructureBatch


def test_structure_batch_snapshots_and_freezes_numeric_arrays():
    numbers = np.array([1, 8], dtype=np.int32)
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    cells = np.eye(3)[None] * 10.0
    pbc = np.ones((1, 3), dtype=np.int32)
    offsets = np.array([0, 2], dtype=np.int64)
    batch = StructureBatch(numbers, positions, cells, pbc, offsets, ("frame",))

    positions[0, 0] = 42.0
    offsets[1] = 1
    assert batch.positions[0, 0] == 0.0
    assert batch.offsets[1] == 2
    for value in (batch.numbers, batch.positions, batch.cells, batch.pbc, batch.offsets):
        assert value.flags.writeable is False
    with pytest.raises(ValueError):
        batch.positions[0, 0] = 1.0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("numbers", [1.5]),
        ("pbc", [[0.5, 0.0, 0.0]]),
        ("offsets", [0.0, 1.5]),
    ],
)
def test_structure_batch_rejects_fractional_integer_fields(field, value):
    kwargs = {
        "numbers": [1],
        "positions": [[0.0, 0.0, 0.0]],
        "cells": np.eye(3)[None] * 10.0,
        "pbc": [[1, 1, 1]],
        "offsets": [0, 1],
        "ids": ("frame",),
    }
    kwargs[field] = value
    with pytest.raises(ValueError, match="integer"):
        StructureBatch(**kwargs)


def test_structure_batch_from_frames_packs_gui_records():
    frames = (
        {
            "numbers": [1, 8],
            "positions": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            "cell": np.eye(3) * 10.0,
            "pbc": [1, 1, 1],
            "id": "periodic",
        },
        {
            "numbers": [6],
            "positions": [[2.0, 0.0, 0.0]],
            "cell": np.zeros((3, 3)),
            "pbc": [0, 0, 0],
            "id": "isolated",
        },
    )

    batch = StructureBatch.from_frames(frame for frame in frames)

    np.testing.assert_array_equal(batch.numbers, [1, 8, 6])
    np.testing.assert_array_equal(batch.offsets, [0, 2, 3])
    np.testing.assert_array_equal(batch.cells[1], np.zeros((3, 3)))
    np.testing.assert_array_equal(batch.pbc, [[1, 1, 1], [0, 0, 0]])
    assert batch.ids == ("periodic", "isolated")
