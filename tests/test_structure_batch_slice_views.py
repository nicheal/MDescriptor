"""Internal StructureBatch block views preserve ownership and semantics."""

from __future__ import annotations

import gc

import numpy as np

from mdescriptor import StructureBatch
from mdescriptor.core.backends import CudaBackend


def test_public_batch_stays_owned_and_slice_reuses_validated_buffers() -> None:
    numbers = np.asarray([1, 8, 6, 7, 16], dtype=np.int32)
    positions = np.arange(15.0).reshape(5, 3)
    cells = np.zeros((3, 3, 3), dtype=np.float64)
    pbc = np.zeros((3, 3), dtype=np.int32)
    offsets = np.asarray([0, 2, 2, 5], dtype=np.int64)
    spins = np.arange(15.0).reshape(5, 3)
    charge_spin = np.arange(6.0).reshape(3, 2)
    batch = StructureBatch(
        numbers,
        positions,
        cells,
        pbc,
        offsets,
        ("first", "empty", "last"),
        spins,
        charge_spin,
    )

    numbers[0] = 99
    positions[0, 0] = 99
    assert batch.numbers[0] == 1
    assert batch.positions[0, 0] == 0

    view = batch._slice_view(1, 3)
    assert view.ids == ("empty", "last")
    assert view.structures == 2
    assert view.atoms == 3
    np.testing.assert_array_equal(view.offsets, [0, 0, 3])
    np.testing.assert_array_equal(view.numbers, [6, 7, 16])
    np.testing.assert_array_equal(view.spins, spins[2:])
    np.testing.assert_array_equal(view.charge_spin, charge_spin[1:])

    for name in ("numbers", "positions", "cells", "pbc", "spins", "charge_spin"):
        assert np.shares_memory(getattr(batch, name), getattr(view, name))
        assert not getattr(view, name).flags.writeable
    assert not np.shares_memory(batch.offsets, view.offsets)
    assert not view.offsets.flags.writeable

    del batch
    gc.collect()
    np.testing.assert_array_equal(view.numbers, [6, 7, 16])
    np.testing.assert_array_equal(view.positions[:, 0], [6.0, 9.0, 12.0])


def test_slice_view_handles_an_empty_batch() -> None:
    batch = StructureBatch(
        np.empty(0, dtype=np.int32),
        np.empty((0, 3), dtype=np.float64),
        np.empty((0, 3, 3), dtype=np.float64),
        np.empty((0, 3), dtype=np.int32),
        np.asarray([0], dtype=np.int64),
        (),
    )

    view = batch._slice_view(0, 0)
    assert view.structures == 0
    assert view.atoms == 0
    assert view.ids == ()
    assert view.offsets.tolist() == [0]
    for value in (view.numbers, view.positions, view.cells, view.pbc, view.offsets):
        assert not value.flags.writeable


def test_cuda_structure_blocks_reuse_views_and_match_unsliced_result() -> None:
    structures = 65
    counts = np.asarray([index % 4 for index in range(structures)], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(counts)))
    batch = StructureBatch(
        np.ones(int(offsets[-1]), dtype=np.int32),
        np.column_stack(
            (
                np.arange(int(offsets[-1]), dtype=np.float64),
                np.zeros((int(offsets[-1]), 2), dtype=np.float64),
            )
        ),
        np.zeros((structures, 3, 3), dtype=np.float64),
        np.zeros((structures, 3), dtype=np.int32),
        offsets,
        tuple(f"frame-{index}" for index in range(structures)),
    )
    blocks: list[StructureBatch] = []

    class FakeBackend:
        def compute(self, block: StructureBatch, control: object) -> dict[str, np.ndarray]:
            del control
            blocks.append(block)
            values = np.column_stack(
                (block.numbers.astype(np.float64), block.positions[:, 0])
            )
            return {"values": values, "row_offsets": block.offsets.copy()}

    backend = CudaBackend("NeighborList", {})
    actual = backend._compute_in_structure_blocks(FakeBackend(), batch, None)
    expected = np.column_stack(
        (batch.numbers.astype(np.float64), batch.positions[:, 0])
    )

    assert len(blocks) == 3
    np.testing.assert_array_equal(actual["values"], expected)
    np.testing.assert_array_equal(actual["row_offsets"], batch.offsets)
    assert tuple(identifier for block in blocks for identifier in block.ids) == batch.ids
    for block in blocks:
        for name in ("numbers", "positions"):
            if block.atoms:
                assert np.shares_memory(getattr(batch, name), getattr(block, name))
            else:
                assert getattr(block, name).base is not None
            assert not getattr(block, name).flags.writeable
        for name in ("cells", "pbc"):
            if block.structures:
                assert np.shares_memory(getattr(batch, name), getattr(block, name))
            else:
                assert getattr(block, name).base is not None
            assert not getattr(block, name).flags.writeable
