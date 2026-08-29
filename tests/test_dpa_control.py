"""Regression coverage for forwarding cooperative cancellation to DPA C++."""

from __future__ import annotations

import numpy as np
import pytest

from mdescriptor.models import DPA4_MODEL, DPA4C_MODEL
from tests._public import DPA4, DPA4C, ComputeControl, StructureBatch

pytestmark = pytest.mark.model


def _batch() -> StructureBatch:
    return StructureBatch(
        np.asarray([1, 8, 1, 8], dtype=np.int32),
        np.asarray(
            [
                [1.0, 1.0, 1.0],
                [2.0, 1.0, 1.0],
                [6.0, 6.0, 6.0],
                [7.0, 6.0, 6.0],
            ],
            dtype=np.float64,
        ),
        np.asarray([np.eye(3, dtype=np.float64) * 12.0] * 2),
        np.ones((2, 3), dtype=np.int32),
        np.asarray([0, 2, 4], dtype=np.int64),
        ("control-forwarding-0", "control-forwarding-1"),
    )


@pytest.mark.parametrize(
    ("descriptor_type", "model"),
    [(DPA4, DPA4_MODEL), (DPA4C, DPA4C_MODEL)],
)
def test_dpa_native_compute_receives_control(descriptor_type, model) -> None:
    descriptor = descriptor_type(model=model)
    kernel = descriptor._kernel
    original_cpp = kernel._cpp
    control = object()
    received: list[object] = []

    class NativeProbe:
        def compute(
            self,
            numbers,
            positions,
            cells,
            pbc,
            offsets,
            type_indices,
            native_control=None,
        ):
            del positions, cells, pbc, offsets, type_indices
            received.append(native_control)
            return np.zeros((numbers.shape[0], kernel.feature_count), dtype=np.float64)

        def close(self):
            pass

    kernel._cpp = NativeProbe()
    try:
        kernel.compute(_batch(), control)
    finally:
        kernel._cpp = original_cpp
        descriptor.close()

    assert received == [control]


@pytest.mark.parametrize(
    ("descriptor_type", "model"),
    [(DPA4, DPA4_MODEL), (DPA4C, DPA4C_MODEL)],
)
def test_dpa_compute_reports_structure_progress(descriptor_type, model) -> None:
    descriptor = descriptor_type(model=model)
    control = ComputeControl()
    try:
        descriptor.compute(_batch(), control=control)
        assert control.total() == 2
        assert control.completed() == 2
    finally:
        descriptor.close()
