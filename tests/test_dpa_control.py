"""Regression coverage for forwarding cooperative cancellation to DPA C++."""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from mdescriptor.models import DPA4_MODEL, DPA4C_MODEL
from tests._public import (
    DPA4,
    DPA4C,
    CancelledError,
    ComputeControl,
    ExecutionOptions,
    StructureBatch,
)

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


def _empty_batch(structures: int = 2) -> StructureBatch:
    return StructureBatch(
        np.empty(0, dtype=np.int32),
        np.empty((0, 3), dtype=np.float64),
        np.asarray([np.eye(3, dtype=np.float64) * 12.0] * structures),
        np.ones((structures, 3), dtype=np.int32),
        np.zeros(structures + 1, dtype=np.int64),
        tuple(f"empty-{index}" for index in range(structures)),
    )


def _many_structures(structures: int = 64) -> StructureBatch:
    numbers = np.tile(np.asarray([1, 8], dtype=np.int32), structures)
    positions = np.tile(
        np.asarray([[1.0, 1.0, 1.0], [2.0, 1.0, 1.0]], dtype=np.float64),
        (structures, 1),
    )
    cells = np.asarray([np.eye(3, dtype=np.float64) * 12.0] * structures)
    pbc = np.ones((structures, 3), dtype=np.int32)
    offsets = np.arange(0, 2 * structures + 1, 2, dtype=np.int64)
    return StructureBatch(
        numbers,
        positions,
        cells,
        pbc,
        offsets,
        tuple(f"progress-{index}" for index in range(structures)),
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


@pytest.mark.parametrize(
    ("descriptor_type", "model"),
    [(DPA4, DPA4_MODEL), (DPA4C, DPA4C_MODEL)],
)
def test_dpa_native_empty_structures_count_as_completed(descriptor_type, model) -> None:
    descriptor = descriptor_type(model=model)
    control = ComputeControl()
    try:
        result = descriptor.compute(_empty_batch(), control=control)
        assert result.values.shape[0] == 0
        assert control.total() == 2
        assert control.completed() == 2
    finally:
        descriptor.close()


def test_dpa4c_native_cancellation_stops_after_completed_structures() -> None:
    descriptor = DPA4C(
        model=DPA4C_MODEL,
        execution=ExecutionOptions(num_threads=1),
    )
    control = ComputeControl()
    errors: list[BaseException] = []

    def compute() -> None:
        try:
            descriptor.compute(_many_structures(), control=control)
        except BaseException as exc:  # pragma: no cover - assertion below checks the type
            errors.append(exc)

    worker = threading.Thread(target=compute)
    worker.start()
    try:
        deadline = time.monotonic() + 30.0
        while worker.is_alive() and time.monotonic() < deadline:
            completed = control.completed()
            if 0 < completed < control.total():
                control.cancel()
                break
            time.sleep(0.001)
        worker.join(timeout=30.0)
        assert not worker.is_alive(), "DPA4C did not stop after cancellation"
        assert errors and isinstance(errors[0], CancelledError)
        assert 0 < control.completed() < control.total()
    finally:
        if worker.is_alive():
            control.cancel()
            worker.join(timeout=30.0)
        descriptor.close()
