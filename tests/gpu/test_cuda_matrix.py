"""CUDA matrix descriptor contract tests."""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest
from ase import Atoms
from tests._cuda import load_cuda_for_tests

from mdescriptor import (
    CancelledError,
    ComputeControl,
    ExecutionOptions,
    MDescriptorError,
    StructureBatch,
)
from mdescriptor.descriptors import CoulombMatrix, EwaldSumMatrix, SineMatrix


def _batch() -> StructureBatch:
    return StructureBatch.from_ase(
        [
            Atoms(
                "NaCl",
                positions=[[0.0, 0.0, 0.0], [2.1, 2.0, 2.2]],
                cell=np.diag([8.0, 8.0, 8.0]),
                pbc=True,
            ),
            Atoms(
                "Si3",
                positions=[[0.2, 0.1, 0.0], [1.6, 0.0, 0.1], [0.4, 1.5, -0.2]],
                cell=np.diag([9.0, 10.0, 11.0]),
                pbc=True,
            ),
        ],
        ids=["salt", "silicon"],
    )


MATRIX_DESCRIPTORS = [
    (CoulombMatrix, {}),
    (SineMatrix, {}),
    (EwaldSumMatrix, {"accuracy": 1e-5, "w": 1.0, "r_cut": 4.0, "g_cut": 3.0, "a": 0.3}),
]


@pytest.mark.gpu
@pytest.mark.parametrize(("descriptor_type", "parameters"), MATRIX_DESCRIPTORS)
@pytest.mark.parametrize("permutation", ["none", "sorted_l2", "eigenspectrum"])
def test_cuda_matrix_contract_matches_cpu_shape_and_output_semantics(
    descriptor_type: type[object], parameters: dict[str, float], permutation: str
) -> None:
    """CPU and CUDA expose the same matrix width, padding, and ordering contract."""

    load_cuda_for_tests()
    batch = _batch()
    common = {"n_atoms_max": 4, "permutation": permutation, **parameters}
    cpu = descriptor_type(**common, execution=ExecutionOptions(device="cpu", num_threads=1))
    gpu = descriptor_type(**common, execution=ExecutionOptions(device="cuda"))
    try:
        expected = cpu.compute(batch)
        try:
            actual = gpu.compute(batch)
        except MDescriptorError as error:
            if error.code == "device_unavailable":
                pytest.skip(str(error))
            raise

        assert actual.level == expected.level == "structure"
        assert actual.values.shape == expected.values.shape == (2, 16 if permutation != "eigenspectrum" else 4)
        assert actual.labels == expected.labels
        assert actual.structure_ids == expected.structure_ids
        np.testing.assert_array_equal(actual.samples, expected.samples)
        np.testing.assert_allclose(np.isfinite(actual.values), True)

        counts = np.diff(batch.offsets)
        for row, count_value in zip(actual.values, counts, strict=True):
            count = int(count_value)
            if permutation == "eigenspectrum":
                assert np.all(row[count:] == 0.0)
                assert np.all(np.abs(row[: max(count - 1, 0)]) >= np.abs(row[1:count]))
            else:
                matrix = row.reshape(4, 4)
                assert np.all(matrix[count:, :] == 0.0)
                assert np.all(matrix[:, count:] == 0.0)
                if permutation == "sorted_l2":
                    norms = np.linalg.norm(matrix[:count, :count], axis=1)
                    assert np.all(norms[:-1] >= norms[1:])
    finally:
        cpu.close()
        gpu.close()


@pytest.mark.gpu
def test_cuda_matrix_can_be_cancelled_from_another_python_thread() -> None:
    """The device synchronize seam must not monopolize Python's GIL."""

    load_cuda_for_tests()
    atoms_per_structure = 64
    structures = 2048
    one_structure_positions = np.zeros((atoms_per_structure, 3), dtype=np.float64)
    one_structure_positions[:, 0] = np.arange(atoms_per_structure, dtype=np.float64) * 0.5
    batch = StructureBatch(
        np.tile(np.ones(atoms_per_structure, dtype=np.int32), structures),
        np.tile(one_structure_positions, (structures, 1)),
        np.tile(np.eye(3, dtype=np.float64) * 100.0, (structures, 1, 1)),
        np.zeros((structures, 3), dtype=np.int32),
        np.arange(structures + 1, dtype=np.int64) * atoms_per_structure,
        tuple(str(index) for index in range(structures)),
    )

    class TimedControl(ComputeControl):
        def __init__(self) -> None:
            super().__init__()
            self.ready = threading.Event()
            self.cancelled_at: float | None = None

        def reset(self, total: int) -> None:
            super().reset(total)
            self.ready.set()

        def cancel(self) -> None:
            self.cancelled_at = time.monotonic()
            super().cancel()

    control = TimedControl()
    descriptor = CoulombMatrix(
        n_atoms_max=atoms_per_structure,
        permutation="none",
        execution=ExecutionOptions(device="cuda"),
    )

    def cancel_later() -> None:
        control.ready.wait(5)
        time.sleep(0.01)
        control.cancel()

    cancel_thread = threading.Thread(target=cancel_later)
    cancel_thread.start()
    finished_at = time.monotonic()
    try:
        try:
            with pytest.raises(CancelledError):
                descriptor.compute(batch, control=control)
        except MDescriptorError as error:
            if error.code == "device_unavailable":
                # A CUDA plugin can be importable while the host still has no
                # usable driver/device.  Keep this optional test consistent
                # with the rest of the GPU suite in that environment.
                control.ready.set()
                pytest.skip(str(error))
            raise
        finished_at = time.monotonic()
    finally:
        cancel_thread.join(timeout=5)
        descriptor.close()

    assert control.cancelled_at is not None
    assert control.cancelled_at < finished_at


@pytest.mark.gpu
@pytest.mark.parametrize(("descriptor_type", "parameters"), MATRIX_DESCRIPTORS)
def test_cuda_dynamic_matrix_width_has_uniform_empty_frame_semantics(
    descriptor_type: type[object], parameters: dict[str, float]
) -> None:
    """Dynamic padding handles all-empty, mixed, and widening batches alike."""

    load_cuda_for_tests()
    empty = StructureBatch(
        np.empty(0, dtype=np.int32),
        np.empty((0, 3), dtype=np.float64),
        np.stack([np.eye(3), np.eye(3)]),
        np.ones((2, 3), dtype=np.int32),
        np.array([0, 0, 0], dtype=np.int64),
        ("empty-0", "empty-1"),
    )
    mixed = StructureBatch(
        np.array([1], dtype=np.int32),
        np.array([[0.0, 0.0, 0.0]], dtype=np.float64),
        np.stack([np.eye(3), np.eye(3)]),
        np.ones((2, 3), dtype=np.int32),
        np.array([0, 1, 1], dtype=np.int64),
        ("one", "empty"),
    )
    wider = StructureBatch(
        np.array([1, 1], dtype=np.int32),
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64),
        np.eye(3, dtype=np.float64)[None, :, :],
        np.ones((1, 3), dtype=np.int32),
        np.array([0, 2], dtype=np.int64),
        ("two",),
    )
    descriptor = descriptor_type(
        permutation="none", **parameters, execution=ExecutionOptions(device="cuda")
    )
    control = ComputeControl()
    try:
        empty_result = descriptor.compute(empty, control=control)
        assert empty_result.values.shape == (2, 0)
        assert empty_result.labels == ()
        assert control.total() == 2
        assert control.completed() == 2

        mixed_result = descriptor.compute(mixed)
        assert mixed_result.values.shape == (2, 1)
        np.testing.assert_array_equal(mixed_result.values[1], 0.0)

        wider_result = descriptor.compute(wider)
        assert wider_result.values.shape == (1, 4)
        assert descriptor.feature_count == 4

        smaller_result = descriptor.compute(mixed)
        assert smaller_result.values.shape == (2, 1)
        assert descriptor.feature_count == 1

        empty_again = descriptor.compute(empty)
        assert empty_again.values.shape == (2, 0)
        assert descriptor.feature_count == 0
    finally:
        descriptor.close()
