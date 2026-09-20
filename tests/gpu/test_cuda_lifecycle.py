"""CUDA backend operation and teardown race regressions."""

from __future__ import annotations

import multiprocessing as mp
import threading
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from tests._cuda import load_cuda_for_tests

from mdescriptor import ComputeControl, StructureBatch
from mdescriptor._cuda_loader import load_cuda_plugin

_WORKER_TIMEOUT = 20.0


def _batch(atom_count: int = 262_144) -> StructureBatch:
    positions = np.zeros((atom_count, 3), dtype=np.float64)
    positions[:, 0] = np.arange(atom_count, dtype=np.float64) * 2.0
    return StructureBatch(
        np.ones(atom_count, dtype=np.int32),
        positions,
        np.zeros((1, 3, 3), dtype=np.float64),
        np.zeros((1, 3), dtype=np.int32),
        np.array([0, atom_count], dtype=np.int64),
        ("lifecycle",),
    )


def _native_backend_and_batch() -> tuple[object, StructureBatch]:
    load_cuda_plugin(Path(__file__).parents[2] / "build-cuda")
    from mdescriptor._cuda import CudaBackend

    return CudaBackend(
        "AtomicComposition",
        {"species": [1], "per_system": False},
    ), _batch()


def _assert_composition_result(result: object, batch: StructureBatch) -> None:
    values = np.asarray(result["values"])
    assert values.shape == (batch.atoms, 1)
    np.testing.assert_array_equal(values, 1.0)


def _compute_compute_worker() -> None:
    backend, batch = _native_backend_and_batch()
    barrier = threading.Barrier(3)
    failures: list[BaseException] = []

    def compute() -> None:
        try:
            barrier.wait(timeout=5.0)
            result = backend.compute(batch, ComputeControl())
            _assert_composition_result(result, batch)
        except BaseException as error:  # pragma: no cover - runs in a child
            failures.append(error)

    workers = [threading.Thread(target=compute, daemon=True) for _ in range(2)]
    for worker in workers:
        worker.start()
    barrier.wait(timeout=5.0)
    deadline = time.monotonic() + 12.0
    for worker in workers:
        worker.join(max(0.0, deadline - time.monotonic()))
    assert not any(worker.is_alive() for worker in workers), "concurrent compute hung"
    if failures:
        raise failures[0]
    backend.close()


def _compute_close_worker() -> None:
    backend, batch = _native_backend_and_batch()
    failures: list[BaseException] = []
    close_failures: list[BaseException] = []

    class ComputeGate:
        def __init__(self) -> None:
            self.reset_seen = threading.Event()
            self.allow_compute = threading.Event()
            self.completed = 0

        def reset(self, total: int) -> None:
            assert total == 1
            self.reset_seen.set()
            assert self.allow_compute.wait(timeout=10.0)

        def cancelled(self) -> bool:
            return False

        def mark_completed(self) -> None:
            self.completed += 1

    control = ComputeGate()

    def compute() -> None:
        try:
            result = backend.compute(batch, control)
            _assert_composition_result(result, batch)
        except BaseException as error:  # pragma: no cover - runs in a child
            failures.append(error)

    compute_thread = threading.Thread(target=compute, daemon=True)
    compute_thread.start()
    assert control.reset_seen.wait(timeout=5.0)

    close_done = threading.Event()

    def close() -> None:
        try:
            backend.close()
        except BaseException as error:  # pragma: no cover - runs in a child
            close_failures.append(error)
        finally:
            close_done.set()

    close_thread = threading.Thread(target=close, daemon=True)
    close_thread.start()
    time.sleep(0.05)
    assert not close_done.is_set(), "close overtook an active compute"

    control.allow_compute.set()
    compute_thread.join(12.0)
    close_thread.join(12.0)
    assert not compute_thread.is_alive(), "compute did not finish after close was queued"
    assert not close_thread.is_alive(), "close hung behind concurrent compute"
    if failures:
        raise failures[0]
    if close_failures:
        raise close_failures[0]
    assert control.completed == 1
    assert close_done.is_set()

    try:
        backend.compute(batch, ComputeControl())
    except RuntimeError as error:
        assert str(error) == "CUDA backend is closed"
    else:
        raise AssertionError("compute succeeded after close")
    backend.close()


def _run_isolated(worker: Callable[[], None]) -> None:
    context = mp.get_context("spawn")
    process = context.Process(target=worker)
    process.start()
    process.join(_WORKER_TIMEOUT)
    if process.is_alive():
        process.terminate()
        process.join(5.0)
        pytest.fail(f"CUDA lifecycle worker exceeded {_WORKER_TIMEOUT:.0f}s")
    assert process.exitcode == 0


@pytest.mark.gpu
def test_cuda_backend_serializes_concurrent_compute_with_timeout() -> None:
    load_cuda_for_tests()
    _run_isolated(_compute_compute_worker)


@pytest.mark.gpu
def test_cuda_backend_waits_for_compute_before_close_with_timeout() -> None:
    load_cuda_for_tests()
    _run_isolated(_compute_close_worker)
