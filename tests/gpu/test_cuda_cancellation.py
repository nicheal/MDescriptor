"""CUDA cancellation exception and reuse regressions."""

from __future__ import annotations

import importlib
import importlib.machinery
import importlib.util
import multiprocessing as mp
import os
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from tests._cuda import load_cuda_for_tests

from mdescriptor import CancelledError, ComputeControl, StructureBatch
from mdescriptor.core.backends import CudaBackend

_WORKER_TIMEOUT = 30.0


def _load_matching_native() -> None:
    configured = os.environ.get("MDESCRIPTOR_NATIVE_PLUGIN_DIR")
    if not configured:
        return
    directory = Path(configured).expanduser().resolve()
    extension = next(
        (
            directory / f"_native{suffix}"
            for suffix in importlib.machinery.EXTENSION_SUFFIXES
            if (directory / f"_native{suffix}").is_file()
        ),
        None,
    )
    if extension is None:
        raise AssertionError(f"matching _native extension is missing from {directory}")
    loaded = sys.modules.get("mdescriptor._native")
    if loaded is not None and Path(loaded.__file__).resolve() == extension:
        return
    package = importlib.import_module("mdescriptor")
    spec = importlib.util.spec_from_file_location("mdescriptor._native", extension)
    if spec is None or spec.loader is None:
        raise AssertionError(f"cannot load native extension {extension}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["mdescriptor._native"] = module
    spec.loader.exec_module(module)
    package.__dict__["_native"] = module


def _batch() -> StructureBatch:
    return StructureBatch(
        np.asarray([1], dtype=np.int32),
        np.asarray([[0.0, 0.0, 0.0]], dtype=np.float64),
        np.zeros((1, 3, 3), dtype=np.float64),
        np.zeros((1, 3), dtype=np.int32),
        np.asarray([0, 1], dtype=np.int64),
        ("cuda-cancel",),
    )


class _CancelOnNativeCheck:
    """Allow the Python precheck, then cancel at the native check seam."""

    def __init__(self, cancel_at: int) -> None:
        self.cancel_at = cancel_at
        self.calls = 0
        self.completed = 0

    def reset(self, total: int) -> None:
        assert total == 1

    def cancelled(self) -> bool:
        self.calls += 1
        return self.calls >= self.cancel_at

    def mark_completed(self) -> None:
        self.completed += 1


def _cancellation_worker() -> None:
    _load_matching_native()
    load_cuda_for_tests()
    cuda = importlib.import_module("mdescriptor._cuda")
    module_path = Path(cuda.__file__).resolve()
    print(f"cuda_module={module_path}")
    assert hasattr(cuda, "CudaCancelledError")

    batch = _batch()
    for cancel_at in (2, 3):
        backend = CudaBackend(
            "AtomicComposition",
            {"species": [1], "per_system": False},
        )
        control = _CancelOnNativeCheck(cancel_at)
        try:
            with pytest.raises(CancelledError):
                backend.compute(batch, control=control)
            assert control.calls >= cancel_at

            result = backend.compute(batch, control=ComputeControl())
            np.testing.assert_array_equal(result.values, [[1.0]])
        finally:
            backend.close()
            backend.close()


def _run_isolated(worker: Callable[[], None]) -> None:
    process = mp.get_context("spawn").Process(target=worker)
    process.start()
    process.join(_WORKER_TIMEOUT)
    if process.is_alive():
        process.terminate()
        process.join(5.0)
        pytest.fail(f"CUDA cancellation worker exceeded {_WORKER_TIMEOUT:.0f}s")
    assert process.exitcode == 0


@pytest.mark.gpu
def test_cuda_cancellation_maps_native_type_and_allows_reuse() -> None:
    _load_matching_native()
    load_cuda_for_tests()
    _run_isolated(_cancellation_worker)
