"""CUDA cancellation mapping keeps native exception identity."""

from __future__ import annotations

import sys
from types import ModuleType

import numpy as np
import pytest

from mdescriptor import CancelledError, MDescriptorError, StructureBatch
from mdescriptor.core.backends import CudaBackend
from mdescriptor.core.errors import is_cuda_cancelled_error


def _batch() -> StructureBatch:
    return StructureBatch(
        np.asarray([1], dtype=np.int32),
        np.asarray([[0.0, 0.0, 0.0]], dtype=np.float64),
        np.zeros((1, 3, 3), dtype=np.float64),
        np.zeros((1, 3), dtype=np.int32),
        np.asarray([0, 1], dtype=np.int64),
        ("cancel",),
    )


class _FakeCudaCancelledError(RuntimeError):
    pass


class _Raises:
    feature_count = 1

    def __init__(self, error: BaseException) -> None:
        self.error = error

    def compute(self, batch: StructureBatch, control: object) -> None:
        del batch, control
        raise self.error


def _fake_cuda_module() -> ModuleType:
    module = ModuleType("mdescriptor._cuda")
    module.CudaCancelledError = _FakeCudaCancelledError
    return module


def test_cuda_native_cancel_type_maps_without_message_matching(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "mdescriptor._cuda", _fake_cuda_module())
    native_error = _FakeCudaCancelledError("arbitrary text")
    assert is_cuda_cancelled_error(native_error)

    backend = CudaBackend("AtomicComposition", {"species": [1]})
    backend._implementation = _Raises(native_error)
    with pytest.raises(CancelledError):
        backend.compute(_batch())


def test_runtime_error_containing_cancel_stays_backend_error(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "mdescriptor._cuda", _fake_cuda_module())
    backend = CudaBackend("AtomicComposition", {"species": [1]})
    backend._implementation = _Raises(RuntimeError("descriptor computation cancelled"))

    with pytest.raises(MDescriptorError) as caught:
        backend.compute(_batch())
    assert caught.value.code == "backend_error"
