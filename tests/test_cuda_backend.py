"""Contract tests for the optional CUDA backend seam."""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms

import mdescriptor
from mdescriptor import (
    ComputeControl,
    DescriptorConfigError,
    ExecutionOptions,
    StructureBatch,
    _runtime,
)
from mdescriptor.descriptors import SOAP, NeighborList


def _batch() -> StructureBatch:
    return StructureBatch.from_ase(
        [Atoms("H2", positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])]
    )


def test_first_cuda_descriptors_declare_the_device_in_static_metadata() -> None:
    cuda_names = {
        "NeighborList",
        "SphericalExpansion",
        "SoapRadialSpectrum",
        "SoapPowerSpectrum",
    }
    for name in mdescriptor.list_descriptors():
        devices = mdescriptor.describe_descriptor(name)["execution"]["devices"]
        assert devices == (["cpu", "cuda"] if name in cuda_names else ["cpu"])


def test_execution_options_accept_only_the_frozen_device_tokens() -> None:
    with pytest.raises(DescriptorConfigError) as caught:
        ExecutionOptions(device="cuda:0")
    assert caught.value.code == "invalid_device"


def test_cuda_plugin_is_lazy_and_receives_public_control(monkeypatch) -> None:
    calls: list[tuple[str, dict]] = []

    class FakeBackend:
        feature_count = 4

        def compute(self, batch, control):
            assert isinstance(batch, StructureBatch)
            assert isinstance(control, ComputeControl)
            assert control.total() == 0
            calls.append(("compute", {}))
            return {
                "values": np.zeros((2, 4), dtype=np.float64),
                "level": "pair",
                "row_offsets": np.array([0, 2], dtype=np.int64),
                "pair_records": np.array(
                    [[0, 1, 0, 0, 0], [1, 0, 0, 0, 0]], dtype=np.float64
                ),
                "labels": ("dx", "dy", "dz", "distance"),
                "metadata": {"descriptor": "NeighborList", "backend": "fake-cuda"},
            }

        def metadata(self):
            return {"descriptor": "NeighborList", "backend": "fake-cuda"}

        def close(self):
            calls.append(("close", {}))

    def factory(name, options):
        calls.append((name, options))
        return FakeBackend()

    monkeypatch.setattr(_runtime, "_CUDA_FACTORY", None)
    monkeypatch.setattr(_runtime, "_CUDA_LOAD_ERROR", None)
    monkeypatch.setattr(_runtime, "create_cuda_backend", factory)
    descriptor = NeighborList(
        cutoff=3.0,
        execution=ExecutionOptions(device="cuda", num_threads=3),
    )
    try:
        assert calls == []
        assert descriptor.metadata["execution"] == {"device": "cuda", "num_threads": 3}
        control = ComputeControl()
        result = descriptor.compute(_batch(), control=control)
        assert calls[0][0] == "NeighborList"
        assert calls[0][1]["execution"] == {"device": "cuda", "num_threads": 3}
        assert result.metadata["execution"] == {"device": "cuda", "num_threads": 3}
        assert result.samples.shape == (2, 6)
        assert control.total() == 0
    finally:
        descriptor.close()
    assert calls[-1][0] == "close"


def test_cpu_only_descriptor_rejects_cuda_before_plugin_load(monkeypatch) -> None:
    called = False

    def factory(name, options):
        nonlocal called
        called = True
        raise AssertionError(f"unexpected CUDA factory call for {name}: {options}")

    monkeypatch.setattr(_runtime, "create_cuda_backend", factory)
    with pytest.raises(DescriptorConfigError) as caught:
        SOAP(species=[1], execution=ExecutionOptions(device="cuda"))
    assert caught.value.code == "unsupported_device"
    assert not called
