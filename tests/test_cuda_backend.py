"""Contract tests for the optional CUDA backend seam."""

from __future__ import annotations

from pathlib import Path

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
from mdescriptor.descriptors import DPA4, DPA4C, SOAP, NeighborList

ROOT = Path(__file__).parents[1]


def _batch() -> StructureBatch:
    return StructureBatch.from_ase(
        [Atoms("H2", positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])]
    )


def test_cuda_descriptors_declare_the_device_in_static_metadata() -> None:
    cuda_names = {
        "SOAP",
        "SOAPTurbo",
        "ACSF",
        "ACE",
        "CoulombMatrix",
        "SineMatrix",
        "EwaldSumMatrix",
        "MBTR",
        "LMBTR",
        "ValleOganov",
        "AtomicComposition",
        "NeighborList",
        "SortedDistances",
        "SphericalExpansion",
        "SphericalExpansionByPair",
        "SoapRadialSpectrum",
        "SoapPowerSpectrum",
        "LodeSphericalExpansion",
        "EAD",
        "SO3",
        "SO4",
        "SNAP",
        "LBispectrum",
        "MTP",
        "C00PSMLFF",
        "NEP",
        "DPA4",
        "DPA4C",
    }
    for name in mdescriptor.list_descriptors():
        devices = mdescriptor.describe_descriptor(name)["execution"]["devices"]
        assert devices == (["cpu", "cuda"] if name in cuda_names else ["cpu"])


def test_execution_options_accept_only_the_frozen_device_tokens() -> None:
    with pytest.raises(DescriptorConfigError) as caught:
        ExecutionOptions(device="cuda:0")
    assert caught.value.code == "invalid_device"


def test_cuda_graph_migration_has_no_host_graph_or_pair_reorder() -> None:
    """The migrated public paths must not reintroduce their old CPU seams."""

    backend = (ROOT / "cpp/cuda/src/backend.cu").read_text(encoding="utf-8")
    extended = (ROOT / "cpp/cuda/src/extended_descriptors.cu").read_text(encoding="utf-8")
    graph = (ROOT / "cpp/cuda/src/neighbor_graph.cu").read_text(encoding="utf-8")
    assert "mdescriptor::build_neighbor_graph" not in backend
    assert "cpu_pair_order" not in extended
    assert "build_nep_image_neighbors_kernel" not in graph
    assert "build_nep_images" not in graph
    assert "NeighborGraphOrdering::Canonical" in backend
    assert "NeighborGraphOrdering::Canonical" in extended
    assert "count_filtered_neighbor_records" in backend


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


@pytest.mark.parametrize("descriptor_type", [DPA4, DPA4C])
@pytest.mark.model
def test_dpa_cuda_payload_is_private_and_keeps_numpy_tensors(descriptor_type) -> None:
    descriptor = descriptor_type(execution=ExecutionOptions(device="cuda"))
    try:
        payload = descriptor._backend.options["_cuda_payload"]
        assert set(("feature_count", "labels", "type_numbers")) <= set(payload)
        assert payload["feature_count"] == descriptor.feature_count
        assert payload["labels"] == tuple(descriptor._kernel.options["_cuda_payload"]["labels"])
        assert isinstance(payload["type_numbers"], np.ndarray)
        assert payload["type_numbers"].dtype == np.int32
        assert isinstance(payload["model"], dict)
        assert any(
            isinstance(value, np.ndarray)
            for value in payload["model"].values()
            if not isinstance(value, list)
        )
        assert "_cuda_payload" not in descriptor.configuration.to_dict()
        assert "_cuda_payload" not in descriptor.metadata
    finally:
        descriptor.close()


def test_extended_descriptor_cuda_path_is_lazy(monkeypatch) -> None:
    calls = []

    def factory(name, options):
        calls.append((name, options))
        raise AssertionError("the CUDA plugin must remain lazy during construction")

    monkeypatch.setattr(_runtime, "create_cuda_backend", factory)
    descriptor = SOAP(species=[1], execution=ExecutionOptions(device="cuda"))
    try:
        assert calls == []
        assert descriptor._backend.options["_cuda_feature_count"] > 0
    finally:
        descriptor.close()
