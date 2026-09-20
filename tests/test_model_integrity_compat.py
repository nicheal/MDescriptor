"""Compatibility failures for extensions predating immutable model snapshots."""

from types import SimpleNamespace

import pytest

from mdescriptor import _runtime
from mdescriptor.core.errors import ModelLoadError
from mdescriptor.descriptors._kernels import mtp as mtp_module
from mdescriptor.descriptors._kernels import nep as nep_module


class _OldNepOptions:
    model_path = ""
    model_digest = ""
    num_threads = 0

    def __setattr__(self, name, value):
        if name == "model_data":
            raise AttributeError(name)
        object.__setattr__(self, name, value)


class _OldMtpOptions:
    species = []
    potential_path = ""
    model_digest = ""
    min_dist = 0.0
    max_dist = 5.0
    radial_basis_size = 4
    radial_funcs_count = 1
    max_rank = 2
    num_threads = 0

    def __setattr__(self, name, value):
        if name == "model_data":
            raise AttributeError(name)
        object.__setattr__(self, name, value)


def test_old_cpu_extension_rejects_a_model_snapshot(monkeypatch) -> None:
    monkeypatch.setattr(nep_module, "_cpp", SimpleNamespace(NepOptions=_OldNepOptions))

    with pytest.raises(ModelLoadError, match="rebuild MDescriptor"):
        nep_module.NepKernel(model_path="missing.nep", model_data=b"snapshot")


def test_old_mtp_extension_rejects_a_model_snapshot(monkeypatch) -> None:
    monkeypatch.setattr(mtp_module, "_cpp", SimpleNamespace(MtpOptions=_OldMtpOptions))

    with pytest.raises(ModelLoadError, match="rebuild MDescriptor"):
        mtp_module.MtpKernel(
            species=[1],
            model_path="missing.mtp",
            model_data=b"snapshot",
        )


def test_old_cuda_plugin_rejects_a_model_snapshot(monkeypatch) -> None:
    calls = []

    def factory(name, options):
        calls.append((name, options))
        return object()

    module = SimpleNamespace(create_backend=factory)
    monkeypatch.setattr(_runtime, "_CUDA_FACTORY", None)
    monkeypatch.setattr(_runtime, "_CUDA_MODEL_SNAPSHOT_ABI", None)
    monkeypatch.setattr(_runtime.importlib, "import_module", lambda name: module)

    with pytest.raises(ModelLoadError, match="rebuild MDescriptor"):
        _runtime.create_cuda_backend("NEP", {"model_data": b"snapshot"})
    assert calls == []
