"""Regression tests for recoverable optional Python runtime paths."""

from __future__ import annotations

import importlib.machinery
import importlib.util
import sys
import threading
from types import SimpleNamespace

import pytest

import mdescriptor
from mdescriptor import Descriptor, _cuda_loader, _runtime
from mdescriptor.descriptors._kernels import mtp as mtp_module
from mdescriptor.descriptors._kernels._base import _Kernel


def test_cuda_loader_rolls_back_failed_candidate_path(monkeypatch, tmp_path) -> None:
    candidate = tmp_path / "cuda"
    candidate.mkdir()
    extension = candidate / f"_cuda{importlib.machinery.EXTENSION_SUFFIXES[0]}"
    extension.touch()
    original_path = list(mdescriptor.__path__)
    import_calls: list[str] = []

    class Loader:
        def exec_module(self, module):
            raise ImportError("plugin dependency is unavailable")

    spec = SimpleNamespace(loader=Loader())
    loaded = SimpleNamespace(__package__="mdescriptor")

    def fail_import(name: str):
        import_calls.append(name)
        raise ImportError("plugin dependency is unavailable")

    monkeypatch.delenv("MDESCRIPTOR_CUDA_PLUGIN_DIR", raising=False)
    monkeypatch.delitem(sys.modules, "mdescriptor._cuda", raising=False)
    monkeypatch.delattr(mdescriptor, "_cuda", raising=False)
    monkeypatch.setattr(
        _cuda_loader.importlib.util, "spec_from_file_location", lambda name, path: spec
    )
    monkeypatch.setattr(
        _cuda_loader.importlib.util, "module_from_spec", lambda value: loaded
    )
    monkeypatch.setattr(_cuda_loader.importlib, "import_module", fail_import)

    with pytest.raises(_cuda_loader.CudaPluginUnavailable):
        _cuda_loader.load_cuda_plugin(candidate)

    assert import_calls == ["mdescriptor._cuda"]
    assert "mdescriptor._cuda" not in sys.modules
    assert not hasattr(mdescriptor, "_cuda")
    assert list(mdescriptor.__path__) == original_path
    assert str(candidate.resolve()) not in mdescriptor.__path__


def test_kernel_close_releases_native_reference() -> None:
    class Native:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    native = Native()
    kernel = _Kernel()
    kernel._native = native

    kernel.close()

    assert native.closed
    assert kernel._native is None


def test_cuda_loader_loads_the_explicit_candidate_file(monkeypatch, tmp_path) -> None:
    candidate = tmp_path / "cuda"
    candidate.mkdir()
    extension = candidate / f"_cuda{importlib.machinery.EXTENSION_SUFFIXES[0]}"
    extension.touch()
    original_path = list(mdescriptor.__path__)

    class Loader:
        def create_module(self, spec):
            return None

        def exec_module(self, module):
            assert module.__package__ == "mdescriptor"
            module.__file__ = str(extension)

            def create_backend(name, options):
                return name, options

            module.create_backend = create_backend

    def make_spec(name, path):
        assert name == "mdescriptor._cuda"
        assert path == extension
        return importlib.util.spec_from_loader(name, Loader())

    monkeypatch.delenv("MDESCRIPTOR_CUDA_PLUGIN_DIR", raising=False)
    monkeypatch.delitem(sys.modules, "mdescriptor._cuda", raising=False)
    monkeypatch.delattr(mdescriptor, "_cuda", raising=False)
    monkeypatch.setattr(_cuda_loader.importlib.util, "spec_from_file_location", make_spec)
    monkeypatch.setattr(
        _cuda_loader.importlib,
        "import_module",
        lambda name: pytest.fail(f"candidate import fell through to {name}"),
    )

    try:
        assert _cuda_loader.load_cuda_plugin(candidate) == candidate.resolve()
        assert sys.modules["mdescriptor._cuda"].__package__ == "mdescriptor"
        assert mdescriptor._cuda is sys.modules["mdescriptor._cuda"]
    finally:
        sys.modules.pop("mdescriptor._cuda", None)
        mdescriptor.__dict__.pop("_cuda", None)
        mdescriptor.__path__[:] = original_path


def test_cuda_loader_ignores_other_abis_and_unrelated_extensions(
    monkeypatch, tmp_path
) -> None:
    candidate = tmp_path / "cuda"
    candidate.mkdir()
    extension = candidate / f"_cuda{importlib.machinery.EXTENSION_SUFFIXES[0]}"
    extension.touch()
    (candidate / "_cuda.cpython-310-x86_64-linux-gnu.so").touch()
    (candidate / "_cuda_test.so").touch()
    loaded = SimpleNamespace(__file__=str(extension), __package__="mdescriptor")

    class Loader:
        def exec_module(self, module):
            return None

    def make_spec(name, path):
        assert name == "mdescriptor._cuda"
        assert path == extension
        return SimpleNamespace(loader=Loader())

    monkeypatch.delenv("MDESCRIPTOR_CUDA_PLUGIN_DIR", raising=False)
    monkeypatch.delitem(sys.modules, "mdescriptor._cuda", raising=False)
    monkeypatch.delattr(mdescriptor, "_cuda", raising=False)
    monkeypatch.setattr(_cuda_loader.importlib.util, "spec_from_file_location", make_spec)
    monkeypatch.setattr(_cuda_loader.importlib.util, "module_from_spec", lambda spec: loaded)

    try:
        assert _cuda_loader.load_cuda_plugin(candidate) == candidate.resolve()
    finally:
        sys.modules.pop("mdescriptor._cuda", None)
        mdescriptor.__dict__.pop("_cuda", None)


def test_cuda_factory_retries_after_a_transient_import_failure(monkeypatch) -> None:
    def factory(name, options):
        return name, options

    module = SimpleNamespace(create_backend=factory)
    imports = iter([ImportError("plugin is not built yet"), module])

    def import_module(name: str):
        assert name == "mdescriptor._cuda"
        value = next(imports)
        if isinstance(value, BaseException):
            raise value
        return value

    monkeypatch.setattr(_runtime, "_CUDA_FACTORY", None)
    monkeypatch.setattr(_runtime.importlib, "import_module", import_module)

    with pytest.raises(ImportError, match="not built"):
        _runtime._cuda_factory()

    assert _runtime._cuda_factory() is factory


def test_mtp_native_construction_stays_inside_initialization_lock(monkeypatch) -> None:
    class TrackingLock:
        def __init__(self):
            self._lock = threading.Lock()
            self.acquisitions = 0
            self.second_acquired = threading.Event()

        def __enter__(self):
            self._lock.acquire()
            self.acquisitions += 1
            if self.acquisitions == 2:
                self.second_acquired.set()
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            self._lock.release()

    lock = TrackingLock()
    first_calculator_entered = threading.Event()
    release_calculator = threading.Event()
    calculator_calls = 0
    calculator_lock = threading.Lock()

    class Options:
        pass

    def create_calculator(options):
        assert options.species == [1]
        nonlocal calculator_calls
        with calculator_lock:
            calculator_calls += 1
            call = calculator_calls
        if call == 1:
            first_calculator_entered.set()
            assert release_calculator.wait(2)
        return SimpleNamespace(feature_count=3)

    fake_cpp = SimpleNamespace(MtpOptions=Options, MtpCalculator=create_calculator)
    monkeypatch.setattr(mtp_module, "_cpp", fake_cpp)

    kernel = mtp_module.MtpKernel(species=[1])
    kernel._init_lock = lock
    first = threading.Thread(target=kernel._create_native)
    second = threading.Thread(target=kernel._create_native)
    second_started = False
    first.start()
    try:
        assert first_calculator_entered.wait(1)
        second.start()
        second_started = True
        assert not lock.second_acquired.wait(0.1)
    finally:
        release_calculator.set()
        first.join(timeout=2)
        if second_started:
            second.join(timeout=2)

    assert not first.is_alive()
    assert not second.is_alive()
    assert calculator_calls == 1
    assert kernel._feature_count == 3
    assert kernel._native is not None


def test_missing_ase_remains_an_import_error_at_descriptor_boundary(monkeypatch) -> None:
    class Probe(Descriptor):
        name = "probe"

        def _compute_batch(self, batch, *, control=None):
            raise AssertionError("input conversion should fail first")

    monkeypatch.setitem(sys.modules, "ase", None)
    with pytest.raises(ImportError, match="ASE is required") as caught:
        Probe().compute(object())
    assert not isinstance(caught.value, mdescriptor.DescriptorInputError)
