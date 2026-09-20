"""The opt-in GPU entry point must exercise the selected current binaries."""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import tests._cuda as cuda_tests
from tests._cuda import load_cuda_for_tests

ROOT = Path(__file__).parents[2]


@pytest.mark.gpu
def test_required_cuda_build_is_loaded_and_usable() -> None:
    """Fail the strict entry point instead of turning an unavailable GPU green."""

    load_cuda_for_tests()

    cuda = importlib.import_module("mdescriptor._cuda")
    cuda_location = getattr(cuda, "__file__", None)
    assert cuda_location
    cuda_path = Path(cuda_location).resolve()
    configured_cuda = os.environ.get("MDESCRIPTOR_CUDA_PLUGIN_DIR")
    if configured_cuda:
        assert cuda_path.parent == Path(configured_cuda).expanduser().resolve()
    assert cuda.MODEL_SNAPSHOT_ABI == 1

    native = importlib.import_module("mdescriptor._native")
    native_location = getattr(native, "__file__", None)
    assert native_location
    native_path = Path(native_location).resolve()
    configured_native = os.environ.get(
        "MDESCRIPTOR_EXPECTED_NATIVE_PLUGIN_DIR",
        os.environ.get("MDESCRIPTOR_NATIVE_PLUGIN_DIR"),
    )
    if configured_native:
        expected_native = Path(configured_native).expanduser().resolve()
        assert native_path.parent == expected_native
    assert hasattr(native.NepOptions(), "model_data")
    assert hasattr(native.MtpOptions(), "model_data")
    print(f"native={native_path}")
    print(f"cuda={cuda_path}")


def _run_simulated_gpu_skip(tmp_path: Path, *, require_gpu: bool) -> subprocess.CompletedProcess[str]:
    test_file = tmp_path / "test_simulated_gpu_skip.py"
    test_file.write_text(
        "import pytest\n"
        "@pytest.mark.gpu\n"
        "def test_simulated_device_loss():\n"
        "    pytest.skip('simulated device unavailable')\n",
        encoding="utf-8",
    )
    environment = os.environ.copy()
    if require_gpu:
        environment["MDESCRIPTOR_REQUIRE_GPU"] = "1"
    else:
        environment.pop("MDESCRIPTOR_REQUIRE_GPU", None)
    source_path = str(ROOT)
    environment["PYTHONPATH"] = source_path + os.pathsep + environment.get("PYTHONPATH", "")
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "--strict-markers",
            "--import-mode=importlib",
            "-c",
            str(ROOT / "pyproject.toml"),
            "-p",
            "tests.conftest",
            str(test_file),
        ],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


def test_strict_mode_rejects_a_gpu_skip_without_hardware(tmp_path: Path) -> None:
    completed = _run_simulated_gpu_skip(tmp_path, require_gpu=True)
    output = completed.stdout + completed.stderr
    assert completed.returncode != 0
    assert "strict GPU gate rejected skip" in output


def test_cpu_mode_keeps_a_gpu_skip_optional(tmp_path: Path) -> None:
    completed = _run_simulated_gpu_skip(tmp_path, require_gpu=False)
    output = completed.stdout + completed.stderr
    assert completed.returncode == 0
    assert "1 skipped" in output


def test_strict_validation_rejects_fallback_after_a_bad_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    actual_directory = tmp_path / "installed"
    configured_file = tmp_path / "bad-cuda.so"
    configured_file.write_text("not an extension", encoding="utf-8")
    fake_cuda = SimpleNamespace(
        __file__=str(actual_directory / "_cuda.so"), MODEL_SNAPSHOT_ABI=1
    )

    class SnapshotOptions:
        model_data = b""

    fake_native = SimpleNamespace(
        __file__=str(actual_directory / "_native.so"),
        NepOptions=SnapshotOptions,
        MtpOptions=SnapshotOptions,
    )
    monkeypatch.setitem(sys.modules, "mdescriptor._cuda", fake_cuda)
    monkeypatch.setitem(sys.modules, "mdescriptor._native", fake_native)
    monkeypatch.setenv("MDESCRIPTOR_CUDA_PLUGIN_DIR", str(configured_file))

    with pytest.raises(pytest.fail.Exception, match="CUDA plugin loaded from"):
        cuda_tests._validate_snapshot_extensions(actual_directory, require_gpu=True)
