"""Regression coverage for immutable model snapshots and native identity."""

from __future__ import annotations

import hashlib
import importlib
import importlib.machinery
import multiprocessing as mp
import os
import subprocess
import sys
from pathlib import Path

import pytest

from mdescriptor.models import ModelResolver

_ROOT = Path(__file__).parents[1]
_NATIVE_PLUGIN_ENV = "MDESCRIPTOR_NATIVE_PLUGIN_DIR"
_CUDA_PLUGIN_ENV = "MDESCRIPTOR_CUDA_PLUGIN_DIR"
_SUBPROCESS_TIMEOUT = 60.0


def _extension_path(value: str | os.PathLike[str], stem: str) -> Path | None:
    """Resolve one exact extension module from an installed or injected path."""

    root = Path(value).expanduser()
    if root.is_file():
        expected = {f"{stem}{suffix}" for suffix in importlib.machinery.EXTENSION_SUFFIXES}
        return root.resolve() if root.name in expected else None
    if not root.is_dir():
        return None
    for suffix in importlib.machinery.EXTENSION_SUFFIXES:
        candidate = root / f"{stem}{suffix}"
        if candidate.is_file():
            return candidate.resolve()
    return None


def _native_module_path() -> Path:
    """Find the native module supplied by the active test environment."""

    configured = os.environ.get(_NATIVE_PLUGIN_ENV)
    configured_name = _NATIVE_PLUGIN_ENV
    if not configured:
        cuda_directory = os.environ.get(_CUDA_PLUGIN_ENV)
        if cuda_directory and _extension_path(cuda_directory, "_native") is not None:
            configured = cuda_directory
            configured_name = _CUDA_PLUGIN_ENV
    if configured:
        path = _extension_path(configured, "_native")
        if path is None:
            pytest.fail(
                f"{configured_name} does not contain a supported _native extension: "
                f"{configured}"
            )
        return path

    try:
        module = importlib.import_module("mdescriptor._native")
    except (ImportError, OSError) as exc:
        pytest.fail(
            "the native extension is unavailable; install the package or set "
            f"{_NATIVE_PLUGIN_ENV} to an injected build directory: {exc}"
        )
    location = getattr(module, "__file__", None)
    if not location:
        pytest.fail("mdescriptor._native has no extension file")
    path = Path(location).resolve()
    expected = {f"_native{suffix}" for suffix in importlib.machinery.EXTENSION_SUFFIXES}
    if path.name not in expected:
        pytest.fail(f"mdescriptor._native is not a supported extension: {path}")
    return path


def _spawn_native_probe(native_path: str, result_queue) -> None:
    """Load the selected extension in a fresh spawn child and report its ABI."""

    import importlib.util

    import mdescriptor

    path = Path(native_path).resolve()
    spec = importlib.util.spec_from_file_location("mdescriptor._native", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["mdescriptor._native"] = module
    spec.loader.exec_module(module)
    mdescriptor.__dict__["_native"] = module
    location = getattr(module, "__file__", None)
    assert location
    actual = Path(location).resolve()
    print(f"spawn-native={actual}")
    result_queue.put(
        (
            str(actual),
            hasattr(module.NepOptions(), "model_data"),
            hasattr(module.MtpOptions(), "model_data"),
        )
    )


def test_spawn_child_uses_the_selected_native_extension() -> None:
    native_path = _native_module_path()
    context = mp.get_context("spawn")
    result_queue = context.Queue()
    process = context.Process(
        target=_spawn_native_probe,
        args=(str(native_path), result_queue),
    )
    process.start()
    process.join(_SUBPROCESS_TIMEOUT)
    if process.is_alive():
        process.terminate()
        process.join(5.0)
        pytest.fail("native spawn probe exceeded the subprocess timeout")
    assert process.exitcode == 0
    actual, nep_snapshot, mtp_snapshot = result_queue.get(timeout=5.0)
    assert actual == str(native_path.resolve())
    assert nep_snapshot and mtp_snapshot


def test_model_resolver_keeps_the_hashed_snapshot(tmp_path: Path) -> None:
    path = tmp_path / "model.bin"
    original = b"model-v1"
    path.write_bytes(original)

    resolved = ModelResolver().resolve(path)
    path.write_bytes(b"model-v2")

    assert resolved.path == path.resolve()
    assert resolved.source == "explicit"
    assert resolved.content == original
    assert resolved.digest == hashlib.sha256(original).hexdigest()


_NATIVE_SCRIPT = r"""
import hashlib
import importlib.util
import sys
from pathlib import Path

import numpy as np
import mdescriptor


def load_extension(module_name, path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get(module_name)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
        package_name, attribute = module_name.rsplit(".", 1)
        setattr(sys.modules[package_name], attribute, module)
    except BaseException:
        if previous is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous
        raise
    return module


native_path = Path(sys.argv[1]).resolve()
mode = sys.argv[2]
path = Path(sys.argv[3])
replacement = Path(sys.argv[4])
cuda_path = Path(sys.argv[5]) if len(sys.argv) > 5 and sys.argv[5] else None
native = load_extension("mdescriptor._native", native_path)
assert Path(native.__file__).resolve() == native_path
for options_name in ("NepOptions", "MtpOptions"):
    options = getattr(native, options_name)()
    assert hasattr(options, "model_data"), (
        f"{native_path} lacks {options_name}.model_data; "
        "the active native extension is older than the snapshot ABI"
    )
print(f"native={native_path}")

from mdescriptor import ExecutionOptions, StructureBatch
from mdescriptor.descriptors import MTP, NEP
from mdescriptor.models.resolver import ModelResolver
from mdescriptor.models.session import clear_loaded_model_cache


def descriptor_batch(kind):
    if kind == "nep":
        return StructureBatch(
            numbers=np.asarray([6, 6], dtype=np.int32),
            positions=np.asarray([[1.0, 1.0, 1.0], [2.0, 1.0, 1.0]]),
            cells=np.asarray([np.eye(3) * 10.0]),
            pbc=np.asarray([[1, 1, 1]], dtype=np.int32),
            offsets=np.asarray([0, 2], dtype=np.int64),
            ids=("nep",),
        )
    if kind == "mtp4":
        numbers = [13, 14]
        positions = [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
        offsets = [0, 2]
    else:
        numbers = [1, 1, 1]
        positions = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        offsets = [0, 3]
    return StructureBatch(
        numbers=np.asarray(numbers, dtype=np.int32),
        positions=np.asarray(positions, dtype=np.float64),
        cells=np.asarray([np.eye(3) * 10.0]),
        pbc=np.asarray([[1, 1, 1]], dtype=np.int32),
        offsets=np.asarray(offsets, dtype=np.int64),
        ids=(kind,),
    )


def compute_descriptor(kind):
    if kind == "nep":
        descriptor = NEP(model=path)
    elif kind == "mtp4":
        descriptor = MTP(species=[13, 14], model=path)
    elif kind == "mtp2":
        descriptor = MTP(species=[1], model=path)
    else:
        raise AssertionError(f"unknown descriptor mode: {kind}")
    try:
        result = descriptor.compute(descriptor_batch(kind))
        return descriptor.feature_count, np.asarray(result.values, dtype=np.float64).copy()
    finally:
        descriptor.close()


if mode in {"nep", "mtp4", "mtp2"}:
    expected_features = {"nep": 1, "mtp4": 5, "mtp2": 3}[mode]
    baseline_count, baseline = compute_descriptor(mode)
    assert baseline_count == expected_features
    assert np.isfinite(baseline).all()
    clear_loaded_model_cache()

    original_resolve = ModelResolver.resolve

    def resolve_then_replace(self, resource):
        resolved = original_resolve(self, resource)
        path.write_bytes(replacement.read_bytes())
        return resolved

    ModelResolver.resolve = resolve_then_replace
    try:
        snapshot_count, snapshot = compute_descriptor(mode)
    finally:
        ModelResolver.resolve = original_resolve
    assert snapshot_count == baseline_count
    assert np.isfinite(snapshot).all()
    np.testing.assert_allclose(snapshot, baseline, rtol=1e-12, atol=1e-12)
    print(
        f"{mode}=shape:{snapshot.shape}, "
        f"checksum:{hashlib.sha256(snapshot.tobytes()).hexdigest()}, "
        f"max_abs_diff:{np.max(np.abs(snapshot - baseline)):.3e}"
    )
elif mode == "cache":
    original = path.read_bytes()
    changed = replacement.read_bytes()

    def make_options(content):
        options = native.NepOptions()
        options.model_path = str(path)
        options.model_digest = "same-stale-digest"
        options.model_data = content
        return options

    first = native.NepCalculator(make_options(original))
    repeat = native.NepCalculator(make_options(original))
    different = native.NepCalculator(make_options(changed))
    try:
        assert list(first.species) == [6]
        assert list(repeat.species) == [6]
        assert list(different.species) == [1]
        assert first.feature_count == repeat.feature_count == different.feature_count
    finally:
        first.close()
        repeat.close()
        different.close()
    print("cache=repeat_same_payload,different_payload_same_digest")
elif mode == "cuda-nep":
    assert cuda_path is not None
    cuda = load_extension("mdescriptor._cuda", cuda_path)
    assert Path(cuda.__file__).resolve() == cuda_path.resolve()
    assert getattr(cuda, "MODEL_SNAPSHOT_ABI", 0) == 1, (
        f"{cuda_path} lacks the immutable model snapshot ABI"
    )
    print(f"cuda={cuda_path.resolve()}")
    descriptor = NEP(model=path, execution=ExecutionOptions(device="cuda"))
    try:
        assert descriptor._backend.options["model_data"] == descriptor.resolved_model.content
        result = descriptor.compute(descriptor_batch("nep"))
        values = np.asarray(result.values, dtype=np.float64)
        assert values.shape == (2, descriptor.feature_count)
        assert np.isfinite(values).all()
        print(f"cuda-nep=shape:{values.shape},max_abs:{np.max(np.abs(values)):.3e}")
    finally:
        descriptor.close()
else:
    raise AssertionError(f"unknown mode: {mode}")
"""


def _run_native_regression(
    mode: str,
    path: Path,
    replacement: Path,
    *,
    cuda_path: Path | None = None,
) -> None:
    native_path = _native_module_path()
    environment = os.environ.copy()
    source_path = str(_ROOT / "src")
    environment["PYTHONPATH"] = source_path + os.pathsep + environment.get("PYTHONPATH", "")
    command = [
        sys.executable,
        "-c",
        _NATIVE_SCRIPT,
        str(native_path),
        mode,
        str(path),
        str(replacement),
        "" if cuda_path is None else str(cuda_path),
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=_ROOT,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
            timeout=_SUBPROCESS_TIMEOUT,
        )
    except subprocess.TimeoutExpired as exc:
        pytest.fail(f"native model regression timed out after {_SUBPROCESS_TIMEOUT:.0f}s: {exc}")
    assert completed.returncode == 0, (
        f"native model regression failed:\nstdout:\n{completed.stdout}\nstderr:\n"
        f"{completed.stderr}"
    )
    assert f"native={native_path.resolve()}" in completed.stdout
    if cuda_path is not None:
        assert f"cuda={cuda_path.resolve()}" in completed.stdout


def _write_nep(path: Path, symbol: str) -> None:
    path.write_text(
        "\n".join(
            [
                f"nep3 1 {symbol}",
                "cutoff 3.0 3.0 8 8",
                "n_max 0 0",
                "basis_size 0 0",
                "l_max 0 0 0",
                "ANN 1 0",
                "0",
                "0",
                "0",
                "0",
                "1",
                "0",
                "1",
            ]
        ),
        encoding="utf-8",
    )


def test_native_nep_uses_snapshot_after_source_replacement(tmp_path: Path) -> None:
    path = tmp_path / "nep.txt"
    _write_nep(path, "C")
    replacement = tmp_path / "replacement.txt"
    _write_nep(replacement, "H")
    _run_native_regression("nep", path, replacement)


@pytest.mark.parametrize(
    ("mode", "filename"),
    [("mtp4", "mlip4_test_mtp.json"), ("mtp2", "mlip2_test.mtp")],
)
def test_native_mtp_branches_use_snapshot_after_source_replacement(
    tmp_path: Path,
    mode: str,
    filename: str,
) -> None:
    path = tmp_path / filename
    path.write_bytes((_ROOT / "tests" / "data" / filename).read_bytes())
    replacement = tmp_path / "replacement.txt"
    replacement.write_text("this is not an MTP model", encoding="utf-8")
    _run_native_regression(mode, path, replacement)


def test_native_cache_identity_uses_snapshot_content(tmp_path: Path) -> None:
    path = tmp_path / "nep.txt"
    _write_nep(path, "C")
    replacement = tmp_path / "replacement.txt"
    _write_nep(replacement, "H")
    _run_native_regression("cache", path, replacement)


def test_dpa_checkpoint_loader_accepts_snapshot_bytes(tmp_path: Path) -> None:
    from mdescriptor.descriptors.model_backed.dpa import load_dpa_checkpoint
    from mdescriptor.models import DPA4C_MODEL

    content = DPA4C_MODEL.read_bytes()
    path = tmp_path / "checkpoint.pt"
    expected_info, _ = load_dpa_checkpoint(
        DPA4C_MODEL,
        expected_descriptor="DPA4C",
    )
    actual_info, checkpoint = load_dpa_checkpoint(
        path,
        content=content,
        expected_descriptor="DPA4C",
    )
    assert actual_info == expected_info
    assert checkpoint["model"]["_extra_state"]["model_params"]["descriptor"]["type"] == "dpa4c"


@pytest.mark.gpu
def test_cuda_nep_uses_snapshot_after_source_replacement(tmp_path: Path) -> None:
    from tests._cuda import load_cuda_for_tests

    load_cuda_for_tests()
    cuda = sys.modules.get("mdescriptor._cuda")
    cuda_location = None if cuda is None else getattr(cuda, "__file__", None)
    if not cuda_location:
        pytest.fail("standard CUDA probe did not expose mdescriptor._cuda")
    cuda_path = Path(cuda_location).resolve()
    path = tmp_path / "nep.txt"
    _write_nep(path, "C")
    replacement = tmp_path / "replacement.txt"
    _write_nep(replacement, "H")
    _run_native_regression("cuda-nep", path, replacement, cuda_path=cuda_path)
