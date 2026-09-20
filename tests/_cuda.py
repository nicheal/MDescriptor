"""Shared CUDA setup for tests that opt into the host GPU."""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import NoReturn

import numpy as np
import pytest
from ase import Atoms

from mdescriptor import ExecutionOptions, MDescriptorError, StructureBatch
from mdescriptor._cuda_loader import CudaPluginUnavailable, load_cuda_plugin

_PROBE_RESULT: str | None = None
_SNAPSHOT_ABI = 1


def _strict_mode() -> bool:
    return os.environ.get("MDESCRIPTOR_REQUIRE_GPU", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _unavailable(message: str, *, require_gpu: bool) -> NoReturn:
    if require_gpu:
        pytest.fail(message)
    pytest.skip(message)


def _loaded_native(require_gpu: bool) -> ModuleType:
    try:
        native = sys.modules.get("mdescriptor._native") or importlib.import_module(
            "mdescriptor._native"
        )
    except (ImportError, OSError) as error:
        _unavailable(f"native extension is unavailable: {error}", require_gpu=require_gpu)
    location = getattr(native, "__file__", None)
    if not location:
        _unavailable("native extension has no extension file", require_gpu=require_gpu)
    assert location
    configured = os.environ.get(
        "MDESCRIPTOR_EXPECTED_NATIVE_PLUGIN_DIR",
        os.environ.get("MDESCRIPTOR_NATIVE_PLUGIN_DIR"),
    )
    if configured:
        actual = Path(location).resolve()
        expected = Path(configured).expanduser().resolve()
        if actual.parent != expected:
            pytest.fail(f"native extension loaded from {actual}, expected {expected}")
    return native


def _validate_snapshot_extensions(plugin_directory: Path, *, require_gpu: bool) -> None:
    cuda = sys.modules.get("mdescriptor._cuda")
    if cuda is None:
        _unavailable(
            "CUDA plugin did not remain loaded after discovery",
            require_gpu=require_gpu,
        )
        return
    location = getattr(cuda, "__file__", None)
    if not location:
        _unavailable("CUDA plugin has no extension file", require_gpu=require_gpu)
        return
    actual_cuda = Path(location).resolve()
    configured_cuda = os.environ.get("MDESCRIPTOR_CUDA_PLUGIN_DIR")
    expected_cuda = (
        Path(configured_cuda).expanduser().resolve()
        if configured_cuda
        else Path(plugin_directory).resolve()
    )
    if configured_cuda and actual_cuda.parent != expected_cuda:
        _unavailable(
            "CUDA plugin loaded from "
            f"{actual_cuda.parent}, expected {expected_cuda}",
            require_gpu=require_gpu,
        )
    if require_gpu and getattr(cuda, "MODEL_SNAPSHOT_ABI", None) != _SNAPSHOT_ABI:
        pytest.fail(
            f"CUDA plugin {actual_cuda} lacks immutable model snapshot ABI "
            f"{_SNAPSHOT_ABI}"
        )

    native = _loaded_native(require_gpu)
    native_location = getattr(native, "__file__", None)
    assert native_location
    for options_name in ("NepOptions", "MtpOptions"):
        options_type = getattr(native, options_name, None)
        if options_type is None or not hasattr(options_type(), "model_data"):
            pytest.fail(
                f"native extension {native_location} lacks {options_name}.model_data; "
                "rebuild MDescriptor"
            )


def load_cuda_for_tests(*, require_gpu: bool | None = None) -> None:
    """Load the CUDA extension and skip when this runner has no usable GPU.

    Importing ``mdescriptor._cuda`` succeeds even on driverless machines, so
    the probe runs one minimal CUDA computation.  The outcome is cached for
    the rest of the process.  Set ``MDESCRIPTOR_REQUIRE_GPU=1`` or pass
    ``require_gpu=True`` for a CI gate that fails instead of skipping.
    """

    strict = _strict_mode() if require_gpu is None else require_gpu
    global _PROBE_RESULT
    if _PROBE_RESULT is not None:
        if _PROBE_RESULT != "ok":
            _unavailable(_PROBE_RESULT, require_gpu=strict)
        if strict:
            configured = os.environ.get("MDESCRIPTOR_CUDA_PLUGIN_DIR")
            _validate_snapshot_extensions(
                Path(configured)
                if configured
                else Path(__file__).parents[1] / "build-cuda",
                require_gpu=True,
            )
        return
    try:
        plugin_directory = load_cuda_plugin(Path(__file__).parents[1] / "build-cuda")
    except CudaPluginUnavailable as error:
        _PROBE_RESULT = str(error)
        _unavailable(_PROBE_RESULT, require_gpu=strict)
        return
    if strict:
        _validate_snapshot_extensions(plugin_directory, require_gpu=True)
    from mdescriptor.descriptors import SphericalExpansion

    try:
        descriptor = SphericalExpansion(
            species=[1], execution=ExecutionOptions(device="cuda")
        )
        try:
            descriptor.compute(_probe_batch())
        finally:
            descriptor.close()
    except MDescriptorError as error:
        _PROBE_RESULT = f"no usable CUDA device: {error}"
        _unavailable(_PROBE_RESULT, require_gpu=strict)
        return
    _PROBE_RESULT = "ok"


def _probe_batch() -> StructureBatch:
    return StructureBatch.from_ase(
        [
            Atoms(
                "H2",
                positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.1]],
                cell=np.eye(3) * 10.0,
                pbc=True,
            )
        ],
        ids=["probe"],
    )


__all__ = ["load_cuda_for_tests"]
