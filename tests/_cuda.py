"""Shared CUDA setup for tests that opt into the host GPU."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from ase import Atoms

from mdescriptor import ExecutionOptions, MDescriptorError, StructureBatch
from mdescriptor._cuda_loader import CudaPluginUnavailable, load_cuda_plugin
from mdescriptor.descriptors import SphericalExpansion

_PROBE_RESULT: str | None = None


def load_cuda_for_tests() -> None:
    """Load the CUDA extension and skip when this runner has no usable GPU.

    Importing ``mdescriptor._cuda`` succeeds even on driverless machines, so
    the probe runs one minimal CUDA computation.  The outcome is cached for
    the rest of the process.
    """

    global _PROBE_RESULT
    if _PROBE_RESULT is not None:
        if _PROBE_RESULT != "ok":
            pytest.skip(_PROBE_RESULT)
        return
    try:
        load_cuda_plugin(Path(__file__).parents[1] / "build-cuda")
    except CudaPluginUnavailable as error:
        _PROBE_RESULT = str(error)
        pytest.skip(_PROBE_RESULT)
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
        pytest.skip(_PROBE_RESULT)
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
