"""GPU regression tests for multi-species local spectra."""

from __future__ import annotations

import importlib
import os
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms

import mdescriptor
from mdescriptor import ExecutionOptions, MDescriptorError, StructureBatch
from mdescriptor.descriptors import (
    SoapPowerSpectrum,
    SoapRadialSpectrum,
    SphericalExpansion,
)


def _load_cuda_plugin() -> None:
    """Make a source-tree CUDA build visible to the editable package."""

    try:
        importlib.import_module("mdescriptor._cuda")
        return
    except (ImportError, OSError):
        pass

    configured = os.environ.get("MDESCRIPTOR_CUDA_PLUGIN_DIR")
    candidates = [
        Path(configured) if configured else None,
        Path(__file__).parents[1] / "build-cuda",
    ]
    for candidate in candidates:
        if candidate is None or not any(candidate.glob("_cuda*.so")):
            continue
        mdescriptor.__path__.insert(0, str(candidate))
        try:
            importlib.import_module("mdescriptor._cuda")
        except (ImportError, OSError):
            continue
        return
    pytest.skip("CUDA plugin is not installed in this test environment")


def _batch() -> StructureBatch:
    return StructureBatch.from_ase(
        [
            Atoms(
                numbers=[1, 8],
                positions=[[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]],
                cell=np.diag([10.0, 10.0, 10.0]),
                pbc=True,
            )
        ]
    )


@pytest.mark.gpu
@pytest.mark.parametrize(
    ("name", "descriptor_type"),
    [
        ("SphericalExpansion", SphericalExpansion),
        ("SoapRadialSpectrum", SoapRadialSpectrum),
        ("SoapPowerSpectrum", SoapPowerSpectrum),
    ],
)
def test_cuda_multispecies_spectra_match_cpu(name, descriptor_type) -> None:
    """The CUDA spectra must match the CPU reference for two species."""

    _load_cuda_plugin()
    batch = _batch()
    parameters = {
        "species": [1, 8],
        "cutoff": 3.5,
        "density_width": 0.6,
        "max_radial": 2,
        "max_angular": 2,
    }
    cpu = descriptor_type(
        **parameters,
        execution=ExecutionOptions(device="cpu", num_threads=1),
    )
    gpu = descriptor_type(
        **parameters,
        execution=ExecutionOptions(device="cuda"),
    )
    try:
        cpu_result = cpu.compute(batch)
        try:
            gpu_result = gpu.compute(batch)
        except MDescriptorError as error:
            if error.code == "device_unavailable":
                pytest.skip(str(error))
            raise
        np.testing.assert_allclose(
            gpu_result.values,
            cpu_result.values,
            rtol=1e-10,
            atol=1e-10,
            err_msg=f"CUDA {name} differs from the CPU reference",
        )
        np.testing.assert_array_equal(gpu_result.samples, cpu_result.samples)
        np.testing.assert_array_equal(gpu_result.row_offsets, cpu_result.row_offsets)
        assert gpu_result.labels == cpu_result.labels
    finally:
        cpu.close()
        gpu.close()
