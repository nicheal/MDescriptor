"""GPU regression tests for multi-species local spectra."""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms

from mdescriptor import ExecutionOptions, MDescriptorError, StructureBatch
from mdescriptor.descriptors import (
    LMBTR,
    MBTR,
    SoapPowerSpectrum,
    SoapRadialSpectrum,
    SphericalExpansion,
    ValleOganov,
)
from tests._cuda import load_cuda_for_tests


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

    load_cuda_for_tests()
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


def _mbtr_batch() -> StructureBatch:
    return StructureBatch.from_ase(
        [
            Atoms(
                numbers=[1, 8, 1],
                positions=[[0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [0.0, 1.2, 0.0]],
                cell=np.diag([10.0, 10.0, 10.0]),
                pbc=True,
            )
        ]
    )


@pytest.mark.gpu
@pytest.mark.parametrize(
    ("descriptor_type", "parameters"),
    [
        (
            MBTR,
            {
                "species": [1, 8],
                "geometry": {"function": "distance"},
                "grid": {"min": 0.0, "max": 4.0, "n": 20, "sigma": 0.1},
                "weighting": {"function": "exp", "scale": 0.3, "threshold": 1e-3},
                "normalization": "none",
            },
        ),
        (
            LMBTR,
            {
                "species": [1, 8],
                "geometry": {"function": "angle"},
                "grid": {"min": 0.0, "max": 180.0, "n": 20, "sigma": 0.1},
                "weighting": {"function": "smooth_cutoff", "r_cut": 3.5},
                "normalization": "l2",
            },
        ),
        (
            ValleOganov,
            {
                "species": [1, 8],
                "function": "distance",
                "n": 20,
                "sigma": 0.1,
                "r_cut": 3.5,
                "geometry": {"function": "inverse_distance"},
                "grid": {"min": 0.0, "max": 2.0, "n": 20, "sigma": 0.1},
                "weighting": {"function": "exp", "scale": 0.7, "threshold": 0.01},
                "normalization": "none",
            },
        ),
    ],
)
def test_cuda_mbtr_family_matches_cpu(
    descriptor_type, parameters: dict[str, object]
) -> None:
    """CPU and CUDA consume the same resolved MBTR controls."""

    load_cuda_for_tests()
    batch = _mbtr_batch()
    cpu = descriptor_type(
        **parameters,
        execution=ExecutionOptions(device="cpu", num_threads=1),
    )
    gpu = descriptor_type(
        **parameters,
        execution=ExecutionOptions(device="cuda"),
    )
    try:
        expected = cpu.compute(batch)
        try:
            actual = gpu.compute(batch)
        except MDescriptorError as error:
            if error.code == "device_unavailable":
                pytest.skip(str(error))
            raise
        np.testing.assert_allclose(actual.values, expected.values, rtol=1e-10, atol=1e-10)
        assert actual.level == expected.level
        assert actual.labels == expected.labels
        if expected.row_offsets is not None:
            np.testing.assert_array_equal(actual.row_offsets, expected.row_offsets)
    finally:
        cpu.close()
        gpu.close()
