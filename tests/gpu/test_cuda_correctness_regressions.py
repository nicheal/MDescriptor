"""Focused CUDA regressions for workspace, species, and zero-distance paths."""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from tests._cuda import load_cuda_for_tests

from mdescriptor import (
    DescriptorConfigError,
    ExecutionOptions,
    MDescriptorError,
    StructureBatch,
)
from mdescriptor.descriptors import MBTR, MTP, EwaldSumMatrix, ValleOganov


def _ewald_batch() -> StructureBatch:
    return StructureBatch.from_ase(
        [
            Atoms(
                "NaCl",
                positions=[[0.0, 0.0, 0.0], [2.1, 2.0, 2.2]],
                cell=np.diag([8.0, 8.0, 8.0]),
                pbc=True,
            ),
            Atoms(
                "Si4",
                positions=[
                    [0.2, 0.1, 0.0],
                    [1.6, 0.0, 0.1],
                    [0.4, 1.5, -0.2],
                    [1.4, 1.2, 1.1],
                ],
                cell=np.diag([9.0, 10.0, 11.0]),
                pbc=True,
            ),
        ]
    )


@pytest.mark.gpu
def test_cuda_ewald_fresh_context_matches_cpu() -> None:
    """Ewald's matrix and scratch slices must survive workspace allocation."""

    load_cuda_for_tests()
    batch = _ewald_batch()
    parameters = {
        "n_atoms_max": 4,
        "permutation": "none",
        "accuracy": 1e-5,
        "w": 1.0,
        "r_cut": 4.0,
        "g_cut": 3.0,
        "a": 0.3,
    }
    cpu = EwaldSumMatrix(**parameters, execution=ExecutionOptions(device="cpu", num_threads=1))
    gpu = EwaldSumMatrix(**parameters, execution=ExecutionOptions(device="cuda"))
    try:
        expected = cpu.compute(batch).values
        try:
            actual = gpu.compute(batch).values
        except MDescriptorError as error:
            if error.code == "device_unavailable":
                pytest.skip(str(error))
            raise
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)
        assert np.isfinite(actual).all()
    finally:
        cpu.close()
        gpu.close()


@pytest.mark.gpu
@pytest.mark.parametrize(
    ("descriptor_type", "parameters"),
    [
        (ValleOganov, {"function": "distance", "n": 4, "r_cut": 3.0}),
        (ValleOganov, {"function": "angle", "n": 4, "r_cut": 3.0}),
        (
            MBTR,
            {
                "geometry": {"function": "distance"},
                "grid": {"min": 0.0, "max": 3.0, "n": 4, "sigma": 0.1},
                "weighting": {"function": "inverse_square", "r_cut": 3.0},
                "normalization": "valle_oganov",
            },
        ),
    ],
)
def test_cuda_valle_oganov_rejects_more_than_64_species(
    descriptor_type, parameters: dict[str, object]
) -> None:
    """The public schema rejects species 65 before backend construction."""

    load_cuda_for_tests()
    with pytest.raises(DescriptorConfigError) as caught:
        descriptor_type(
            species=list(range(1, 66)),
            **parameters,
            execution=ExecutionOptions(device="cuda"),
        )
    assert caught.value.code == "invalid_parameter"
    assert list(caught.value.path or ()) == ["parameters", "species"]


@pytest.mark.gpu
@pytest.mark.parametrize(
    ("descriptor_name", "options"),
    [
        (
            "ValleOganov",
            {"function": "distance", "n": 4, "r_cut": 3.0},
        ),
        (
            "ValleOganov",
            {"function": "angle", "n": 4, "r_cut": 3.0},
        ),
        (
            "MBTR",
            {
                "geometry": {"function": "distance"},
                "grid": {"min": 0.0, "max": 3.0, "n": 4, "sigma": 0.1},
                "weighting": {"function": "inverse_square", "r_cut": 3.0},
                "normalization": "valle_oganov",
            },
        ),
    ],
)
def test_cuda_native_valle_oganov_guard_rejects_more_than_64_species(
    descriptor_name: str, options: dict[str, object]
) -> None:
    """The native guard remains active behind the public schema."""

    load_cuda_for_tests()
    from mdescriptor._cuda import CudaBackend

    batch = StructureBatch.from_ase(
        [
            Atoms(
                numbers=[1, 2],
                positions=[[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]],
                cell=np.diag([8.0, 8.0, 8.0]),
                pbc=True,
            )
        ]
    )
    backend = CudaBackend(
        descriptor_name,
        {"species": list(range(1, 66)), **options},
    )
    try:
        with pytest.raises(ValueError, match="at most 64 species"):
            backend.compute(batch)
    finally:
        backend.close()


@pytest.mark.gpu
def test_cuda_mtp_skips_coincident_distinct_atoms_like_cpu() -> None:
    """A zero-distance distinct pair must not create an infinite unit vector."""

    load_cuda_for_tests()
    batch = StructureBatch.from_ase(
        [
            Atoms(
                numbers=[1, 1],
                positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                cell=np.diag([8.0, 8.0, 8.0]),
                pbc=True,
            )
        ]
    )
    parameters = {
        "species": [1],
        "min_dist": 0.0,
        "max_dist": 3.0,
        "radial_basis_size": 2,
        "max_rank": 2,
    }
    cpu = MTP(**parameters, execution=ExecutionOptions(device="cpu", num_threads=1))
    gpu = MTP(**parameters, execution=ExecutionOptions(device="cuda"))
    try:
        expected = cpu.compute(batch).values
        try:
            actual = gpu.compute(batch).values
        except MDescriptorError as error:
            if error.code == "device_unavailable":
                pytest.skip(str(error))
            raise
        assert np.isfinite(actual).all()
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)
    finally:
        cpu.close()
        gpu.close()
