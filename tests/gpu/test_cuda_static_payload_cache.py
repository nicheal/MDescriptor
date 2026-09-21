"""CUDA static-payload cache lifecycle regressions."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from tests._cuda import load_cuda_for_tests

from mdescriptor import ExecutionOptions, StructureBatch
from mdescriptor.descriptors import ACE, C00PSMLFF, MTP, SphericalExpansionByPair


def _batch(numbers: list[int]) -> StructureBatch:
    positions = np.zeros((len(numbers), 3), dtype=np.float64)
    positions[:, 0] = np.arange(len(numbers), dtype=np.float64) * 1.2
    return StructureBatch.from_ase(
        [
            Atoms(
                numbers=numbers,
                positions=positions,
                cell=np.diag([10.0, 10.0, 10.0]),
                pbc=True,
            )
        ]
    )


def _descriptor(name: str):
    execution = ExecutionOptions(device="cuda")
    if name == "ACE":
        return ACE(species=[1, 8], N=3, maxdeg=4, rcut=3.5, execution=execution)
    if name == "MTP":
        model = Path(__file__).parents[1] / "data" / "mlip4_test_mtp.json"
        return MTP(species=[13, 14], model=model, execution=execution)
    if name == "SphericalExpansionByPair":
        return SphericalExpansionByPair(
            species=[1, 8],
            cutoff=3.0,
            density_width=0.5,
            max_radial=2,
            max_angular=2,
            execution=execution,
        )
    return C00PSMLFF(
        species=[1, 8], r_cut=3.0, n_radial=3, l_max=2, execution=execution
    )


@pytest.mark.gpu
@pytest.mark.parametrize(
    ("name", "numbers"),
    [
        ("ACE", [1, 8, 1]),
        ("MTP", [13, 14]),
        ("C00PSMLFF", [8, 1, 1]),
        ("SphericalExpansionByPair", [1, 8, 1]),
    ],
)
def test_cuda_static_payload_reuse_and_descriptor_lifetime(
    name: str, numbers: list[int]
) -> None:
    """Static basis/model arrays are reusable but never outlive one descriptor."""

    load_cuda_for_tests()
    batch = _batch(numbers)
    descriptor = _descriptor(name)
    try:
        first = np.asarray(descriptor.compute(batch).values)
        snapshot = first.copy()
        second = np.asarray(descriptor.compute(batch).values)
        np.testing.assert_array_equal(second, snapshot)
    finally:
        descriptor.close()
    np.testing.assert_array_equal(first, snapshot)

    replacement = _descriptor(name)
    try:
        np.testing.assert_array_equal(replacement.compute(batch).values, snapshot)
    finally:
        replacement.close()


@pytest.mark.gpu
@pytest.mark.parametrize(
    "overrides", [{}, {"include_angular": False}, {"radial_sigma": 0.0}]
)
def test_cuda_c00ps_self_correction_matches_cpu(
    overrides: dict[str, object],
) -> None:
    load_cuda_for_tests()
    batch = _batch([8, 1, 1])
    parameters = {
        "species": [1, 8],
        "r_cut": 3.0,
        "n_radial": 3,
        "l_max": 2,
        **overrides,
    }
    cpu = C00PSMLFF(**parameters, execution=ExecutionOptions(device="cpu", num_threads=1))
    gpu = C00PSMLFF(**parameters, execution=ExecutionOptions(device="cuda"))
    try:
        np.testing.assert_allclose(
            gpu.compute(batch).values,
            cpu.compute(batch).values,
            rtol=1e-10,
            atol=1e-10,
        )
    finally:
        cpu.close()
        gpu.close()
