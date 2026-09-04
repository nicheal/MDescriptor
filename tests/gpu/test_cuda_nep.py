"""CUDA NEP descriptor parity tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.io import read
from tests._cuda import load_cuda_for_tests

from mdescriptor import ExecutionOptions, MDescriptorError, StructureBatch
from mdescriptor.descriptors import NEP
from mdescriptor.models import NEP_MODEL


def _batch() -> StructureBatch:
    return StructureBatch.from_ase(
        [
            Atoms(
                "OHH",
                positions=[
                    [0.0, 0.0, 0.0],
                    [0.96, 0.0, 0.0],
                    [-0.24, 0.93, 0.0],
                ],
                cell=np.diag([8.0, 8.0, 8.0]),
                pbc=True,
            ),
            Atoms(
                "C2",
                positions=[[0.0, 0.0, 0.0], [1.42, 0.13, -0.07]],
                cell=np.diag([10.0, 11.0, 12.0]),
                pbc=True,
            ),
        ],
        ids=["water", "carbon"],
    )


def _mixed_periodic_isolated_batch() -> StructureBatch:
    """Exercise device expansion and the isolated branch in one batch."""

    return StructureBatch.from_ase(
        [
            Atoms(
                "C2",
                positions=[[0.0, 0.0, 0.0], [1.42, 0.13, -0.07]],
                cell=np.diag([8.0, 8.0, 8.0]),
                pbc=True,
            ),
            Atoms(
                "OHH",
                positions=[
                    [0.0, 0.0, 0.0],
                    [0.96, 0.0, 0.0],
                    [-0.24, 0.93, 0.0],
                ],
                cell=np.zeros((3, 3)),
                pbc=False,
            ),
        ],
        ids=["periodic-small", "isolated-water"],
    )


def _isolated_batch() -> StructureBatch:
    return StructureBatch.from_ase(
        [
            Atoms(
                "C2",
                positions=[[0.0, 0.0, 0.0], [1.42, 0.13, -0.07]],
                cell=np.zeros((3, 3)),
                pbc=False,
            ),
            Atoms(
                "OHH",
                positions=[
                    [0.0, 0.0, 0.0],
                    [0.96, 0.0, 0.0],
                    [-0.24, 0.93, 0.0],
                ],
                cell=np.zeros((3, 3)),
                pbc=False,
            ),
        ],
        ids=["isolated-carbon", "isolated-water"],
    )


@pytest.mark.gpu
@pytest.mark.model
def test_cuda_nep_matches_cpu_contract_and_values() -> None:
    """CUDA NEP preserves the atom result contract and reference tolerance."""

    load_cuda_for_tests()
    batch = _batch()
    cpu = NEP(model=NEP_MODEL, execution=ExecutionOptions(device="cpu", num_threads=1))
    gpu = NEP(model=NEP_MODEL, execution=ExecutionOptions(device="cuda"))
    try:
        expected = cpu.compute(batch)
        try:
            actual = gpu.compute(batch)
        except MDescriptorError as error:
            if error.code == "device_unavailable":
                pytest.skip(str(error))
            raise

        assert actual.level == expected.level == "atom"
        assert actual.values.shape == expected.values.shape
        assert actual.feature_count == expected.feature_count == 35
        assert actual.labels == expected.labels
        np.testing.assert_array_equal(actual.samples, expected.samples)
        np.testing.assert_array_equal(actual.row_offsets, expected.row_offsets)
        assert actual.structure_ids == expected.structure_ids
        np.testing.assert_allclose(
            actual.values,
            expected.values,
            rtol=1.0e-6,
            atol=1.0e-7,
            err_msg="CUDA NEP differs from the CPU descriptor reference",
        )
        assert actual.metadata["execution"] == {
            "device": "cuda",
            "num_threads": None,
        }
    finally:
        cpu.close()
        gpu.close()


@pytest.mark.gpu
@pytest.mark.model
def test_cuda_nep_periodic_carbon_is_stable_against_nepadapters() -> None:
    """Periodic cell-list order must not move NEP outside the reference tolerance."""

    load_cuda_for_tests()
    from nep_adapters import NEPCalculator

    structure = read(
        Path(__file__).parents[2] / "benchmarks/_datasets/legacy/carbon_dataset_pbc.xyz",
        index=34,
    )
    batch = StructureBatch.from_ase([structure], ids=["carbon-pbc-frame34"])
    gpu = NEP(model=NEP_MODEL, execution=ExecutionOptions(device="cuda"))
    reference = NEPCalculator(str(NEP_MODEL), backend="cuda")
    try:
        expected = np.asarray(reference.predict_descriptors([structure]), dtype=np.float64)
        previous = None
        for _ in range(3):
            actual = np.asarray(gpu.compute(batch).values, dtype=np.float64)
            if previous is not None:
                np.testing.assert_array_equal(
                    actual,
                    previous,
                    err_msg="CUDA NEP periodic neighbor order is not deterministic",
                )
            np.testing.assert_allclose(
                actual,
                expected,
                rtol=1.0e-6,
                atol=1.0e-7,
                err_msg="CUDA NEP periodic cell-list order differs from NEPAdapters",
            )
            previous = actual
    finally:
        gpu.close()
        reference.close()


@pytest.mark.gpu
@pytest.mark.model
def test_cuda_nep_mixed_periodic_and_isolated_batch_matches_cpu() -> None:
    """The device-expanded periodic and isolated structures share one graph."""

    load_cuda_for_tests()
    batch = _mixed_periodic_isolated_batch()
    cpu = NEP(model=NEP_MODEL, execution=ExecutionOptions(device="cpu", num_threads=1))
    gpu = NEP(model=NEP_MODEL, execution=ExecutionOptions(device="cuda"))
    try:
        expected = cpu.compute(batch)
        actual = gpu.compute(batch)
        np.testing.assert_array_equal(actual.row_offsets, expected.row_offsets)
        np.testing.assert_allclose(
            actual.values,
            expected.values,
            rtol=1.0e-6,
            atol=1.0e-7,
            err_msg="CUDA NEP mixed periodic/isolated graph differs from CPU",
        )
    finally:
        cpu.close()
        gpu.close()


@pytest.mark.gpu
@pytest.mark.model
def test_cuda_nep_isolated_batch_matches_cpu() -> None:
    """An all-isolated batch must not apply periodic wrapping or cross-talk."""

    load_cuda_for_tests()
    batch = _isolated_batch()
    cpu = NEP(model=NEP_MODEL, execution=ExecutionOptions(device="cpu", num_threads=1))
    gpu = NEP(model=NEP_MODEL, execution=ExecutionOptions(device="cuda"))
    try:
        expected = cpu.compute(batch)
        actual = gpu.compute(batch)
        np.testing.assert_array_equal(actual.row_offsets, expected.row_offsets)
        np.testing.assert_allclose(
            actual.values,
            expected.values,
            rtol=1.0e-6,
            atol=1.0e-7,
            err_msg="CUDA NEP isolated graph differs from CPU",
        )
    finally:
        cpu.close()
        gpu.close()
