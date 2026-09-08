"""CUDA DPA4/DPA4C descriptor parity tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from tests._cuda import load_cuda_for_tests

from mdescriptor import ExecutionOptions, MDescriptorError, StructureBatch
from mdescriptor.descriptors import DPA4, DPA4C
from mdescriptor.models import DPA4_MODEL, DPA4C_MODEL


def _batch() -> StructureBatch:
    return StructureBatch.from_ase(
        Atoms(
            "H2",
            positions=[[0.0, 0.0, 0.0], [1.0, 0.2, -0.1]],
        ),
        ids=["h2"],
    )


def _periodic_structures() -> list[Atoms]:
    return [
        Atoms(
            "H2",
            positions=[[0.0, 0.0, 0.0], [8.2, -0.2, 0.1]],
            cell=np.diag([8.0, 8.0, 8.0]),
            pbc=True,
        ),
        Atoms(
            "H2",
            positions=[[0.4, 0.1, 0.0], [1.6, 0.0, -0.2]],
            cell=np.zeros((3, 3)),
            pbc=False,
        ),
    ]


def _periodic_batch() -> StructureBatch:
    """Exercise device-side wrapping and periodic image enumeration."""

    return StructureBatch.from_ase(
        _periodic_structures(),
        ids=["periodic-wrapped", "isolated"],
    )


@pytest.mark.gpu
@pytest.mark.model
@pytest.mark.parametrize("pairs", [1, 384])
@pytest.mark.parametrize("isolated_frames", [False, True])
def test_cuda_dpa4_connected_and_isolated_atoms_match_cpu(
    pairs: int, isolated_frames: bool
) -> None:
    load_cuda_for_tests()
    positions = np.zeros((2 * pairs + 1, 3))
    positions[:, 0] = (np.arange(2 * pairs + 1) // 2) * 10.0
    positions[1::2, 0] += 0.9
    structures = [Atoms(numbers=np.ones(len(positions), dtype=int), positions=positions)]
    if isolated_frames:
        structures = [Atoms("H"), *structures, Atoms("H2", positions=[[0, 0, 0], [10, 0, 0]])]
    batch = StructureBatch.from_ase(structures)
    cpu = DPA4(execution=ExecutionOptions(device="cpu", num_threads=4))
    gpu = DPA4(execution=ExecutionOptions(device="cuda"))
    try:
        expected = cpu.compute(batch).values
        if pairs == 1:
            from mdescriptor.descriptors.model_backed.dpa import (
                compute_batch,
                load_dpa_checkpoint,
                new_runtime,
            )

            _, checkpoint = load_dpa_checkpoint(DPA4_MODEL, expected_descriptor="DPA4")
            reference = compute_batch(new_runtime(DPA4_MODEL, checkpoint), batch)
            np.testing.assert_allclose(expected, reference, atol=4e-5, rtol=2e-5)
        actual = gpu.compute(batch).values
        np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=2e-5)
        if pairs == 1 and isolated_frames:
            begin = 0
            for structure in structures:
                end = begin + len(structure)
                single = gpu.compute(StructureBatch.from_ase(structure)).values
                np.testing.assert_allclose(single, expected[begin:end], atol=1e-5, rtol=2e-5)
                begin = end
            np.testing.assert_array_equal(gpu.compute(batch).values, actual)
    finally:
        cpu.close()
        gpu.close()


@pytest.mark.gpu
@pytest.mark.model
def test_cuda_dpa4_message_grid_tile_tail_matches_cpu() -> None:
    load_cuda_for_tests()
    # Fill one 768-node message tile and leave a one-node tail. Widely spaced
    # pairs bound edge work while exercising scratch reuse across grid stages.
    positions = np.zeros((769, 3))
    positions[:, 0] = (np.arange(769) // 2) * 10.0
    positions[1::2, 0] += 0.9
    # Make the tail part of a trimer, so it exercises a nonzero message.
    positions[-1, 0] = positions[-2, 0] + 0.9
    batch = StructureBatch.from_ase(Atoms(numbers=np.ones(769, dtype=int), positions=positions))
    cpu = DPA4(execution=ExecutionOptions(device="cpu", num_threads=4))
    gpu = DPA4(execution=ExecutionOptions(device="cuda"))
    try:
        expected = cpu.compute(batch).values
        actual = gpu.compute(batch).values
        np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=2e-5)
    finally:
        cpu.close()
        gpu.close()


@pytest.mark.gpu
@pytest.mark.model
@pytest.mark.parametrize(
    ("descriptor_type", "model", "feature_count"),
    [
        (DPA4, DPA4_MODEL, 64),
        (DPA4C, DPA4C_MODEL, 219),
    ],
)
def test_cuda_dpa_matches_cpu_contract_and_values(
    descriptor_type: type[object], model: Path, feature_count: int
) -> None:
    """CUDA DPA descriptors preserve the public result and numerical contract."""

    load_cuda_for_tests()
    batch = _batch()
    cpu = descriptor_type(
        model=model,
        execution=ExecutionOptions(device="cpu", num_threads=1),
    )
    gpu = descriptor_type(
        model=model,
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

        assert actual.level == expected.level == "atom"
        assert actual.values.shape == expected.values.shape == (2, feature_count)
        assert actual.feature_count == expected.feature_count == feature_count
        assert actual.labels == expected.labels
        np.testing.assert_array_equal(actual.samples, expected.samples)
        np.testing.assert_array_equal(actual.row_offsets, expected.row_offsets)
        assert actual.structure_ids == expected.structure_ids
        np.testing.assert_allclose(
            actual.values,
            expected.values,
            rtol=2.0e-5,
            atol=1.0e-5,
            err_msg=f"CUDA {descriptor_type.__name__} differs from the CPU reference",
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
@pytest.mark.parametrize(
    ("descriptor_type", "model", "feature_count"),
    [
        (DPA4, DPA4_MODEL, 64),
        (DPA4C, DPA4C_MODEL, 219),
    ],
)
def test_cuda_dpa_device_graph_matches_cpu_for_periodic_and_isolated_batch(
    descriptor_type: type[object], model: Path, feature_count: int
) -> None:
    """The device graph handles PBC without cross-structure leakage.

    The ordinary CPU parity test above is the numerical contract.  This test
    isolates graph construction: a mixed batch must be exactly the same as
    evaluating each structure independently on the same CUDA implementation.
    That catches host-side graph fallback, bad structure mapping, and image
    data leaking between CSR rows without turning a deliberately tiny,
    ill-conditioned periodic H2 example into a cross-backend precision gate.
    """

    load_cuda_for_tests()
    structures = _periodic_structures()
    batch = StructureBatch.from_ase(
        structures,
        ids=["periodic-wrapped", "isolated"],
    )
    gpu = descriptor_type(
        model=model,
        execution=ExecutionOptions(device="cuda"),
    )
    try:
        actual = gpu.compute(batch)
        independent = [
            gpu.compute(StructureBatch.from_ase(structure, ids=[str(index)]))
            for index, structure in enumerate(structures)
        ]
        expected_values = np.concatenate(
            [result.values for result in independent],
            axis=0,
        )
        expected_samples = np.concatenate(
            [result.samples + np.array([index, 0]) for index, result in enumerate(independent)],
            axis=0,
        )
        assert actual.values.shape == expected_values.shape == (4, feature_count)
        np.testing.assert_array_equal(actual.values, expected_values)
        np.testing.assert_array_equal(actual.samples, expected_samples)
        np.testing.assert_array_equal(actual.row_offsets, batch.offsets)
    finally:
        gpu.close()
