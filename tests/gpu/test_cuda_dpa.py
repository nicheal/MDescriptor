"""CUDA DPA4/DPA4C descriptor parity tests."""

from __future__ import annotations

import importlib
import os
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms

import mdescriptor
from mdescriptor import ExecutionOptions, MDescriptorError, StructureBatch
from mdescriptor.descriptors import DPA4, DPA4C
from mdescriptor.models import DPA4_MODEL, DPA4C_MODEL


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
        Path(__file__).parents[2] / "build-cuda",
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

    _load_cuda_plugin()
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

    _load_cuda_plugin()
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
