"""CUDA matrix descriptor contract tests."""

from __future__ import annotations

import importlib
import os
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms

import mdescriptor
from mdescriptor import ExecutionOptions, MDescriptorError, StructureBatch
from mdescriptor.descriptors import CoulombMatrix, EwaldSumMatrix, SineMatrix


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
        [
            Atoms(
                "NaCl",
                positions=[[0.0, 0.0, 0.0], [2.1, 2.0, 2.2]],
                cell=np.diag([8.0, 8.0, 8.0]),
                pbc=True,
            ),
            Atoms(
                "Si3",
                positions=[[0.2, 0.1, 0.0], [1.6, 0.0, 0.1], [0.4, 1.5, -0.2]],
                cell=np.diag([9.0, 10.0, 11.0]),
                pbc=True,
            ),
        ],
        ids=["salt", "silicon"],
    )


MATRIX_DESCRIPTORS = [
    (CoulombMatrix, {}),
    (SineMatrix, {}),
    (EwaldSumMatrix, {"accuracy": 1e-5, "w": 1.0, "r_cut": 4.0, "g_cut": 3.0, "a": 0.3}),
]


@pytest.mark.gpu
@pytest.mark.parametrize(("descriptor_type", "parameters"), MATRIX_DESCRIPTORS)
@pytest.mark.parametrize("permutation", ["none", "sorted_l2", "eigenspectrum"])
def test_cuda_matrix_contract_matches_cpu_shape_and_output_semantics(
    descriptor_type: type[object], parameters: dict[str, float], permutation: str
) -> None:
    """CPU and CUDA expose the same matrix width, padding, and ordering contract."""

    _load_cuda_plugin()
    batch = _batch()
    common = {"n_atoms_max": 4, "permutation": permutation, **parameters}
    cpu = descriptor_type(**common, execution=ExecutionOptions(device="cpu", num_threads=1))
    gpu = descriptor_type(**common, execution=ExecutionOptions(device="cuda"))
    try:
        expected = cpu.compute(batch)
        try:
            actual = gpu.compute(batch)
        except MDescriptorError as error:
            if error.code == "device_unavailable":
                pytest.skip(str(error))
            raise

        assert actual.level == expected.level == "structure"
        assert actual.values.shape == expected.values.shape == (2, 16 if permutation != "eigenspectrum" else 4)
        assert actual.labels == expected.labels
        assert actual.structure_ids == expected.structure_ids
        np.testing.assert_array_equal(actual.samples, expected.samples)
        np.testing.assert_allclose(np.isfinite(actual.values), True)

        counts = np.diff(batch.offsets)
        for row, count_value in zip(actual.values, counts, strict=True):
            count = int(count_value)
            if permutation == "eigenspectrum":
                assert np.all(row[count:] == 0.0)
                assert np.all(np.abs(row[: max(count - 1, 0)]) >= np.abs(row[1:count]))
            else:
                matrix = row.reshape(4, 4)
                assert np.all(matrix[count:, :] == 0.0)
                assert np.all(matrix[:, count:] == 0.0)
                if permutation == "sorted_l2":
                    norms = np.linalg.norm(matrix[:count, :count], axis=1)
                    assert np.all(norms[:-1] >= norms[1:])
    finally:
        cpu.close()
        gpu.close()
