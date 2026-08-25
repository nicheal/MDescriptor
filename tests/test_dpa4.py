import math

import numpy as np
import pytest

from mdescriptor.models import DPA4_MODEL
from tests._public import DPA4, ModelLoadError, StructureBatch

pytestmark = pytest.mark.model

MODEL = DPA4_MODEL


def _batch() -> StructureBatch:
    return StructureBatch(
        np.array([1, 8, 1, 8], dtype=np.int32),
        np.array(
            [
                [8.0, 8.0, 8.0],
                [9.0, 8.0, 8.0],
                [10.0, 10.0, 10.0],
                [28.0, 28.0, 28.0],
            ]
        ),
        np.array([np.eye(3) * 20.0, np.eye(3) * 20.0]),
        np.ones((2, 3), dtype=np.int32),
        np.array([0, 3, 4], dtype=np.int64),
        ("first", "second"),
    )


def test_official_checkpoint_and_batch_output():
    calculator = DPA4(model=MODEL)

    result = calculator.compute(_batch())

    assert result.values.shape == (4, 64)
    assert np.isfinite(result.values).all()
    assert result.level == "atom"
    assert result.metadata["backend"] == "mdescriptor-dpa4-official-native"
    assert result.row_offsets.tolist() == [0, 3, 4]
    assert result.labels[0] == "dpa4:scalar,channel=0"


def test_geometry_rotation_and_atom_permutation_are_invariant():
    calculator = DPA4(model=MODEL)
    batch = _batch()
    reference = calculator.compute(batch).values

    angle = 0.37
    rotation = np.array(
        [
            [math.cos(angle), -math.sin(angle), 0.0],
            [math.sin(angle), math.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    rotated = StructureBatch(
        batch.numbers,
        batch.positions @ rotation.T,
        batch.cells @ rotation.T,
        batch.pbc,
        batch.offsets,
        batch.ids,
    )
    np.testing.assert_allclose(calculator.compute(rotated).values, reference, atol=2e-5)

    order = np.array([2, 0, 1, 3])
    permuted = StructureBatch(
        batch.numbers[order],
        batch.positions[order],
        batch.cells,
        batch.pbc,
        batch.offsets,
        batch.ids,
    )
    np.testing.assert_allclose(calculator.compute(permuted).values, reference[order], atol=2e-5)


def test_dpa4_rejects_project_native_archive(tmp_path):
    import torch

    path = tmp_path / "mdescriptor.pt"
    torch.save({"format": "mdescriptor.dpa4.v1", "config": {}, "state_dict": {}}, path)

    with pytest.raises(ModelLoadError, match="failed to load DPA4 model"):
        DPA4(model=path)
