import json
from pathlib import Path

import numpy as np
import pytest

from mdescriptor.core.errors import ModelLoadError
from mdescriptor.descriptors.model_backed._vendor.dpa4desc.weights import (
    load_torch_checkpoint,
)
from mdescriptor.descriptors.model_backed.dpa import validate_dpa_checkpoint_mapping
from mdescriptor.models import DPA4C_MODEL
from tests._public import DPA4C, StructureBatch

pytestmark = pytest.mark.model

ROOT = Path(__file__).parents[1]
MODEL = DPA4C_MODEL
GOLDEN = ROOT / "tests" / "data" / "dpa4c_air_h2o_golden.json"

def _fixture() -> tuple[StructureBatch, np.ndarray]:
    payload = json.loads(GOLDEN.read_text(encoding="utf-8"))
    batch = StructureBatch(
        np.asarray(payload["numbers"], dtype=np.int32),
        np.asarray(payload["positions"], dtype=np.float64),
        np.asarray(payload["cells"], dtype=np.float64),
        np.asarray(payload["pbc"], dtype=np.int32),
        np.asarray(payload["offsets"], dtype=np.int64),
        ("golden-0", "golden-1"),
    )
    expected = np.asarray(payload["values"], dtype=np.float64).reshape(-1, 219)
    return batch, expected


def test_dpa4c_matches_official_golden_fixture():
    batch, expected = _fixture()
    result = DPA4C(model=MODEL).compute(batch)
    assert result.values.shape == (6, 219)
    assert result.level == "atom"
    assert result.metadata["backend"] == "mdescriptor-dpa4c-numpy"
    assert len(result.labels) == 219
    np.testing.assert_allclose(result.values, expected, rtol=2e-5, atol=1e-6)


def test_dpa4c_is_rotation_and_atom_permutation_invariant():
    batch, _ = _fixture()
    calculator = DPA4C(model=MODEL)
    baseline = calculator.compute(batch).values

    rotation = np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    positions = batch.positions.copy()
    cells = batch.cells.copy()
    positions[:3] = positions[:3] @ rotation.T
    cells[0] = cells[0] @ rotation.T
    rotated = StructureBatch(batch.numbers, positions, cells, batch.pbc, batch.offsets, batch.ids)
    np.testing.assert_allclose(calculator.compute(rotated).values, baseline, rtol=2e-5, atol=1e-6)

    order = np.asarray([1, 2, 0, 3, 4, 5])
    permuted = StructureBatch(
        batch.numbers[order],
        batch.positions[order],
        batch.cells,
        batch.pbc,
        batch.offsets,
        batch.ids,
    )
    expected = baseline[order]
    np.testing.assert_allclose(calculator.compute(permuted).values, expected, rtol=2e-5, atol=1e-6)


def test_dpa4c_maps_atomic_numbers_through_checkpoint_type_map():
    batch, _ = _fixture()
    calculator = DPA4C(model=MODEL)
    with pytest.raises(ValueError, match="absent from the checkpoint type_map"):
        calculator.compute(
            StructureBatch(
                batch.numbers.copy() * 0 + 119,
                batch.positions,
                batch.cells,
                batch.pbc,
                batch.offsets,
                batch.ids,
            )
        )


def test_dpa4c_uses_the_bundled_checkpoint_by_default():
    calculator = DPA4C()
    assert calculator.model_path.endswith("DPA4C-Air-OMat24-v20260819.pt")


def test_dpa4c_calibration_is_an_explicit_runtime_option():
    batch, _ = _fixture()
    calibrated = DPA4C(model=MODEL, calibrate=True).compute(batch)
    raw = DPA4C(model=MODEL, calibrate=False).compute(batch)
    assert calibrated.metadata["details"]["calibrated"] is True
    assert raw.metadata["details"]["calibrated"] is False
    assert not np.allclose(calibrated.values, raw.values)


def test_dpa4c_default_calibration_is_frozen_in_configuration():
    descriptor = DPA4C(model=MODEL)
    try:
        assert descriptor.configuration.parameters["calibrate"] is True
    finally:
        descriptor.close()


@pytest.mark.parametrize("mutation", ["missing", "shape", "dtype", "type_map"])
def test_dpa4c_checkpoint_schema_failures_are_model_load_errors(mutation):
    checkpoint = load_torch_checkpoint(str(MODEL))
    if mutation == "type_map":
        checkpoint["model"]["_extra_state"]["model_params"].pop("type_map")
    else:
        state = checkpoint["model"]
        key = next(
            name
            for name in state
            if name.endswith("descriptor.readout.gram_scale")
        )
        if mutation == "missing":
            del state[key]
        elif mutation == "shape":
            state[key] = state[key][:-1]
        else:
            state[key] = state[key].astype(np.float64)
    with pytest.raises(ModelLoadError):
        validate_dpa_checkpoint_mapping(checkpoint, expected_descriptor="DPA4C")
