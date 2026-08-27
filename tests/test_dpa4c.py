import numpy as np
import pytest

from mdescriptor.core.errors import ModelLoadError
from mdescriptor.descriptors.model_backed._vendor.dpa4desc.weights import (
    load_torch_checkpoint,
)
from mdescriptor.descriptors.model_backed.dpa import (
    _frame_inputs,
    compute_batch,
    load_dpa_checkpoint,
    new_runtime,
    validate_dpa_checkpoint_mapping,
)
from mdescriptor.models import DPA4C_MODEL
from tests._public import DPA4C, StructureBatch

pytestmark = pytest.mark.model

MODEL = DPA4C_MODEL

def _fixture() -> StructureBatch:
    return StructureBatch(
        np.asarray([8, 1, 1, 8, 1, 1], dtype=np.int32),
        np.asarray(
            [[4.00, 4.00, 4.00], [4.76, 4.58, 4.00], [3.24, 4.58, 4.00],
             [7.00, 7.00, 7.00], [7.76, 7.58, 7.00], [6.24, 7.58, 7.00]],
            dtype=np.float64,
        ),
        np.stack((np.eye(3) * 12.0, np.eye(3) * 12.0)),
        np.ones((2, 3), dtype=np.int32),
        np.asarray([0, 3, 6], dtype=np.int64),
        ("golden-0", "golden-1"),
    )


def test_dpa4c_matches_official_golden_fixture():
    from tests._golden import assert_descriptor_golden

    assert_descriptor_golden("DPA4C")


def test_dpa4c_is_rotation_and_atom_permutation_invariant():
    batch = _fixture()
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
    batch = _fixture()
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
    batch = _fixture()
    calibrated = DPA4C(model=MODEL, calibrate=True).compute(batch)
    raw = DPA4C(model=MODEL, calibrate=False).compute(batch)
    assert calibrated.metadata["details"]["calibrated"] is True
    assert raw.metadata["details"]["calibrated"] is False
    assert not np.allclose(calibrated.values, raw.values)


def test_dpa4c_reference_uses_descriptor_precision_boundary():
    batch = _fixture()
    _info, checkpoint = load_dpa_checkpoint(MODEL, expected_descriptor="DPA4C")
    evaluator = new_runtime(MODEL, checkpoint)
    actual = compute_batch(evaluator, batch)
    for frame in range(batch.structures):
        begin = int(batch.offsets[frame])
        end = int(batch.offsets[frame + 1])
        symbols = [{1: "H", 8: "O"}[int(number)] for number in batch.numbers[begin:end]]
        atype = evaluator.symbols_to_atype(symbols)
        coord_ext, atype_ext, mapping, nlist = _frame_inputs(
            evaluator,
            batch.positions[begin:end],
            atype,
            batch.cells[frame],
        )
        expected = evaluator.descriptor.call(
            coord_ext,
            atype_ext,
            nlist,
            mapping,
        )[0]
        np.testing.assert_array_equal(
            actual[begin:end],
            np.asarray(expected, dtype=np.float64).reshape(end - begin, -1),
        )


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
