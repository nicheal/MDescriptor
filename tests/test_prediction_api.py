"""Small contract tests for the prediction result and shared lifecycle."""

from __future__ import annotations

import numpy as np
import pytest

from mdescriptor import (
    CancelledError,
    ClosedDescriptorError,
    ComputeControl,
    ExecutionOptions,
    StructureBatch,
)
from mdescriptor.core.errors import DescriptorInputError, ModelLoadError
from mdescriptor.core.prediction_result import PredictionResult
from mdescriptor.models import ModelResource
from mdescriptor.models.session import identity_model_artifact
from mdescriptor.predictors._base import _Predictor
from mdescriptor.predictors.dpa4c import (
    _add_energy_fitting_payload,
    _map_type_indices,
)
from mdescriptor.predictors.nep import NEP


def _batch() -> StructureBatch:
    return StructureBatch(
        np.asarray([1, 8, 1], dtype=np.int32),
        np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        np.zeros((1, 3, 3)),
        np.zeros((1, 3), dtype=np.int32),
        np.asarray([0, 3], dtype=np.int64),
        ("water",),
    )


def test_prediction_result_is_an_owned_read_only_snapshot():
    energy = np.asarray([4.0])
    atom_energy = np.asarray([1.0, 2.0, 1.0])
    forces = np.zeros((3, 3))
    offsets = np.asarray([0, 3], dtype=np.int64)
    result = PredictionResult(
        energy,
        atom_energy,
        forces,
        ("water",),
        offsets,
        {"model": {"digest": "abc"}},
    )
    energy[0] = -1.0
    offsets[1] = 0

    assert result.energy[0] == 4.0
    assert result.offsets[-1] == 3
    for value in (result.energy, result.atom_energy, result.forces, result.offsets):
        assert not value.flags.writeable
    with pytest.raises(ValueError):
        result.forces[0, 0] = 1.0
    with pytest.raises(TypeError):
        result.metadata["new"] = 1
    with pytest.raises(TypeError):
        result.metadata["model"]["digest"] = "changed"


def test_prediction_result_validates_shapes_and_offsets():
    with pytest.raises(ValueError, match="forces"):
        PredictionResult([1.0], [1.0], np.zeros((1, 2)), ("one",), [0, 1])
    with pytest.raises(ValueError, match="offsets"):
        PredictionResult([1.0], [1.0], np.zeros((1, 3)), ("one",), [1, 1])


class _FakePredictor(_Predictor):
    name = "TestPredictor"

    def __init__(self, model: ModelResource):
        super().__init__(
            model,
            ExecutionOptions(),
            default_model=model,
            loader=identity_model_artifact,
        )

    def _create_backend(self):
        return _FakeBackend()


class _FakeBackend:
    def __init__(self):
        self.closed = False

    def predict(self, batch: StructureBatch, control: object):
        assert batch.structures == 1
        assert control is None
        return np.asarray([4.0]), np.asarray([1.0, 2.0, 1.0]), np.zeros((3, 3))

    def close(self):
        self.closed = True


def test_predictor_reuses_batch_lifecycle_control_and_model_metadata(tmp_path):
    model_path = tmp_path / "model.nep"
    model_path.write_bytes(b"test model")
    predictor = _FakePredictor(ModelResource.explicit(model_path))
    result = predictor.predict(_batch())

    assert result.structure_ids == ("water",)
    assert result.metadata["model"]["resolved"]["digest"]
    assert result.metadata["execution"]["device"] == "cpu"
    changed = predictor.metadata
    changed["model"]["resolved"]["digest"] = "mutated"
    assert predictor.metadata["model"]["resolved"]["digest"] != "mutated"

    control = ComputeControl()
    control.cancel()
    with pytest.raises(CancelledError):
        predictor.predict(_batch(), control)
    predictor.close()
    assert predictor.closed
    with pytest.raises(ClosedDescriptorError):
        predictor.predict(_batch())


def test_empty_batch_returns_empty_values_and_completes_control(tmp_path):
    model_path = tmp_path / "model.nep"
    model_path.write_bytes(b"test model")
    predictor = _FakePredictor(ModelResource.explicit(model_path))
    batch = StructureBatch(
        np.empty(0, dtype=np.int32),
        np.empty((0, 3)),
        np.zeros((1, 3, 3)),
        np.zeros((1, 3), dtype=np.int32),
        np.asarray([0, 0], dtype=np.int64),
        ("empty",),
    )
    control = ComputeControl()
    result = predictor.predict(batch, control)
    assert result.energy.tolist() == [0.0]
    assert result.atom_energy.shape == (0,)
    assert result.forces.shape == (0, 3)
    assert control.completed() == control.total() == 1
    predictor.close()


def test_nep_cuda_backend_is_lazy_and_receives_public_control(monkeypatch, tmp_path):
    import mdescriptor._runtime as runtime
    from mdescriptor.predictors import NEP

    model_path = tmp_path / "model.nep"
    model_path.write_bytes(b"test model")
    calls = []

    class FakeCudaPredictor(_FakeBackend):
        def predict(self, batch, control):
            assert isinstance(control, ComputeControl)
            return np.asarray([4.0]), np.asarray([1.0, 2.0, 1.0]), np.zeros((3, 3))

    def create(name, options):
        calls.append((name, options))
        return FakeCudaPredictor()

    monkeypatch.setattr(runtime, "create_cuda_predictor", create)
    predictor = NEP(
        model=model_path,
        execution=ExecutionOptions(device="cuda"),
    )
    assert calls == []
    control = ComputeControl()
    assert predictor.predict(_batch(), control).metadata["execution"]["device"] == "cuda"
    assert calls[0][0] == "NEP"
    assert calls[0][1]["model_data"] == b"test model"
    predictor.close()


def test_dpa4c_fitting_payload_extracts_energy_weights_and_biases():
    model = {
        "_extra_state": {
            "model_params": {
                "fitting_net": {
                    "type": "ener",
                    "neuron": [2],
                    "activation_function": "silu",
                    "numb_fparam": 0,
                    "numb_aparam": 0,
                    "dim_case_embd": 0,
                }
            }
        },
        "model.Default.atomic_model.fitting_net.nets._module_networks.0.layers.0.w": np.eye(
            2, dtype=np.float32
        ),
        "model.Default.atomic_model.fitting_net.nets._module_networks.0.layers.0.b": np.zeros(
            2, dtype=np.float32
        ),
        "model.Default.atomic_model.fitting_net.nets._module_networks.0.layers.1.w": np.ones(
            (2, 1), dtype=np.float32
        ),
        "model.Default.atomic_model.fitting_net.nets._module_networks.0.layers.1.b": np.zeros(
            1, dtype=np.float32
        ),
        "model.Default.atomic_model.fitting_net.bias_atom_e": np.asarray(
            [[0.25], [0.5]], dtype=np.float64
        ),
        "model.Default.atomic_model.out_bias": np.asarray([[[1.0], [2.0]]], dtype=np.float64),
    }
    payload = {"feature_count": 2}
    _add_energy_fitting_payload(payload, {"model": model}, ["H", "O"])

    assert payload["fitting_neurons"].tolist() == [2]
    assert payload["fitting_weights"].shape == (6,)
    assert payload["fitting_biases"].shape == (3,)
    assert payload["fitting_atom_bias"].dtype == np.float64
    assert payload["output_bias"].dtype == np.float64
    np.testing.assert_array_equal(payload["fitting_atom_bias"], [0.25, 0.5])
    np.testing.assert_array_equal(payload["output_bias"], [1.0, 2.0])


def test_dpa4c_fitting_payload_rejects_unsupported_activations():
    with pytest.raises(ModelLoadError, match="SiLU"):
        _add_energy_fitting_payload(
            {"feature_count": 1},
            {
                "model": {
                    "_extra_state": {
                        "model_params": {"fitting_net": {"activation_function": "tanh"}}
                    }
                }
            },
            ["H"],
        )


def test_dpa4c_type_mapping_uses_checkpoint_order_and_reports_missing_species():
    np.testing.assert_array_equal(
        _map_type_indices(np.asarray([8, 1], dtype=np.int32), ("H", "O")),
        [1, 0],
    )
    with pytest.raises(DescriptorInputError, match="absent from the DPA4C type map"):
        _map_type_indices(np.asarray([6], dtype=np.int32), ("H", "O"))


def test_nep_input_validation_rejects_unsupported_species_and_spin_fields():
    predictor = object.__new__(NEP)
    predictor.execution = ExecutionOptions()
    predictor._species = (1, 8)
    predictor._validate_batch(_batch())

    carbon = StructureBatch(
        np.asarray([6], dtype=np.int32),
        np.zeros((1, 3)),
        np.zeros((1, 3, 3)),
        np.zeros((1, 3), dtype=np.int32),
        np.asarray([0, 1], dtype=np.int64),
        ("carbon",),
    )
    with pytest.raises(DescriptorInputError) as unsupported:
        predictor._validate_batch(carbon)
    assert unsupported.value.code == "unsupported_species"
    assert unsupported.value.path == ("input", "numbers")

    batch = _batch()
    with_spins = StructureBatch(
        batch.numbers,
        batch.positions,
        batch.cells,
        batch.pbc,
        batch.offsets,
        batch.ids,
        spins=np.zeros((batch.atoms, 3)),
    )
    with pytest.raises(DescriptorInputError) as spin_error:
        predictor._validate_batch(with_spins)
    assert spin_error.value.path == ("input", "spins")

    with_charge_spin = StructureBatch(
        batch.numbers,
        batch.positions,
        batch.cells,
        batch.pbc,
        batch.offsets,
        batch.ids,
        charge_spin=np.zeros((batch.structures, 2)),
    )
    with pytest.raises(DescriptorInputError) as charge_spin_error:
        predictor._validate_batch(with_charge_spin)
    assert charge_spin_error.value.path == ("input", "charge_spin")
