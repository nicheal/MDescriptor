"""Energy and force comparisons against pinned DeepMD-kit 3.2.0."""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from ase.data import atomic_numbers

from mdescriptor import ExecutionOptions, StructureBatch
from mdescriptor.core.errors import DescriptorInputError
from mdescriptor.models import DPA4C_MODEL
from mdescriptor.predictors import DPA4C

pytestmark = [pytest.mark.reference, pytest.mark.deepmd, pytest.mark.model]


def _systems() -> list[Atoms]:
    return [
        Atoms(
            "OHH",
            positions=[[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]],
            pbc=False,
        ),
        Atoms(
            "OHH",
            positions=[[0.2, 0.3, 0.4], [1.16, 0.3, 0.4], [-0.04, 1.23, 0.4]],
            cell=[[8.0, 0.0, 0.0], [1.2, 8.0, 0.0], [0.4, 0.7, 8.0]],
            pbc=True,
        ),
        Atoms(
            "HO",
            positions=[[0.0, 0.0, 0.0], [5.999, 0.0, 0.0]],
            cell=np.eye(3) * 18.0,
            pbc=True,
        ),
        Atoms(
            "HO",
            positions=[[0.0, 0.0, 0.0], [6.001, 0.0, 0.0]],
            cell=np.eye(3) * 18.0,
            pbc=True,
        ),
    ]


def _reference(batch: StructureBatch):
    import deepmd
    from deepmd.infer import DeepPot

    if getattr(deepmd, "__version__", None) != "3.2.0":
        pytest.fail("DPA4C prediction reference requires deepmd-kit==3.2.0", pytrace=False)
    reference = DeepPot(str(DPA4C_MODEL), neighbor_graph_method="ase")
    types = {
        int(atomic_numbers[symbol]): index for index, symbol in enumerate(reference.get_type_map())
    }
    energy = np.empty(batch.structures, dtype=np.float64)
    atom_energy: list[np.ndarray] = []
    forces: list[np.ndarray] = []
    try:
        for structure in range(batch.structures):
            begin, end = map(int, batch.offsets[structure : structure + 2])
            try:
                atype = np.asarray(
                    [types[int(number)] for number in batch.numbers[begin:end]],
                    dtype=np.int32,
                )
            except KeyError as exc:
                raise AssertionError(f"test atom is absent from DeepMD type_map: {exc}") from exc
            cell = (
                batch.cells[structure].reshape(1, 9) if np.all(batch.pbc[structure] == 1) else None
            )
            value = reference.eval(
                batch.positions[begin:end].reshape(1, -1),
                cell,
                atype,
                atomic=True,
            )
            energy[structure] = float(np.asarray(value[0]).reshape(-1)[0])
            forces.append(np.asarray(value[1], dtype=np.float64).reshape(end - begin, 3))
            atom_energy.append(np.asarray(value[3], dtype=np.float64).reshape(end - begin))
    finally:
        close = getattr(reference, "close", None)
        if callable(close):
            close()
    return energy, np.concatenate(atom_energy), np.concatenate(forces)


def test_dpa4c_energy_atom_energy_and_forces_match_deepmd():
    batch = StructureBatch.from_ase(
        _systems(), ids=("isolated", "triclinic", "cutoff_inside", "cutoff_outside")
    )
    expected_energy, expected_atom_energy, expected_forces = _reference(batch)
    predictor = DPA4C(model=DPA4C_MODEL, execution=ExecutionOptions(device="cpu"))
    try:
        actual = predictor.predict(batch)
    finally:
        predictor.close()

    np.testing.assert_allclose(actual.energy, expected_energy, rtol=2e-5, atol=1e-5)
    np.testing.assert_allclose(actual.atom_energy, expected_atom_energy, rtol=2e-5, atol=1e-5)
    np.testing.assert_allclose(actual.forces, expected_forces, rtol=2e-4, atol=1e-4)
    np.testing.assert_allclose(
        actual.energy,
        np.add.reduceat(actual.atom_energy, actual.offsets[:-1]),
        rtol=2e-6,
        atol=2e-6,
    )


def test_dpa4c_predict_accepts_ase_atoms_and_lists():
    predictor = DPA4C(model=DPA4C_MODEL, execution=ExecutionOptions(device="cpu"))
    try:
        for value in (_systems()[0], _systems()[:2]):
            batch = StructureBatch.from_ase(value)
            expected_energy, expected_atom_energy, expected_forces = _reference(batch)
            result = predictor.predict(value)

            assert result.offsets.tolist() == batch.offsets.tolist()
            np.testing.assert_allclose(result.energy, expected_energy, rtol=2e-5, atol=1e-5)
            np.testing.assert_allclose(
                result.atom_energy, expected_atom_energy, rtol=2e-5, atol=1e-5
            )
            np.testing.assert_allclose(result.forces, expected_forces, rtol=2e-4, atol=1e-4)
    finally:
        predictor.close()


@pytest.mark.parametrize("device", ("cpu", "cuda"))
def test_dpa4c_predict_rejects_spins_and_charge_spin(device: str):
    base = StructureBatch.from_ase(_systems()[0])
    predictor = DPA4C(model=DPA4C_MODEL, execution=ExecutionOptions(device=device))
    try:
        for name, values, path in (
            ("spins", np.zeros((base.atoms, 3)), ("input", "spins")),
            (
                "charge_spin",
                np.zeros((base.structures, 2)),
                ("input", "charge_spin"),
            ),
        ):
            batch = StructureBatch(
                base.numbers,
                base.positions,
                base.cells,
                base.pbc,
                base.offsets,
                base.ids,
                **{name: values},
            )
            with pytest.raises(DescriptorInputError) as error:
                predictor.predict(batch)
            assert error.value.code == "unsupported_input"
            assert error.value.path == path
    finally:
        predictor.close()


@pytest.mark.parametrize("device", ("cpu", "cuda"))
def test_dpa4c_predict_rejects_species_outside_checkpoint_type_map(device: str):
    predictor = DPA4C(model=DPA4C_MODEL, execution=ExecutionOptions(device=device))
    batch = StructureBatch(
        np.asarray([119], dtype=np.int32),
        np.zeros((1, 3)),
        np.zeros((1, 3, 3)),
        np.zeros((1, 3), dtype=np.int32),
        np.asarray([0, 1], dtype=np.int64),
        ("unsupported-element",),
    )
    try:
        with pytest.raises(DescriptorInputError) as error:
            predictor.predict(batch)
        assert error.value.path == ("input", "numbers")
        assert "atomic number 119 has no known element symbol" in str(error.value)
    finally:
        predictor.close()


@pytest.mark.gpu
def test_dpa4c_cuda_energy_atom_energy_and_forces_match_deepmd():
    from tests._cuda import load_cuda_for_tests

    load_cuda_for_tests()
    batch = StructureBatch.from_ase(
        _systems(), ids=("isolated", "triclinic", "cutoff_inside", "cutoff_outside")
    )
    expected_energy, expected_atom_energy, expected_forces = _reference(batch)
    predictor = DPA4C(model=DPA4C_MODEL, execution=ExecutionOptions(device="cuda"))
    try:
        actual = predictor.predict(batch)
    finally:
        predictor.close()

    np.testing.assert_allclose(actual.energy, expected_energy, rtol=2e-5, atol=1e-5)
    np.testing.assert_allclose(actual.atom_energy, expected_atom_energy, rtol=2e-5, atol=1e-5)
    np.testing.assert_allclose(actual.forces, expected_forces, rtol=2e-4, atol=1e-4)


@pytest.mark.gpu
@pytest.mark.parametrize("widths", ([], [257]), ids=("linear", "wide-hidden"))
def test_dpa4c_cuda_fitting_supports_linear_and_wide_layers(monkeypatch, widths):
    from tests._cuda import load_cuda_for_tests

    load_cuda_for_tests()
    original_builder = DPA4C._build_predictor_payload

    def custom_builder(self):
        type_map, type_numbers, payload = original_builder(self)
        feature_count = payload["feature_count"]
        if widths:
            rng = np.random.default_rng(17)
            hidden_weights = rng.normal(0.0, 0.015, (feature_count, widths[0])).astype(np.float32)
            hidden_bias = rng.normal(0.0, 0.01, widths[0]).astype(np.float32)
            output_weights = rng.normal(0.0, 0.02, widths[0]).astype(np.float32)
            payload["fitting_weights"] = np.concatenate(
                (hidden_weights.reshape(-1), output_weights)
            )
            payload["fitting_biases"] = np.concatenate(
                (hidden_bias, np.asarray([0.01], dtype=np.float32))
            )
        else:
            payload["fitting_weights"] = np.linspace(-0.02, 0.02, feature_count, dtype=np.float32)
            payload["fitting_biases"] = np.asarray([0.01], dtype=np.float32)
        payload["fitting_neurons"] = np.asarray(widths, dtype=np.int32)
        return type_map, type_numbers, payload

    monkeypatch.setattr(DPA4C, "_build_predictor_payload", custom_builder)
    batch = StructureBatch.from_ase(_systems()[0])
    cpu = DPA4C(model=DPA4C_MODEL, execution=ExecutionOptions(device="cpu"))
    try:
        expected = cpu.predict(batch)
    finally:
        cpu.close()
    cuda = DPA4C(model=DPA4C_MODEL, execution=ExecutionOptions(device="cuda"))
    try:
        actual = cuda.predict(batch)
    finally:
        cuda.close()

    np.testing.assert_allclose(actual.energy, expected.energy, rtol=2e-5, atol=1e-5)
    np.testing.assert_allclose(actual.atom_energy, expected.atom_energy, rtol=2e-5, atol=1e-5)
    np.testing.assert_allclose(actual.forces, expected.forces, rtol=2e-4, atol=1e-4)


def test_dpa4c_force_matches_energy_finite_difference():
    batch = StructureBatch.from_ase(_systems()[0])
    predictor = DPA4C(model=DPA4C_MODEL)
    try:
        result = predictor.predict(batch)
        step = 2.0e-3
        displaced = []
        for sign in (1.0, -1.0):
            positions = batch.positions.copy()
            positions[1, 0] += sign * step
            displaced.append(
                StructureBatch(
                    batch.numbers,
                    positions,
                    batch.cells,
                    batch.pbc,
                    batch.offsets,
                    batch.ids,
                )
            )
        plus, minus = (predictor.predict(item).energy[0] for item in displaced)
    finally:
        predictor.close()
    finite_difference = -(plus - minus) / (2.0 * step)
    np.testing.assert_allclose(result.forces[1, 0], finite_difference, rtol=2e-2, atol=2e-2)
