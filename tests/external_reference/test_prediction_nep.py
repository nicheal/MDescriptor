"""Energy and force comparisons against pinned NEPAdapters 1.0.2."""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms

from mdescriptor import ExecutionOptions, StructureBatch
from mdescriptor.core.errors import DescriptorInputError
from mdescriptor.models import NEP_MODEL
from mdescriptor.predictors import NEP

pytestmark = [pytest.mark.reference, pytest.mark.nepadapters, pytest.mark.model]


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
            positions=[[0.0, 0.0, 0.0], [5.99, 0.0, 0.0]],
            cell=np.eye(3) * 18.0,
            pbc=True,
        ),
        Atoms(
            "HO",
            positions=[[0.0, 0.0, 0.0], [6.01, 0.0, 0.0]],
            cell=np.eye(3) * 18.0,
            pbc=True,
        ),
    ]


def _reference(batch: StructureBatch):
    import nep_adapters
    from nep_adapters import NEPCalculator

    if getattr(nep_adapters, "__version__", None) != "1.0.2":
        pytest.fail("NEP prediction reference requires nep-adapters==1.0.2", pytrace=False)
    # Use the public structure API one frame at a time. This avoids relying on
    # predict_arrays' positional pbc argument and works around a mixed-batch
    # segfault in NEPAdapters 1.0.2. A large periodic cell is equivalent to
    # isolation for this model because it puts every image beyond its cutoff.
    reference = NEPCalculator(str(NEP_MODEL))
    try:
        energies: list[float] = []
        atom_energies: list[np.ndarray] = []
        forces: list[np.ndarray] = []
        for index in range(batch.structures):
            begin, end = map(int, batch.offsets[index : index + 2])
            cell = batch.cells[index].reshape(3, 3).copy()
            pbc = batch.pbc[index].astype(bool)
            if not np.all(pbc):
                cell = np.eye(3) * 30.0
                pbc = np.ones(3, dtype=bool)
            atoms = Atoms(
                numbers=batch.numbers[begin:end],
                positions=batch.positions[begin:end],
                cell=cell,
                pbc=pbc,
            )
            result = reference.predict_structures([atoms])
            energies.append(float(result.energy[0]))
            atom_energies.append(np.asarray(result.potential, dtype=np.float64))
            forces.append(np.asarray(result.forces, dtype=np.float64))
        return (
            np.asarray(energies, dtype=np.float64),
            np.concatenate(atom_energies),
            np.concatenate(forces),
        )
    finally:
        reference.close()


def test_nep_energy_atom_energy_and_forces_match_nep_adapters():
    batch = StructureBatch.from_ase(
        _systems(), ids=("isolated", "triclinic", "cutoff_inside", "cutoff_outside")
    )
    expected_energy, expected_atom_energy, expected_forces = _reference(batch)
    predictor = NEP(model=NEP_MODEL, execution=ExecutionOptions(device="cpu"))
    try:
        actual = predictor.predict(batch)
    finally:
        predictor.close()

    np.testing.assert_allclose(actual.energy, expected_energy, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(actual.atom_energy, expected_atom_energy, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(actual.forces, expected_forces, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        actual.energy,
        np.add.reduceat(actual.atom_energy, actual.offsets[:-1]),
        rtol=1e-12,
        atol=1e-12,
    )


def test_nep_predict_accepts_ase_atoms_and_lists():
    predictor = NEP(model=NEP_MODEL, execution=ExecutionOptions(device="cpu"))
    try:
        for value in (_systems()[0], _systems()[:2]):
            batch = StructureBatch.from_ase(value)
            expected_energy, expected_atom_energy, expected_forces = _reference(batch)
            result = predictor.predict(value)

            assert result.offsets.tolist() == batch.offsets.tolist()
            np.testing.assert_allclose(result.energy, expected_energy, rtol=1e-6, atol=1e-6)
            np.testing.assert_allclose(
                result.atom_energy, expected_atom_energy, rtol=1e-6, atol=1e-6
            )
            np.testing.assert_allclose(result.forces, expected_forces, rtol=1e-6, atol=1e-6)
    finally:
        predictor.close()


def test_nep_predict_rejects_species_outside_model():
    batch = StructureBatch(
        np.asarray([119], dtype=np.int32),
        np.zeros((1, 3)),
        np.zeros((1, 3, 3)),
        np.zeros((1, 3), dtype=np.int32),
        np.asarray([0, 1], dtype=np.int64),
        ("unsupported-element",),
    )
    predictor = NEP(model=NEP_MODEL, execution=ExecutionOptions(device="cpu"))
    try:
        with pytest.raises(DescriptorInputError) as error:
            predictor.predict(batch)
        assert error.value.code == "unsupported_species"
        assert error.value.path == ("input", "numbers")
        assert error.value.details == {"atomic_numbers": [119]}
    finally:
        predictor.close()


@pytest.mark.parametrize("device", ("cpu", "cuda"))
def test_nep_predict_rejects_spin_inputs_consistently(device: str):
    batch = StructureBatch.from_ase(_systems()[0])
    predictor = NEP(model=NEP_MODEL, execution=ExecutionOptions(device=device))
    try:
        for name, values, path in (
            ("spins", np.zeros((batch.atoms, 3)), ("input", "spins")),
            (
                "charge_spin",
                np.zeros((batch.structures, 2)),
                ("input", "charge_spin"),
            ),
        ):
            with pytest.raises(DescriptorInputError) as error:
                predictor.predict(
                    StructureBatch(
                        batch.numbers,
                        batch.positions,
                        batch.cells,
                        batch.pbc,
                        batch.offsets,
                        batch.ids,
                        **{name: values},
                    )
                )
            assert error.value.code == "unsupported_input"
            assert error.value.path == path
    finally:
        predictor.close()


@pytest.mark.gpu
def test_nep_cuda_energy_atom_energy_and_forces_match_nep_adapters():
    from tests._cuda import load_cuda_for_tests

    load_cuda_for_tests()
    batch = StructureBatch.from_ase(
        _systems(), ids=("isolated", "triclinic", "cutoff_inside", "cutoff_outside")
    )
    expected_energy, expected_atom_energy, expected_forces = _reference(batch)
    predictor = NEP(model=NEP_MODEL, execution=ExecutionOptions(device="cuda"))
    try:
        actual = predictor.predict(batch)
    finally:
        predictor.close()

    np.testing.assert_allclose(actual.energy, expected_energy, rtol=1e-5, atol=1e-4)
    np.testing.assert_allclose(actual.atom_energy, expected_atom_energy, rtol=1e-5, atol=1e-4)
    np.testing.assert_allclose(actual.forces, expected_forces, rtol=1e-5, atol=1e-4)


def test_nep_force_matches_energy_finite_difference():
    system = _systems()[0]
    batch = StructureBatch.from_ase(system)
    predictor = NEP(model=NEP_MODEL)
    try:
        result = predictor.predict(batch)
        step = 1.0e-5
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
    np.testing.assert_allclose(result.forces[1, 0], finite_difference, rtol=2e-4, atol=2e-4)


def test_nep_zbl_short_range_matches_nep_adapters_and_finite_difference():
    system = Atoms(
        "OH",
        positions=[[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]],
        pbc=False,
    )
    batch = StructureBatch.from_ase(system)
    expected_energy, expected_atom_energy, expected_forces = _reference(batch)
    predictor = NEP(model=NEP_MODEL, execution=ExecutionOptions(device="cpu"))
    try:
        result = predictor.predict(batch)
        np.testing.assert_allclose(result.energy, expected_energy, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(result.atom_energy, expected_atom_energy, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(result.forces, expected_forces, rtol=1e-6, atol=1e-6)

        step = 1.0e-5
        energies = []
        for sign in (1.0, -1.0):
            positions = batch.positions.copy()
            positions[1, 0] += sign * step
            displaced = StructureBatch(
                batch.numbers,
                positions,
                batch.cells,
                batch.pbc,
                batch.offsets,
                batch.ids,
            )
            energies.append(predictor.predict(displaced).energy[0])
    finally:
        predictor.close()
    finite_difference = -(energies[0] - energies[1]) / (2.0 * step)
    np.testing.assert_allclose(result.forces[1, 0], finite_difference, rtol=2e-4, atol=2e-4)
