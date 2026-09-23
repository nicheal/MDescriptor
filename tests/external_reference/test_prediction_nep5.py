"""NEP5 energy and force comparisons against pinned NEPAdapters 1.0.2."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from ase import Atoms

from mdescriptor import ExecutionOptions, StructureBatch
from mdescriptor.models import ModelResource
from mdescriptor.predictors import NEP

pytestmark = [pytest.mark.reference, pytest.mark.nepadapters, pytest.mark.model]

MODEL = ModelResource.explicit(Path(__file__).resolve().parents[1] / "data" / "nep5_prediction.txt")


def _system() -> Atoms:
    # The model declares species in H, O order while this structure starts with O.
    return Atoms(
        "OHH",
        positions=[[0.2, 0.3, 0.4], [1.3, 0.3, 0.4], [0.2, 1.5, 0.4]],
        cell=[[12.0, 0.0, 0.0], [1.5, 12.0, 0.0], [0.3, 0.4, 12.0]],
        pbc=True,
    )


def _reference(system: Atoms) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import nep_adapters
    from nep_adapters import NEPCalculator

    if getattr(nep_adapters, "__version__", None) != "1.0.2":
        pytest.fail("NEP5 reference requires nep-adapters==1.0.2", pytrace=False)
    calculator = NEPCalculator(str(MODEL.path))
    try:
        result = calculator.predict_structures([system])
        return (
            np.asarray(result.energy, dtype=np.float64),
            np.asarray(result.potential, dtype=np.float64),
            np.asarray(result.forces, dtype=np.float64),
        )
    finally:
        calculator.close()


def _assert_matches_reference(device: str) -> None:
    batch = StructureBatch.from_ase(_system())
    expected_energy, expected_atom_energy, expected_forces = _reference(_system())
    predictor = NEP(model=MODEL, execution=ExecutionOptions(device=device))
    try:
        actual = predictor.predict(batch)
    finally:
        predictor.close()

    rtol, atol = (1e-6, 1e-6) if device == "cpu" else (1e-5, 1e-4)
    np.testing.assert_allclose(actual.energy, expected_energy, rtol=rtol, atol=atol)
    np.testing.assert_allclose(actual.atom_energy, expected_atom_energy, rtol=rtol, atol=atol)
    np.testing.assert_allclose(actual.forces, expected_forces, rtol=rtol, atol=atol)
    np.testing.assert_allclose(
        actual.energy,
        np.add.reduceat(actual.atom_energy, actual.offsets[:-1]),
        rtol=1e-12,
        atol=1e-12,
    )


def test_nep5_energy_atom_energy_and_forces_match_nep_adapters():
    _assert_matches_reference("cpu")


@pytest.mark.gpu
def test_nep5_cuda_energy_atom_energy_and_forces_match_nep_adapters():
    from tests._cuda import load_cuda_for_tests

    load_cuda_for_tests()
    _assert_matches_reference("cuda")
