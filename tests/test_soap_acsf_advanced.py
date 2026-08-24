import numpy as np
import pytest
from ase import Atoms

from mdescriptor import AcsfCalculator, SoapCalculator, StructureBatch


def _system():
    return Atoms(
        "OHH",
        positions=[[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]],
        cell=np.diag([8.0, 8.0, 8.0]),
        pbc=True,
    )


def test_acsf_g1_to_g5_matches_reference():
    pytest.importorskip("dscribe")
    from dscribe.descriptors import ACSF

    system = _system()
    parameters = {
        "r_cut": 3.5,
        "g2_params": [[0.4, 0.0], [1.0, 0.5]],
        "g3_params": [0.7, 1.3],
        "g4_params": [[0.2, 1.0, 1.0], [0.5, 2.0, -1.0]],
        "g5_params": [[0.3, 1.0, 1.0], [0.6, 2.0, -1.0]],
        "species": [1, 8],
        "periodic": True,
    }
    expected = ACSF(**parameters).create(system)
    actual = AcsfCalculator(**parameters).compute(StructureBatch.from_ase(system)).values
    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-10)
    assert actual.shape == (3, 22)


@pytest.mark.parametrize(
    "parameters",
    [
        {
            "rbf": "gto",
            "weighting": {"function": "poly", "r0": 3.5, "c": 1.2, "m": 2.0, "w0": 0.8},
            "compression": {"mode": "off", "species_weighting": None},
        },
        {
            "rbf": "gto",
            "weighting": {"function": "pow", "r0": 1.5, "c": 1.1, "d": 0.3, "m": 2.0, "w0": 0.8},
            "compression": {"mode": "mu2", "species_weighting": {1: 1.2, 8: 0.7}},
        },
        {
            "rbf": "gto",
            "weighting": {"function": "exp", "r0": 1.5, "c": 1.1, "d": 0.3, "w0": 0.8},
            "compression": {"mode": "mu1nu1", "species_weighting": {1: 1.2, 8: 0.7}},
        },
        {
            "rbf": "gto",
            "weighting": None,
            "compression": {"mode": "crossover", "species_weighting": {1: 1.2, 8: 0.7}},
        },
        {
            "rbf": "polynomial",
            "weighting": None,
            "compression": {"mode": "off", "species_weighting": None},
        },
    ],
)
def test_soap_advanced_parameters_match_reference(parameters):
    pytest.importorskip("dscribe")
    from dscribe.descriptors import SOAP

    system = _system()
    common = {
        "r_cut": 3.5,
        "n_max": 3,
        "l_max": 2,
        "sigma": 0.5,
        "average": "off",
        "species": [1, 8],
        "periodic": True,
        **parameters,
    }
    expected = SOAP(**common).create(system)
    result = SoapCalculator(**common).compute(StructureBatch.from_ase(system))
    np.testing.assert_allclose(result.values, expected, rtol=1e-8, atol=1e-9)
    assert result.values.shape[1] == len(result.labels)


def test_descriptor_dtype_is_preserved():
    system = _system()
    batch = StructureBatch.from_ase(system)
    acsf = AcsfCalculator(species=[1, 8], dtype="float32").compute(batch)
    soap = SoapCalculator(species=[1, 8], r_cut=3.5, n_max=2, l_max=1, dtype="float32").compute(batch)
    assert acsf.values.dtype == np.float32
    assert soap.values.dtype == np.float32


def test_sparse_output_matches_dense_values():
    pytest.importorskip("sparse")
    system = _system()
    batch = StructureBatch.from_ase(system)
    for sparse_calculator, dense_calculator in (
        (AcsfCalculator(species=[1, 8], sparse=True), AcsfCalculator(species=[1, 8])),
        (
            SoapCalculator(species=[1, 8], r_cut=3.5, n_max=2, l_max=1, sparse=True),
            SoapCalculator(species=[1, 8], r_cut=3.5, n_max=2, l_max=1),
        ),
    ):
        result = sparse_calculator.compute(batch)
        assert result.values.__class__.__name__ == "COO"
        dense = dense_calculator.compute(batch).values
        np.testing.assert_allclose(result.values.todense(), dense)
