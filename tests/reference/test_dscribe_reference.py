"""Pinned DScribe 2.1.2 comparisons used by the reference CI job."""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms

from mdescriptor import StructureBatch
from mdescriptor.descriptors import (
    ACSF,
    LMBTR,
    MBTR,
    SOAP,
    CoulombMatrix,
    EwaldSumMatrix,
    SineMatrix,
    ValleOganov,
)

pytestmark = pytest.mark.reference


def _rows(value: np.ndarray) -> np.ndarray:
    """Normalize DScribe's single-structure output to MDescriptor's row layout."""

    result = np.asarray(value)
    if result.ndim == 1:
        return result.reshape(1, -1)
    if result.ndim != 2:
        raise AssertionError(f"expected a one- or two-dimensional reference, got {result.shape}")
    return result


def _water() -> Atoms:
    return Atoms(
        "OHH",
        positions=[[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]],
        cell=np.diag([8.0, 8.0, 8.0]),
        pbc=True,
    )


def test_soap_and_acsf_match_pinned_dscribe():
    from dscribe.descriptors import ACSF as DscribeACSF
    from dscribe.descriptors import SOAP as DscribeSOAP

    system = _water()
    acsf_parameters = {
        "r_cut": 3.5,
        "g2_params": [[0.4, 0.0], [1.0, 0.5]],
        "g3_params": [0.7, 1.3],
        "g4_params": [[0.2, 1.0, 1.0], [0.5, 2.0, -1.0]],
        "g5_params": [[0.3, 1.0, 1.0], [0.6, 2.0, -1.0]],
        "species": [1, 8],
        "periodic": True,
    }
    expected_acsf = DscribeACSF(**acsf_parameters).create(system)
    actual_acsf = ACSF(
        **{key: value for key, value in acsf_parameters.items() if key != "periodic"}
    ).compute(StructureBatch.from_ase(system)).values
    np.testing.assert_allclose(actual_acsf, expected_acsf, rtol=1e-9, atol=1e-10)

    soap_parameters = {
        "r_cut": 3.5,
        "n_max": 3,
        "l_max": 2,
        "sigma": 0.5,
        "average": "off",
        "species": [1, 8],
        "periodic": True,
        "rbf": "gto",
        "compression": {"mode": "off", "species_weighting": None},
    }
    expected_soap = DscribeSOAP(**soap_parameters).create(system)
    actual_soap = SOAP(
        **{key: value for key, value in soap_parameters.items() if key != "periodic"}
    ).compute(StructureBatch.from_ase(system)).values
    np.testing.assert_allclose(actual_soap, expected_soap, rtol=1e-8, atol=1e-9)


def test_matrices_match_pinned_dscribe():
    from dscribe.descriptors import (
        CoulombMatrix as DscribeCoulombMatrix,
    )
    from dscribe.descriptors import (
        EwaldSumMatrix as DscribeEwaldSumMatrix,
    )
    from dscribe.descriptors import (
        SineMatrix as DscribeSineMatrix,
    )

    system = Atoms(
        "NaCl",
        positions=[[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]],
        cell=np.diag([8.0, 8.0, 8.0]),
        pbc=True,
    )
    batch = StructureBatch.from_ase(system)
    for native, reference in (
        (
            CoulombMatrix(n_atoms_max=3, permutation="none").compute(batch).values,
            _rows(DscribeCoulombMatrix(n_atoms_max=3, permutation="none").create(system)),
        ),
        (
            SineMatrix(n_atoms_max=3, permutation="none").compute(batch).values,
            _rows(DscribeSineMatrix(n_atoms_max=3, permutation="none").create(system)),
        ),
    ):
        np.testing.assert_allclose(native, reference, rtol=1e-9, atol=1e-11)

    parameters = {"accuracy": 1e-5, "w": 1.0, "r_cut": 6.0, "g_cut": 3.0, "a": 0.3}
    native = EwaldSumMatrix(n_atoms_max=2, permutation="none", **parameters).compute(batch).values
    reference = _rows(
        DscribeEwaldSumMatrix(n_atoms_max=2, permutation="none").create(
            system, **parameters
        )
    )
    np.testing.assert_allclose(native, reference, rtol=1e-9, atol=1e-11)


def test_mbtr_and_valle_oganov_match_pinned_dscribe():
    from dscribe.descriptors import (
        LMBTR as DscribeLMBTR,
    )
    from dscribe.descriptors import (
        MBTR as DscribeMBTR,
    )
    from dscribe.descriptors import (
        ValleOganov as DscribeValleOganov,
    )

    system = _water()
    batch = StructureBatch.from_ase(system)
    common = {
        "species": [1, 8],
        "geometry": {"function": "distance"},
        "grid": {"min": 0.0, "max": 4.0, "n": 20, "sigma": 0.1},
        "weighting": {"function": "exp", "scale": 0.3, "threshold": 1e-3},
        "periodic": True,
    }
    expected_mbtr = _rows(DscribeMBTR(**common).create(system))
    actual_mbtr = MBTR(**common).compute(batch).values
    np.testing.assert_allclose(actual_mbtr, expected_mbtr, rtol=1e-9, atol=1e-11)

    expected_lmbtr = _rows(DscribeLMBTR(**common).create(system))
    actual_lmbtr = LMBTR(**common).compute(batch).values
    np.testing.assert_allclose(actual_lmbtr, expected_lmbtr, rtol=1e-9, atol=1e-11)

    parameters = {"species": [1, 8], "function": "distance", "n": 20, "sigma": 0.1, "r_cut": 3.5}
    expected_valle = _rows(DscribeValleOganov(**parameters).create(system))
    actual_valle = ValleOganov(**parameters).compute(batch).values
    np.testing.assert_allclose(actual_valle, expected_valle, rtol=1e-9, atol=1e-11)
