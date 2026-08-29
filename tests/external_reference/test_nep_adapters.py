"""Pinned ``nep-adapters`` comparison for the bundled NEP descriptor."""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms

from mdescriptor import StructureBatch
from mdescriptor.descriptors import NEP
from mdescriptor.models import NEP_MODEL

pytestmark = [pytest.mark.reference, pytest.mark.nepadapters, pytest.mark.model]


def _water() -> Atoms:
    return Atoms(
        "OHH",
        positions=[[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]],
        cell=np.diag([8.0, 8.0, 8.0]),
        pbc=True,
    )


def test_nep_matches_nep_adapters():
    from nep_adapters import NEPCalculator

    system = _water()
    reference = NEPCalculator(str(NEP_MODEL))
    try:
        expected = np.asarray(reference.get_descriptor(system), dtype=np.float64)
    finally:
        reference.close()

    descriptor = NEP(model=NEP_MODEL)
    try:
        actual = np.asarray(
            descriptor.compute(StructureBatch.from_ase(system)).values,
            dtype=np.float64,
        )
    finally:
        descriptor.close()

    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)
