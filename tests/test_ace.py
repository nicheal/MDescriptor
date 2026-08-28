"""Contract and symmetry checks for the ACE1-compatible descriptor."""

from __future__ import annotations

import inspect

import numpy as np
import pytest
from ase import Atoms

from mdescriptor import (
    DescriptorConfigError,
    DescriptorConfiguration,
    OutputOptions,
    StructureBatch,
    create_descriptor,
)
from mdescriptor.descriptors import ACE


def _water_batch() -> StructureBatch:
    cell = np.diag([14.0, 14.0, 14.0])
    positions = np.asarray(
        [[7.0, 7.0, 7.0], [7.8, 7.3, 7.0], [6.3, 7.4, 7.0]],
        dtype=np.float64,
    )
    rotation = np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    rotated = (positions - 7.0) @ rotation.T + 7.0
    systems = [
        Atoms("OHH", positions=positions, cell=cell, pbc=True),
        Atoms("OHH", positions=rotated, cell=cell @ rotation.T, pbc=True),
        Atoms("OHH", positions=positions + [0.23, -0.17, 0.11], cell=cell, pbc=True),
        Atoms("OHH", positions=positions[[0, 2, 1]], cell=cell, pbc=True),
    ]
    return StructureBatch.from_ase(systems)


def test_ace_has_explicit_ace1_option_surface_and_canonical_species():
    expected = {
        "species", "N", "r0", "trans", "wL", "maxdeg", "D", "rcut", "rin",
        "pcut", "pin", "constants", "output", "execution",
    }
    assert set(inspect.signature(ACE).parameters) == expected
    descriptor = ACE(species=["H", "O"], N=2, maxdeg=4, rcut=3.5)
    assert descriptor.configuration.parameters["species"] == (1, 8)
    assert descriptor.configuration.parameters["trans"] == {
        "type": "PolyTransform", "p": 2.0, "r0": 2.5, "a": 1.0,
    }
    assert descriptor.configuration.parameters["rin"] == 1.25
    descriptor.close()


def test_ace_is_finite_and_invariant_to_rotation_translation_and_permutation():
    result = ACE(species=[1, 8], N=3, maxdeg=5, rcut=3.0, rin=0.0).compute(_water_batch())
    values = np.asarray(result.values)
    assert values.shape[0] == 12
    assert values.shape[1] == result.feature_count > 0
    assert result.level == "atom"
    assert result.labels == tuple(f"ace1:feature={index}" for index in range(values.shape[1]))
    assert np.isfinite(values).all()
    reference = values[:3]
    np.testing.assert_allclose(reference, values[3:6], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(reference, values[6:9], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(reference, values[9:12][[0, 2, 1]], rtol=1e-10, atol=1e-12)


def test_ace_supports_explicit_sparse_degree_and_order_vectors():
    explicit = ACE(
        species=["Si"], N=2, maxdeg=4,
        D={"type": "SparsePSHDegree", "wL": 1.25, "csp": 1.0},
        rcut=3.0, rin=0.0,
    )
    vector = ACE(species=[14], N=2, maxdeg=[4.0, 1.0], wL=[1.25, 1.0], rcut=3.0, rin=0.0)
    assert explicit.feature_count > 0
    assert vector.feature_count > 0
    assert explicit.configuration.parameters["D"]["wL"] == 1.25
    assert vector.configuration.parameters["maxdeg"] == (4.0, 1.0)
    explicit.close()
    vector.close()

    broadcast = ACE(species=[14], N=2, maxdeg=[4.0, 1.0], wL=1.25, rcut=3.0, rin=0.0)
    assert broadcast.configuration.parameters["wL"] == (1.25, 1.25)
    broadcast.close()


def test_ace_order_vectors_round_trip_through_configuration():
    descriptor = ACE(
        species=[14],
        N=2,
        maxdeg=[4.0, 1.0],
        wL=[1.25, 1.0],
        rcut=3.0,
        rin=0.0,
    )
    try:
        configuration = DescriptorConfiguration.from_dict(
            descriptor.configuration.to_dict()
        )
    finally:
        descriptor.close()

    rebuilt = create_descriptor(configuration)
    try:
        assert rebuilt.configuration.to_dict() == configuration.to_dict()
    finally:
        rebuilt.close()


def test_ace_output_options_and_configuration_round_trip():
    descriptor = ACE(
        species=[1, 8], N=1, maxdeg=3, rcut=3.0, rin=0.0,
        output=OutputOptions(dtype="float32"),
    )
    result = descriptor.compute(_water_batch())
    assert np.asarray(result.values).dtype == np.float32
    assert result.metadata["details"]["reference"]["version"] == "0.12.5"
    descriptor.close()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"species": ["not-an-element"]},
        {"species": [1, 1]},
        {"species": [1], "N": 0},
        {"species": [1], "maxdeg": [4.0, 3.0]},
        {"species": [1], "D": {"type": "SparsePSHDegree"}, "maxdeg": [4.0]},
    ],
)
def test_ace_rejects_invalid_configuration(kwargs):
    with pytest.raises(DescriptorConfigError):
        ACE(**kwargs)
