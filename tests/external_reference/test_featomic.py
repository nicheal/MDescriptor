"""Featomic 0.6.6 comparisons for the local descriptor family."""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from scripts.external_reference import (
    _MAX_ANGULAR,
    _MAX_RADIAL,
    _SPECIES,
    _block_data,
    _flatten_atomic_composition,
    _flatten_power_spectrum,
    _flatten_radial_spectrum,
    _flatten_sorted_distances,
    _flatten_spherical_expansion,
    _flatten_spherical_expansion_by_pair,
)

from mdescriptor import StructureBatch
from mdescriptor.descriptors import (
    AtomicComposition,
    LodeSphericalExpansion,
    NeighborList,
    SoapPowerSpectrum,
    SoapRadialSpectrum,
    SortedDistances,
    SphericalExpansion,
    SphericalExpansionByPair,
)

pytestmark = [pytest.mark.reference, pytest.mark.featomic]

_CUTOFF = 3.5


def _require_featomic():
    """Fail the selected reference job when its provider is unavailable."""

    try:
        import featomic
    except ImportError as exc:  # pragma: no cover - exercised in misconfigured CI
        pytest.fail(
            "Featomic reference job requires featomic==0.6.6; "
            f"import failed: {exc}",
            pytrace=False,
        )
    return featomic


def _water() -> Atoms:
    return Atoms(
        "OHH",
        positions=[[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]],
        cell=np.diag([8.0, 8.0, 8.0]),
        pbc=True,
    )


def _neighbor_rows(tensor_map) -> np.ndarray:
    rows = []
    for key in tensor_map.keys:
        samples, vectors = _block_data(tensor_map[key])
        vectors = vectors.reshape(samples.shape[0], 3)
        rows.append(np.column_stack((samples, vectors, np.linalg.norm(vectors, axis=1))))
    if not rows:
        return np.empty((0, 10), dtype=np.float64)
    return np.concatenate(rows, axis=0)


def _sort_neighbor_rows(rows: np.ndarray) -> np.ndarray:
    if len(rows) == 0:
        return rows
    order = np.lexsort(tuple(rows[:, column] for column in range(5, -1, -1)))
    return rows[order]


def test_basic_local_descriptors_match_featomic():
    featomic = _require_featomic()
    system = _water()
    batch = StructureBatch.from_ase(system)

    expected = _flatten_atomic_composition(
        featomic.AtomicComposition(per_system=False).compute(system), len(system), per_system=False
    )
    actual = AtomicComposition(species=_SPECIES, per_system=False).compute(batch).values
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    expected = _flatten_sorted_distances(
        featomic.SortedDistances(
            cutoff=_CUTOFF, max_neighbors=4, separate_neighbor_types=True
        ).compute(system),
        system.numbers,
    )
    actual = SortedDistances(
        species=_SPECIES,
        cutoff=_CUTOFF,
        max_neighbors=4,
        separate_neighbor_types=True,
    ).compute(batch).values
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    expected = _neighbor_rows(
        featomic.NeighborList(cutoff=_CUTOFF, full_neighbor_list=True).compute(system)
    )
    actual_result = NeighborList(cutoff=_CUTOFF, full_neighbor_list=True).compute(batch)
    actual = np.column_stack((np.asarray(actual_result.samples), actual_result.values))
    np.testing.assert_allclose(
        _sort_neighbor_rows(actual), _sort_neighbor_rows(expected), rtol=1e-12, atol=1e-12
    )


def test_spherical_expansion_family_matches_featomic():
    featomic = _require_featomic()
    from featomic.basis import Gto, TensorProduct
    from featomic.cutoff import Cutoff, ShiftedCosine
    from featomic.density import Gaussian

    system = _water()
    batch = StructureBatch.from_ase(system)
    cutoff = Cutoff(_CUTOFF, ShiftedCosine(width=0.5))
    density = Gaussian(width=0.6)
    basis = TensorProduct(
        max_angular=_MAX_ANGULAR,
        radial=Gto(max_radial=_MAX_RADIAL, radius=_CUTOFF),
    )

    expected = _flatten_spherical_expansion(
        featomic.SphericalExpansion(cutoff=cutoff, density=density, basis=basis).compute(system),
        len(system),
    )
    actual = SphericalExpansion(
        species=_SPECIES,
        cutoff=_CUTOFF,
        density_width=0.6,
        max_radial=_MAX_RADIAL,
        max_angular=_MAX_ANGULAR,
    ).compute(batch).values
    np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=5e-8)

    expected = _flatten_spherical_expansion_by_pair(
        featomic.SphericalExpansionByPair(
            cutoff=cutoff, density=density, basis=basis
        ).compute(system),
        np.asarray(system.numbers, dtype=np.int64),
    )
    actual = SphericalExpansionByPair(
        species=_SPECIES,
        cutoff=_CUTOFF,
        density_width=0.6,
        max_radial=_MAX_RADIAL,
        max_angular=_MAX_ANGULAR,
    ).compute(batch).values
    np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=5e-8)

    expected = _flatten_power_spectrum(
        featomic.SoapPowerSpectrum(cutoff=cutoff, density=density, basis=basis).compute(system),
        len(system),
    )
    actual = SoapPowerSpectrum(
        species=_SPECIES,
        cutoff=_CUTOFF,
        density_width=0.6,
        max_radial=_MAX_RADIAL,
        max_angular=_MAX_ANGULAR,
    ).compute(batch).values
    np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=5e-8)

    expected = _flatten_radial_spectrum(
        featomic.SoapRadialSpectrum(
            cutoff=cutoff,
            density=density,
            basis={"radial": Gto(max_radial=_MAX_RADIAL, radius=_CUTOFF)},
        ).compute(system),
        len(system),
    )
    actual = SoapRadialSpectrum(
        species=_SPECIES,
        cutoff=_CUTOFF,
        density_width=0.6,
        max_radial=_MAX_RADIAL,
        max_angular=_MAX_ANGULAR,
    ).compute(batch).values
    np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=5e-8)


def test_lode_spherical_expansion_matches_featomic():
    featomic = _require_featomic()
    from featomic.basis import Gto, TensorProduct
    from featomic.density import SmearedPowerLaw

    system = _water()
    batch = StructureBatch.from_ase(system)
    basis = TensorProduct(
        max_angular=_MAX_ANGULAR,
        radial=Gto(max_radial=_MAX_RADIAL, radius=_CUTOFF),
    )
    expected = _flatten_spherical_expansion(
        featomic.LodeSphericalExpansion(
            density=SmearedPowerLaw(smearing=0.5, exponent=1),
            basis=basis,
            k_cutoff=2.5,
        ).compute(system),
        len(system),
    )
    actual = LodeSphericalExpansion(
        species=_SPECIES,
        cutoff=_CUTOFF,
        density_width=0.5,
        max_radial=_MAX_RADIAL,
        max_angular=_MAX_ANGULAR,
        k_cutoff=2.5,
        exponent=1,
        radial_radius=_CUTOFF,
    ).compute(batch).values
    np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=5e-8)
