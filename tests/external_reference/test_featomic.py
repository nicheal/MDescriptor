"""Featomic 0.6.6 comparisons for the local descriptor family."""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms

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


_SPECIES = (1, 8)
_CUTOFF = 3.5
_MAX_RADIAL = 2
_MAX_ANGULAR = 2
_RADIAL_COUNT = _MAX_RADIAL + 1


def _water() -> Atoms:
    return Atoms(
        "OHH",
        positions=[[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]],
        cell=np.diag([8.0, 8.0, 8.0]),
        pbc=True,
    )


def _block_data(block) -> tuple[np.ndarray, np.ndarray]:
    samples = np.asarray(block.samples.values, dtype=np.int64)
    raw_values = np.asarray(block.values, dtype=np.float64)
    width = int(np.prod(raw_values.shape[1:], dtype=np.int64))
    values = raw_values.reshape(samples.shape[0], width)
    return samples, values


def _keys_and_blocks(tensor_map):
    return {
        tuple(int(value) for value in key): _block_data(tensor_map[key])
        for key in tensor_map.keys
    }


def _values_for_atom(data, atom: int, width: int) -> np.ndarray:
    samples, values = data
    matches = np.flatnonzero((samples[:, 0] == 0) & (samples[:, 1] == atom))
    if len(matches) == 0:
        return np.zeros(width, dtype=np.float64)
    if len(matches) != 1:
        raise AssertionError(f"expected one reference row for atom {atom}, got {len(matches)}")
    return values[matches[0]]


def _flatten_atomic_composition(tensor_map, atom_count: int, *, per_system: bool) -> np.ndarray:
    blocks = _keys_and_blocks(tensor_map)
    if per_system:
        result = np.zeros((1, len(_SPECIES)), dtype=np.float64)
        for column, species in enumerate(_SPECIES):
            samples, values = blocks[(species,)]
            if samples.shape != (1, 1):
                raise AssertionError(f"unexpected per-system samples for {species}: {samples.shape}")
            result[0, column] = values[0, 0]
        return result

    result = np.zeros((atom_count, len(_SPECIES)), dtype=np.float64)
    for column, species in enumerate(_SPECIES):
        samples, values = blocks[(species,)]
        for sample, value in zip(samples, values, strict=True):
            result[int(sample[1]), column] = value[0]
    return result


def _flatten_sorted_distances(tensor_map, center_species: np.ndarray) -> np.ndarray:
    blocks = _keys_and_blocks(tensor_map)
    result = np.zeros((len(center_species), len(_SPECIES) * 4), dtype=np.float64)
    for atom, center in enumerate(center_species):
        offset = 0
        for neighbor in _SPECIES:
            data = blocks.get(
                (int(center), neighbor),
                (np.empty((0, 2)), np.empty((0, 4))),
            )
            result[atom, offset : offset + 4] = _values_for_atom(data, atom, 4)
            offset += 4
    return result


def _flatten_spherical_expansion(tensor_map, atom_count: int) -> np.ndarray:
    blocks = _keys_and_blocks(tensor_map)
    group_width = sum((2 * angular + 1) * _RADIAL_COUNT for angular in range(_MAX_ANGULAR + 1))
    result = np.zeros((atom_count, len(_SPECIES) * len(_SPECIES) * group_width), dtype=np.float64)
    offset = 0
    for center in _SPECIES:
        for neighbor in _SPECIES:
            for angular in range(_MAX_ANGULAR + 1):
                candidates = [
                    (key, data)
                    for key, data in blocks.items()
                    if key[0] == angular and key[2:] == (center, neighbor)
                ]
                if len(candidates) > 1:
                    raise AssertionError(f"multiple Featomic blocks for {(angular, center, neighbor)}")
                width = (2 * angular + 1) * _RADIAL_COUNT
                data = candidates[0][1] if candidates else (np.empty((0, 2)), np.empty((0, width)))
                if data[1].shape[1] != width:
                    raise AssertionError(f"unexpected spherical block width: {data[1].shape[1]} != {width}")
                for atom in range(atom_count):
                    start = offset + (2 * angular + 1) * _RADIAL_COUNT
                    result[atom, start - width : start] = _values_for_atom(data, atom, width)
                offset += width
    return result


def _flatten_power_spectrum(tensor_map, atom_count: int) -> np.ndarray:
    blocks = _keys_and_blocks(tensor_map)
    group_width = (_MAX_ANGULAR + 1) * _RADIAL_COUNT * _RADIAL_COUNT
    group_count = len(_SPECIES) * (len(_SPECIES) + 1) // 2
    result = np.zeros((atom_count, len(_SPECIES) * group_count * group_width), dtype=np.float64)
    offset = 0
    for center in _SPECIES:
        for first_index, first in enumerate(_SPECIES):
            for second in _SPECIES[first_index:]:
                data = blocks.get(
                    (center, first, second),
                    (np.empty((0, 2)), np.empty((0, group_width))),
                )
                if data[1].shape[1] != group_width:
                    raise AssertionError(f"unexpected power-spectrum block width: {data[1].shape[1]}")
                for atom in range(atom_count):
                    result[atom, offset : offset + group_width] = _values_for_atom(
                        data, atom, group_width
                    )
                offset += group_width
    return result


def _flatten_radial_spectrum(tensor_map, atom_count: int) -> np.ndarray:
    blocks = _keys_and_blocks(tensor_map)
    result = np.zeros((atom_count, len(_SPECIES) * len(_SPECIES) * _RADIAL_COUNT), dtype=np.float64)
    offset = 0
    for center in _SPECIES:
        for neighbor in _SPECIES:
            data = blocks.get(
                (center, neighbor),
                (np.empty((0, 2)), np.empty((0, _RADIAL_COUNT))),
            )
            if data[1].shape[1] != _RADIAL_COUNT:
                raise AssertionError(f"unexpected radial-spectrum block width: {data[1].shape[1]}")
            for atom in range(atom_count):
                result[atom, offset : offset + _RADIAL_COUNT] = _values_for_atom(
                    data, atom, _RADIAL_COUNT
                )
            offset += _RADIAL_COUNT
    return result


def _flatten_spherical_expansion_by_pair(tensor_map, atomic_numbers: np.ndarray) -> np.ndarray:
    blocks = _keys_and_blocks(tensor_map)
    group_width = sum((2 * angular + 1) * _RADIAL_COUNT for angular in range(_MAX_ANGULAR + 1))
    result = np.zeros((len(atomic_numbers), len(atomic_numbers), group_width), dtype=np.float64)
    for first_atom, first_type in enumerate(atomic_numbers):
        for second_atom, second_type in enumerate(atomic_numbers):
            offset = 0
            for angular in range(_MAX_ANGULAR + 1):
                width = (2 * angular + 1) * _RADIAL_COUNT
                data = blocks.get(
                    (angular, 1, int(first_type), int(second_type)),
                    (np.empty((0, 6)), np.empty((0, width))),
                )
                samples, values = data
                matches = np.flatnonzero(
                    (samples[:, 0] == 0)
                    & (samples[:, 1] == first_atom)
                    & (samples[:, 2] == second_atom)
                )
                if len(matches) == 0 and first_atom == second_atom and angular > 0:
                    offset += width
                    continue
                if len(matches) != 1:
                    raise AssertionError(
                        "expected one Featomic pair row for "
                        f"({first_atom}, {second_atom}, l={angular}), got {len(matches)}"
                    )
                result[first_atom, second_atom, offset : offset + width] = values[matches[0]]
                offset += width
    return result.reshape(len(atomic_numbers) * len(atomic_numbers), group_width)


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
    featomic = pytest.importorskip("featomic")
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
    featomic = pytest.importorskip("featomic")
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
    featomic = pytest.importorskip("featomic")
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
