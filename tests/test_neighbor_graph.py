import numpy as np

from mdescriptor._legacy.core import _build_neighbor_graph
from mdescriptor._legacy.extra import _periodic_neighbors
from tests._public import StructureBatch


def test_native_neighbor_graph_matches_bruteforce_periodic_images():
    cell = np.array(
        [[1.8, 0.0, 0.0], [0.25, 1.7, 0.0], [0.1, 0.2, 1.9]],
        dtype=np.float64,
    )
    positions = np.array([[0.18, 0.29, 0.41], [0.96, 0.85, 1.12]], dtype=np.float64)
    numbers = np.array([8, 14], dtype=np.int32)
    cutoff = 2.3

    offsets, atoms, shifts, displacements, distance2 = _build_neighbor_graph(
        numbers, positions, cell, np.ones(3, dtype=np.int32), cutoff
    )

    expected = {}
    for center in range(len(positions)):
        for atom in range(len(positions)):
            for n0 in range(-4, 5):
                for n1 in range(-4, 5):
                    for n2 in range(-4, 5):
                        shift = np.array([n0, n1, n2], dtype=np.int32)
                        displacement = positions[atom] + shift @ cell - positions[center]
                        squared = float(displacement @ displacement)
                        if squared <= cutoff**2:
                            expected[(center, atom, n0, n1, n2)] = (displacement, squared)

    actual = {}
    for center in range(len(positions)):
        for index in range(int(offsets[center]), int(offsets[center + 1])):
            shift = tuple(int(value) for value in shifts[index])
            actual[(center, int(atoms[index]), *shift)] = (
                displacements[index],
                float(distance2[index]),
            )

    assert set(actual) == set(expected)
    for key, (vector, squared) in actual.items():
        expected_vector, expected_squared = expected[key]
        np.testing.assert_allclose(vector, expected_vector, rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(squared, expected_squared, rtol=0.0, atol=1e-12)

    assert (0, 0, 0, 0, 0) in actual
    assert any(key[:2] == (0, 0) and key[2:] != (0, 0, 0) for key in actual)


def test_python_neighbor_view_filters_only_exact_self():
    cell = np.diag([1.8, 1.7, 1.9])
    system = StructureBatch(
        np.array([8], dtype=np.int32),
        np.array([[0.2, 0.3, 0.4]], dtype=np.float64),
        cell[None, ...],
        np.ones((1, 3), dtype=np.int32),
        np.array([0, 1], dtype=np.int64),
        ("test",),
    )

    neighbors = _periodic_neighbors(system, 0, 2.3)
    assert all(shift != (0, 0, 0) for _, _, _, shift in neighbors[0])
    assert any(shift == (1, 0, 0) for _, _, _, shift in neighbors[0])
