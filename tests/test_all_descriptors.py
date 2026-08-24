import numpy as np
import pytest
from ase import Atoms

from mdescriptor import (
    CancelledError,
    ComputeControl,
    DESCRIPTOR_CATALOG,
    MBTRCalculator,
    MODEL_DESCRIPTOR_CATALOG,
    StructureBatch,
)


def _batch():
    atoms = Atoms(
        "NaCl2Si",
        positions=[[0.1, 0.2, 0.3], [1.3, 1.1, 1.0], [2.1, 0.4, 2.3], [3.4, 2.2, 1.8]],
        cell=[[6.7, 0.2, 0.1], [0.3, 6.9, 0.4], [0.1, 0.2, 6.8]],
        pbc=True,
    )
    return StructureBatch.from_ase([atoms])


def _calculator(name, cls):
    if name == "ACSF":
        return cls(species=[8, 11, 14, 17], r_cut=3.5)
    if name in {"CoulombMatrix", "SineMatrix", "EwaldSumMatrix"}:
        return cls(n_atoms_max=4)
    if name == "AtomicComposition":
        return cls(species=[8, 11, 14, 17])
    if name == "SortedDistances":
        return cls(species=[8, 11, 14, 17], cutoff=3.5, max_neighbors=4)
    if name == "NeighborList":
        return cls(cutoff=3.5)
    if name == "SOAPTurbo":
        return cls(
            species=[8, 11, 14, 17],
            alpha_max=[2, 2, 2, 2],
            l_max=2,
            rcut_hard=3.5,
            rcut_soft=3.0,
            atom_sigma_r=0.5,
            atom_sigma_t=0.5,
        )
    return cls()


def test_every_catalog_descriptor_is_native_and_finite():
    assert len(DESCRIPTOR_CATALOG) == 24
    assert len(set(DESCRIPTOR_CATALOG.values())) == len(DESCRIPTOR_CATALOG)
    batch = _batch()
    for name, cls in DESCRIPTOR_CATALOG.items():
        result = _calculator(name, cls).compute(batch)
        assert result.metadata["backend"] == "mdescriptor-cpp", name
        assert result.values.ndim == 2 and result.values.shape[0] > 0, name
        assert np.isfinite(result.values).all(), name


def test_model_descriptor_catalog_is_separate():
    assert tuple(MODEL_DESCRIPTOR_CATALOG) == ("NEP", "DPA4", "DPA4C")
    assert len(set(MODEL_DESCRIPTOR_CATALOG.values())) == len(MODEL_DESCRIPTOR_CATALOG)
    assert set(DESCRIPTOR_CATALOG).isdisjoint(MODEL_DESCRIPTOR_CATALOG)


def test_soap_turbo_has_rotation_invariant_core_and_central_filter():
    from mdescriptor import SoapTurboCalculator

    positions = np.asarray([[3.2, 3.3, 3.5], [4.4, 3.8, 3.5], [3.0, 4.1, 4.2]])
    rotation = np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    center = np.asarray([4.0, 4.0, 4.0])
    rotated = (positions - center) @ rotation.T + center
    systems = [
        Atoms("OHH", positions=positions, cell=np.diag([10.0] * 3), pbc=True),
        Atoms("OHH", positions=rotated, cell=np.diag([10.0] * 3), pbc=True),
    ]
    parameters = {
        "species": [1, 8],
        "alpha_max": [2, 2],
        "l_max": 2,
        "rcut_hard": 3.0,
        "rcut_soft": 2.5,
        "atom_sigma_r": [0.4, 0.5],
        "atom_sigma_t": [0.4, 0.5],
        "basis": "poly3gauss",
    }
    result = SoapTurboCalculator(**parameters).compute(StructureBatch.from_ase(systems))
    channels = sum(parameters["alpha_max"])
    assert result.values.shape == (6, channels * (channels + 1) // 2 * (parameters["l_max"] + 1))
    np.testing.assert_allclose(result.values[:3], result.values[3:], rtol=1e-10, atol=1e-12)

    filtered = SoapTurboCalculator(**parameters, central_species=[8]).compute(
        StructureBatch.from_ase([systems[0]])
    )
    np.testing.assert_array_equal(filtered.values[1:], 0.0)


@pytest.mark.parametrize(
    ("compression", "feature_count"),
    {
        "trivial": 21,
        "0_0": 3,
        "0_1": 6,
        "0_2": 9,
        "1_0": 6,
        "1_1": 12,
        "1_2": 24,
        "2_0": 9,
        "2_1": 24,
        "2_2": 30,
    }.items(),
)
def test_soap_turbo_upstream_compression_modes(compression, feature_count):
    from mdescriptor import SoapTurboCalculator

    system = Atoms(
        "OH",
        positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.8]],
        cell=np.diag([8.0, 8.0, 8.0]),
        pbc=True,
    )
    result = SoapTurboCalculator(
        species=[1, 8], alpha_max=[2, 2], l_max=2,
        rcut_hard=3.0, rcut_soft=2.5, compression=compression,
    ).compute(StructureBatch.from_ase([system]))
    assert result.values.shape == (2, feature_count)
    assert np.isfinite(result.values).all()


def test_matrix_kernel_has_expected_coulomb_diagonal_and_shapes():
    from mdescriptor import CoulombMatrixCalculator, EwaldSumMatrixCalculator, SineMatrixCalculator

    system = Atoms("NaCl", positions=[[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]], cell=np.diag([8.0, 8.0, 8.0]), pbc=True)
    batch = StructureBatch.from_ase([system])
    values = CoulombMatrixCalculator(3, permutation="none").compute(batch).values[0].reshape(3, 3)
    assert values[0, 0] == pytest.approx(0.5 * 11**2.4)
    assert values[1, 1] == pytest.approx(0.5 * 17**2.4)
    assert values[0, 1] == pytest.approx(11 * 17 / 1.2)
    assert SineMatrixCalculator(3).compute(batch).values.shape == (1, 9)
    assert EwaldSumMatrixCalculator(3).compute(batch).values.shape == (1, 9)


def test_ewald_matches_dscribe_at_real_cutoff_boundary():
    dscribe = pytest.importorskip("dscribe")
    from dscribe.descriptors import EwaldSumMatrix
    from mdescriptor import EwaldSumMatrixCalculator

    system = Atoms(
        ["O", "H", "H"] * 6,
        positions=[
            [3.0, 3.0, 3.0], [3.76, 3.58, 3.0], [2.24, 3.58, 3.0],
            [9.0, 3.0, 3.0], [9.76, 3.58, 3.0], [8.24, 3.58, 3.0],
            [3.0, 9.0, 9.0], [3.76, 9.58, 9.0], [2.24, 9.58, 9.0],
            [9.0, 9.0, 9.0], [9.76, 9.58, 9.0], [8.24, 9.58, 9.0],
            [3.0, 3.0, 9.0], [3.76, 3.58, 9.0], [2.24, 3.58, 9.0],
            [9.0, 9.0, 3.0], [9.76, 9.58, 3.0], [8.24, 9.58, 3.0],
        ],
        cell=np.diag([12.0, 12.0, 12.0]),
        pbc=True,
    )
    parameters = {"accuracy": 1e-5, "w": 1.0, "r_cut": 6.0, "g_cut": 3.0, "a": 0.3}
    reference = EwaldSumMatrix(18, permutation="none").create(system, **parameters)
    actual = EwaldSumMatrixCalculator(18, permutation="none", **parameters).compute(
        StructureBatch.from_ase([system])
    ).values[0]
    np.testing.assert_allclose(actual, reference, rtol=1e-10, atol=1e-12)


def test_matrix_eigenspectrum_is_native_and_padded():
    from mdescriptor import CoulombMatrixCalculator

    batch = _batch()
    result = CoulombMatrixCalculator(6, permutation="eigenspectrum").compute(batch)
    assert result.values.shape == (1, 6)
    assert np.isfinite(result.values).all()


def test_sine_sorted_l2_matches_dscribe_numpy_sort_contract():
    from mdescriptor import SineMatrixCalculator

    batch = _batch()
    raw = SineMatrixCalculator(6, permutation="none").compute(batch).values[0].reshape(6, 6)
    sorted_values = SineMatrixCalculator(6, permutation="sorted_l2").compute(batch).values[0]
    count = len(batch.numbers)
    matrix = raw[:count, :count]
    order = np.argsort(-np.linalg.norm(matrix, axis=1), kind="stable")
    expected = np.zeros((6, 6))
    expected[:count, :count] = matrix[order][:, order]
    np.testing.assert_array_equal(sorted_values.reshape(6, 6), expected)


def test_coulomb_eigenspectrum_keeps_batch_structure_stride():
    from ase import Atoms
    from mdescriptor import CoulombMatrixCalculator

    systems = [
        Atoms("NaCl", positions=[[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]], cell=np.diag([8.0] * 3), pbc=True),
        Atoms("Si3", positions=[[0.0, 0.0, 0.0], [1.4, 0.0, 0.0], [0.0, 1.4, 0.0]], cell=np.diag([8.0] * 3), pbc=True),
    ]
    batch = StructureBatch.from_ase(systems)
    raw = CoulombMatrixCalculator(4, permutation="none").compute(batch).values
    spectrum = CoulombMatrixCalculator(4, permutation="eigenspectrum").compute(batch).values
    for index, count in enumerate((2, 3)):
        matrix = raw[index].reshape(4, 4)[:count, :count]
        eigenvalues = np.linalg.eigvalsh(matrix)
        expected = eigenvalues[np.argsort(np.abs(eigenvalues))[::-1]]
        np.testing.assert_allclose(spectrum[index, :count], expected, rtol=1e-10, atol=1e-10)
        np.testing.assert_array_equal(spectrum[index, count:], 0.0)


def test_coulomb_sorted_l2_matches_dscribe_tie_order():
    from ase import Atoms
    from mdescriptor import CoulombMatrixCalculator

    positions = [
        [9.0, 12.0, 12.0], [10.35, 12.35, 12.0], [11.7, 12.0, 12.0],
        [13.05, 12.35, 12.0], [14.4, 12.0, 12.0], [15.75, 12.35, 12.0],
        [17.1, 12.0, 12.0], [18.45, 12.35, 12.0], [19.8, 12.0, 12.0],
        [21.15, 12.35, 12.0], [22.5, 12.0, 12.0], [23.85, 12.35, 12.0],
    ]
    batch = StructureBatch.from_ase([Atoms("C12", positions=positions, cell=np.diag([24.0] * 3), pbc=True)])
    raw = CoulombMatrixCalculator(12, permutation="none").compute(batch).values[0].reshape(12, 12)
    sorted_matrix = CoulombMatrixCalculator(12, permutation="sorted_l2").compute(batch).values[0].reshape(12, 12)
    order = [6, 5, 4, 7, 8, 3, 9, 2, 1, 10, 0, 11]
    np.testing.assert_array_equal(sorted_matrix, raw[order][:, order])


def test_mbtr_family_is_native():
    from mdescriptor import EwaldSumMatrixCalculator, LMBTRCalculator, ValleOganovCalculator

    batch = _batch()
    config = {
        "species": [11, 14, 17],
        "geometry": {"function": "angle"},
        "grid": {"min": 0.0, "max": 180.0, "n": 30, "sigma": 0.1},
        "weighting": {"function": "exp", "scale": 0.3, "threshold": 1e-3},
    }
    calculators = [
        EwaldSumMatrixCalculator(4),
        MBTRCalculator(**config),
        LMBTRCalculator(**config),
        ValleOganovCalculator(species=[11, 14, 17], function="angle", n=30, sigma=0.1, r_cut=4.0),
    ]
    for calculator in calculators:
        result = calculator.compute(batch)
        assert result.metadata["backend"] == "mdescriptor-cpp"
        assert np.isfinite(result.values).all()


def test_valle_oganov_near_linear_angles_match_dscribe():
    pytest.importorskip("dscribe")
    from pathlib import Path

    from ase.io import read
    from dscribe.descriptors import ValleOganov
    from mdescriptor import ValleOganovCalculator

    dataset = Path(__file__).parents[1] / "benchmarks" / "soap_diverse_dataset_300.xyz"
    systems = [read(dataset, index=index) for index in (8, 11)]
    species = [1, 6, 8, 14, 16, 17]
    parameters = {"function": "angle", "n": 90, "sigma": 2.0, "r_cut": 6.0}
    reference = ValleOganov(species, **parameters).create(systems, n_jobs=1)
    actual = ValleOganovCalculator(species=species, **parameters).compute(
        StructureBatch.from_ase(systems)
    ).values
    np.testing.assert_allclose(actual, reference, rtol=1e-9, atol=1e-7)


def test_lmbtr_k3_matches_dscribe_channel_layout():
    pytest.importorskip("dscribe")
    from dscribe.descriptors import LMBTR
    from mdescriptor import LMBTRCalculator

    system = Atoms(
        "OHH",
        positions=[[12.0, 12.0, 12.0], [12.76, 12.58, 12.0], [11.24, 12.58, 12.0]],
        cell=np.diag([24.0, 24.0, 24.0]),
        pbc=True,
    )
    batch = StructureBatch.from_ase([system])
    species = [1, 8]
    for geometry, grid in (
        ("angle", {"min": 0.0, "max": 180.0, "n": 30, "sigma": 1.5}),
        ("cosine", {"min": -1.0, "max": 1.0, "n": 30, "sigma": 0.03}),
    ):
        parameters = {
            "species": species,
            "periodic": True,
            "geometry": {"function": geometry},
            "grid": grid,
            "weighting": {"function": "exp", "scale": 0.5, "threshold": 1e-3},
        }
        reference = np.asarray(LMBTR(**parameters).create(system, n_jobs=1))
        actual = LMBTRCalculator(**parameters).compute(batch).values
        np.testing.assert_allclose(actual, reference, rtol=1e-9, atol=1e-10)


def test_featomic_family_uses_native_backend():
    from mdescriptor import (
        AtomicCompositionCalculator, LodeSphericalExpansionCalculator, NeighborListCalculator,
        SoapPowerSpectrumCalculator, SoapRadialSpectrumCalculator, SortedDistancesCalculator,
        SphericalExpansionByPairCalculator, SphericalExpansionCalculator,
    )

    batch = _batch()
    calculators = {
        "AtomicComposition": AtomicCompositionCalculator([8, 11, 14, 17]),
        "NeighborList": NeighborListCalculator(cutoff=3.5),
        "SortedDistances": SortedDistancesCalculator([8, 11, 14, 17], cutoff=3.5, max_neighbors=4),
        "SphericalExpansion": SphericalExpansionCalculator([8, 11, 14, 17], cutoff=3.5, density_width=0.6, max_radial=2, max_angular=2),
        "SphericalExpansionByPair": SphericalExpansionByPairCalculator([8, 11, 14, 17], cutoff=3.5, density_width=0.6, max_radial=2, max_angular=2),
        "SoapRadialSpectrum": SoapRadialSpectrumCalculator([8, 11, 14, 17], cutoff=3.5, density_width=0.6, max_radial=2, max_angular=2),
        "SoapPowerSpectrum": SoapPowerSpectrumCalculator([8, 11, 14, 17], cutoff=3.5, density_width=0.6, max_radial=2, max_angular=2),
        "LodeSphericalExpansion": LodeSphericalExpansionCalculator([8, 11, 14, 17], cutoff=3.5, density_width=0.5, max_radial=2, max_angular=2),
    }
    for name, calculator in calculators.items():
        result = calculator.compute(batch)
        assert result.metadata["backend"] == "mdescriptor-cpp", name
        assert np.isfinite(result.values).all(), name


def test_pyxtal_family_repeats_identically_in_native_kernel():
    from mdescriptor import EadCalculator, LbispectrumCalculator, SnapCalculator, So3Calculator, So4Calculator

    batch = _batch()
    calculators = [
        EadCalculator({"L": 2, "eta": [0.05, 0.1], "Rs": [0.0, 0.5]}, Rc=3.5),
        So3Calculator(nmax=2, lmax=2, rcut=3.5, alpha=2.0),
        So4Calculator(lmax=2, rcut=3.5, normalize_U=True),
        SnapCalculator(lmax=2, rcut=3.5, weights={14: 1.0}),
        LbispectrumCalculator(twojmax=3, diagonal=3, rcut=3.5),
        LbispectrumCalculator(
            twojmax=3,
            diagonal=1,
            element_profile={11: {"r": 1.8, "w": 0.9}, 14: {"r": 2.0, "w": 1.1}, 17: {"r": 1.7, "w": 1.2}},
            rcutfac=1.1,
            rmin0=0.1,
            rcut=3.5,
        ),
    ]
    for calculator in calculators:
        first = calculator.compute(batch)
        second = calculator.compute(batch)
        assert first.metadata["backend"] == "mdescriptor-cpp"
        np.testing.assert_array_equal(first.values, second.values)


def test_native_extra_descriptors_honor_cancellation():
    control = ComputeControl()
    control.reset(1)
    control.cancel()
    with pytest.raises(CancelledError):
        MBTRCalculator(species=[8, 11, 14, 17]).compute(_batch(), control)
