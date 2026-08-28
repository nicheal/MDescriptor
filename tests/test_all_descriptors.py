import numpy as np
import pytest
from ase import Atoms

from tests._public import (
    MBTR,
    AssetPolicy,
    CancelledError,
    ComputeControl,
    StructureBatch,
    builtin_registry,
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
    if name == "ACE":
        return cls(species=[8, 11, 14, 17], N=2, maxdeg=4, rcut=3.5)
    if name == "SOAP":
        return cls(species=[8, 11, 14, 17], r_cut=3.5, n_max=2, l_max=2)
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
    if name in {"MBTR", "LMBTR"}:
        return cls(
            species=[8, 11, 14, 17],
            geometry={"function": "distance"},
            grid={"min": 0.0, "max": 4.0, "n": 20, "sigma": 0.1},
            weighting={"function": "exp", "scale": 0.3, "threshold": 1e-3},
        )
    if name == "ValleOganov":
        return cls(species=[8, 11, 14, 17], function="distance", n=20, sigma=0.1, r_cut=3.5)
    if name == "MTP":
        return cls(species=[8, 11, 14, 17], min_dist=0.1, max_dist=3.5, radial_basis_size=2, max_rank=2)
    if name == "C00PSMLFF":
        return cls(species=[8, 11, 14, 17], r_cut=3.5, n_radial=2, l_max=2)
    if name in {
        "SphericalExpansion", "SphericalExpansionByPair", "SoapRadialSpectrum",
        "SoapPowerSpectrum", "LodeSphericalExpansion",
    }:
        return cls(
            species=[8, 11, 14, 17], cutoff=3.5, density_width=0.6,
            max_radial=2, max_angular=2,
        )
    if name == "EAD":
        return cls(parameters={"L": 2, "eta": [0.05, 0.1], "Rs": [0.0, 0.5]}, Rc=3.5)
    return cls()


def _registered_descriptors(policy):
    return tuple(
        (spec.name, spec.load_class())
        for spec in builtin_registry
        if spec.asset_policy is policy
    )


def test_every_standalone_descriptor_is_native_and_finite():
    descriptors = _registered_descriptors(AssetPolicy.NONE) + _registered_descriptors(AssetPolicy.OPTIONAL)
    assert len(descriptors) == 25
    assert len({descriptor_class for _, descriptor_class in descriptors}) == len(descriptors)
    batch = _batch()
    for name, cls in descriptors:
        result = _calculator(name, cls).compute(batch)
        assert result.metadata["backend"] == "mdescriptor-cpp", name
        assert result.values.ndim == 2 and result.values.shape[0] > 0, name
        assert np.isfinite(result.values).all(), name


def test_model_backed_descriptors_are_separate():
    descriptors = _registered_descriptors(AssetPolicy.REQUIRED)
    assert tuple(name for name, _ in descriptors) == ("NEP", "DPA4", "DPA4C")
    assert len({descriptor_class for _, descriptor_class in descriptors}) == len(descriptors)
    standalone_names = {
        name for name, _ in _registered_descriptors(AssetPolicy.NONE) + _registered_descriptors(AssetPolicy.OPTIONAL)
    }
    assert standalone_names.isdisjoint(name for name, _ in descriptors)


def test_soap_turbo_has_rotation_invariant_core_and_central_filter():
    from tests._public import SOAPTurbo

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
    result = SOAPTurbo(**parameters).compute(StructureBatch.from_ase(systems))
    channels = sum(parameters["alpha_max"])
    assert result.values.shape == (6, channels * (channels + 1) // 2 * (parameters["l_max"] + 1))
    np.testing.assert_allclose(result.values[:3], result.values[3:], rtol=1e-10, atol=1e-12)

    filtered = SOAPTurbo(**parameters, central_species=[8]).compute(
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
    from tests._public import SOAPTurbo

    system = Atoms(
        "OH",
        positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.8]],
        cell=np.diag([8.0, 8.0, 8.0]),
        pbc=True,
    )
    result = SOAPTurbo(
        species=[1, 8], alpha_max=[2, 2], l_max=2,
        rcut_hard=3.0, rcut_soft=2.5, compression=compression,
    ).compute(StructureBatch.from_ase([system]))
    assert result.values.shape == (2, feature_count)
    assert np.isfinite(result.values).all()


def test_matrix_kernel_has_expected_coulomb_diagonal_and_shapes():
    from tests._public import CoulombMatrix, EwaldSumMatrix, SineMatrix

    system = Atoms("NaCl", positions=[[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]], cell=np.diag([8.0, 8.0, 8.0]), pbc=True)
    batch = StructureBatch.from_ase([system])
    values = CoulombMatrix(n_atoms_max=3, permutation="none").compute(batch).values[0].reshape(3, 3)
    assert values[0, 0] == pytest.approx(0.5 * 11**2.4)
    assert values[1, 1] == pytest.approx(0.5 * 17**2.4)
    assert values[0, 1] == pytest.approx(11 * 17 / 1.2)
    assert SineMatrix(n_atoms_max=3).compute(batch).values.shape == (1, 9)
    assert EwaldSumMatrix(n_atoms_max=3).compute(batch).values.shape == (1, 9)


@pytest.mark.reference
def test_ewald_matches_reference_at_real_cutoff_boundary():
    from dscribe.descriptors import EwaldSumMatrix as DscribeEwaldSumMatrix

    from tests._public import EwaldSumMatrix

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
    reference = DscribeEwaldSumMatrix(n_atoms_max=18, permutation="none").create(system, **parameters)
    actual = EwaldSumMatrix(n_atoms_max=18, permutation="none", **parameters).compute(
        StructureBatch.from_ase([system])
    ).values[0]
    np.testing.assert_allclose(actual, reference, rtol=1e-10, atol=1e-12)


def test_matrix_eigenspectrum_is_native_and_padded():
    from tests._public import CoulombMatrix

    batch = _batch()
    result = CoulombMatrix(n_atoms_max=6, permutation="eigenspectrum").compute(batch)
    assert result.values.shape == (1, 6)
    assert np.isfinite(result.values).all()


def test_sine_sorted_l2_matches_reference_order_for_non_ties():
    from tests._public import SineMatrix

    batch = _batch()
    raw = SineMatrix(n_atoms_max=6, permutation="none").compute(batch).values[0].reshape(6, 6)
    sorted_values = SineMatrix(n_atoms_max=6, permutation="sorted_l2").compute(batch).values[0]
    count = len(batch.numbers)
    matrix = raw[:count, :count]
    # This fixture has distinct row norms; the native tie policy is therefore
    # equivalent to the reference descending-norm order here.
    order = np.argsort(-np.linalg.norm(matrix, axis=1), kind="stable")
    expected = np.zeros((6, 6))
    expected[:count, :count] = matrix[order][:, order]
    np.testing.assert_array_equal(sorted_values.reshape(6, 6), expected)


def test_coulomb_eigenspectrum_keeps_batch_structure_stride():
    from ase import Atoms

    from tests._public import CoulombMatrix

    systems = [
        Atoms("NaCl", positions=[[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]], cell=np.diag([8.0] * 3), pbc=True),
        Atoms("Si3", positions=[[0.0, 0.0, 0.0], [1.4, 0.0, 0.0], [0.0, 1.4, 0.0]], cell=np.diag([8.0] * 3), pbc=True),
    ]
    batch = StructureBatch.from_ase(systems)
    raw = CoulombMatrix(n_atoms_max=4, permutation="none").compute(batch).values
    spectrum = CoulombMatrix(n_atoms_max=4, permutation="eigenspectrum").compute(batch).values
    for index, count in enumerate((2, 3)):
        matrix = raw[index].reshape(4, 4)[:count, :count]
        eigenvalues = np.linalg.eigvalsh(matrix)
        expected = eigenvalues[np.argsort(np.abs(eigenvalues))[::-1]]
        np.testing.assert_allclose(spectrum[index, :count], expected, rtol=1e-10, atol=1e-10)
        np.testing.assert_array_equal(spectrum[index, count:], 0.0)


def test_coulomb_sorted_l2_uses_stable_input_tie_order():
    from ase import Atoms

    from tests._public import CoulombMatrix

    positions = [
        [9.0, 12.0, 12.0], [10.35, 12.35, 12.0], [11.7, 12.0, 12.0],
        [13.05, 12.35, 12.0], [14.4, 12.0, 12.0], [15.75, 12.35, 12.0],
        [17.1, 12.0, 12.0], [18.45, 12.35, 12.0], [19.8, 12.0, 12.0],
        [21.15, 12.35, 12.0], [22.5, 12.0, 12.0], [23.85, 12.35, 12.0],
    ]
    batch = StructureBatch.from_ase([Atoms("C12", positions=positions, cell=np.diag([24.0] * 3), pbc=True)])
    raw = CoulombMatrix(n_atoms_max=12, permutation="none").compute(batch).values[0].reshape(12, 12)
    sorted_matrix = CoulombMatrix(n_atoms_max=12, permutation="sorted_l2").compute(batch).values[0].reshape(12, 12)
    # Equal row norms use the original atom index as the deterministic
    # secondary key.  The input order therefore resolves each mirrored pair.
    order = [5, 6, 4, 7, 3, 8, 2, 9, 1, 10, 0, 11]
    np.testing.assert_array_equal(sorted_matrix, raw[order][:, order])


def test_mbtr_family_is_native():
    from tests._public import LMBTR, EwaldSumMatrix, ValleOganov

    batch = _batch()
    config = {
        "species": [11, 14, 17],
        "geometry": {"function": "angle"},
        "grid": {"min": 0.0, "max": 180.0, "n": 30, "sigma": 0.1},
        "weighting": {"function": "exp", "scale": 0.3, "threshold": 1e-3},
    }
    calculators = [
        EwaldSumMatrix(n_atoms_max=4),
        MBTR(**config),
        LMBTR(**config),
        ValleOganov(species=[11, 14, 17], function="angle", n=30, sigma=0.1, r_cut=4.0),
    ]
    for calculator in calculators:
        result = calculator.compute(batch)
        assert result.metadata["backend"] == "mdescriptor-cpp"
        assert np.isfinite(result.values).all()


@pytest.mark.reference
def test_valle_oganov_near_linear_angles_match_reference():
    from dscribe.descriptors import ValleOganov as DscribeValleOganov

    from tests._public import ValleOganov

    systems = [
        Atoms(
            "OHH",
            positions=[[4.0, 4.0, 4.0], [4.96, 4.0, 4.0], [3.76, 4.93, 4.0]],
            cell=np.diag([20.0, 20.0, 20.0]),
            pbc=True,
        ),
        Atoms(
            "OHH",
            positions=[[10.0, 10.0, 10.0], [10.96, 10.0, 10.0], [9.76, 10.93, 10.0]],
            cell=np.diag([20.0, 20.0, 20.0]),
            pbc=True,
        ),
    ]
    species = [1, 6, 8, 14, 16, 17]
    parameters = {"function": "angle", "n": 90, "sigma": 2.0, "r_cut": 6.0}
    reference = DscribeValleOganov(species=species, **parameters).create(systems, n_jobs=1)
    actual = ValleOganov(species=species, **parameters).compute(
        StructureBatch.from_ase(systems)
    ).values
    np.testing.assert_allclose(actual, reference, rtol=1e-9, atol=1e-7)


@pytest.mark.reference
def test_lmbtr_k3_matches_reference_channel_layout():
    from dscribe.descriptors import LMBTR as DscribeLMBTR

    from tests._public import LMBTR

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
        reference = np.asarray(DscribeLMBTR(**parameters).create(system, n_jobs=1))
        actual = LMBTR(**parameters).compute(batch).values
        np.testing.assert_allclose(actual, reference, rtol=1e-9, atol=1e-10)


def test_local_descriptor_family_uses_native_backend():
    from tests._public import (
        AtomicComposition,
        LodeSphericalExpansion,
        NeighborList,
        SoapPowerSpectrum,
        SoapRadialSpectrum,
        SortedDistances,
        SphericalExpansion,
        SphericalExpansionByPair,
    )

    batch = _batch()
    calculators = {
        "AtomicComposition": AtomicComposition(species=[8, 11, 14, 17]),
        "NeighborList": NeighborList(cutoff=3.5),
        "SortedDistances": SortedDistances(species=[8, 11, 14, 17], cutoff=3.5, max_neighbors=4),
        "SphericalExpansion": SphericalExpansion(species=[8, 11, 14, 17], cutoff=3.5, density_width=0.6, max_radial=2, max_angular=2),
        "SphericalExpansionByPair": SphericalExpansionByPair(species=[8, 11, 14, 17], cutoff=3.5, density_width=0.6, max_radial=2, max_angular=2),
        "SoapRadialSpectrum": SoapRadialSpectrum(species=[8, 11, 14, 17], cutoff=3.5, density_width=0.6, max_radial=2, max_angular=2),
        "SoapPowerSpectrum": SoapPowerSpectrum(species=[8, 11, 14, 17], cutoff=3.5, density_width=0.6, max_radial=2, max_angular=2),
        "LodeSphericalExpansion": LodeSphericalExpansion(species=[8, 11, 14, 17], cutoff=3.5, density_width=0.5, max_radial=2, max_angular=2),
    }
    for name, calculator in calculators.items():
        result = calculator.compute(batch)
        assert result.metadata["backend"] == "mdescriptor-cpp", name
        assert np.isfinite(result.values).all(), name


def test_rotational_descriptor_family_repeats_identically_in_native_kernel():
    from tests._public import EAD, SNAP, SO3, SO4, LBispectrum

    batch = _batch()
    calculators = [
        EAD(parameters={"L": 2, "eta": [0.05, 0.1], "Rs": [0.0, 0.5]}, Rc=3.5),
        SO3(nmax=2, lmax=2, rcut=3.5, alpha=2.0),
        SO4(lmax=2, rcut=3.5, normalize_U=True),
        SNAP(lmax=2, rcut=3.5, weights={14: 1.0}),
        LBispectrum(twojmax=3, diagonal=3, rcut=3.5),
        LBispectrum(
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
        MBTR(species=[8, 11, 14, 17]).compute(_batch(), control=control)
