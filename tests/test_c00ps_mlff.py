import numpy as np
import pytest
from ase import Atoms

from tests._public import C00PSMLFF, CancelledError, ComputeControl, StructureBatch


def _system(positions):
    return Atoms(
        "OH2",
        positions=positions,
        cell=np.diag([10.0, 10.0, 10.0]),
        pbc=True,
    )


def test_c00ps_mlff_shape_labels_and_metadata():
    batch = StructureBatch.from_ase([
        _system([[5.0, 5.0, 5.0], [5.8, 5.0, 5.0], [5.0, 5.9, 5.0]])
    ])
    calculator = C00PSMLFF(species=[1, 8], r_cut=3.0, n_radial=3, l_max=2)
    result = calculator.compute(batch)
    # The common q-grid cutoff gives nrb(l) = [3, 2, 2] here.
    expected_features = 2 * 3 + (2 * 3) * (2 * 3 + 1) // 2
    expected_features += 2 * ((2 * 2) * (2 * 2 + 1) // 2)

    assert result.level == "atom"
    assert result.values.shape == (3, expected_features)
    assert result.metadata["descriptor"] == "C00PSMLFF"
    assert result.metadata["details"]["source"] == "C00/PS radial-angular MLFF descriptor core"
    assert len(result.labels) == expected_features
    assert result.labels[0] == "c00ps_mlff:c00,z=1,n=0"
    assert np.isfinite(result.values).all()


def test_c00ps_mlff_is_translation_and_rotation_invariant():
    positions = np.asarray([[5.0, 5.0, 5.0], [5.8, 5.0, 5.0], [5.0, 5.9, 5.0]])
    rotation = np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    center = np.asarray([5.0, 5.0, 5.0])
    shifted = positions + np.asarray([1.25, -0.5, 0.75])
    rotated = (positions - center) @ rotation.T + center
    batch = StructureBatch.from_ase([_system(positions), _system(shifted), _system(rotated)])

    result = C00PSMLFF(species=[1, 8], r_cut=3.0, n_radial=2, l_max=3).compute(batch)
    values = result.values.reshape(3, 3, -1)

    np.testing.assert_allclose(values[0], values[1], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(values[0], values[2], rtol=1e-10, atol=1e-12)


def test_c00ps_mlff_radial_or_angular_only_modes():
    batch = StructureBatch.from_ase([
        _system([[5.0, 5.0, 5.0], [5.8, 5.0, 5.0], [5.0, 5.9, 5.0]])
    ])

    radial = C00PSMLFF(
        species=[1, 8], r_cut=3.0, n_radial=4, l_max=1, include_angular=False
    ).compute(batch)
    angular = C00PSMLFF(
        species=[1, 8], r_cut=3.0, n_radial=4, l_max=1, include_radial=False
    ).compute(batch)

    assert radial.values.shape == (3, 8)
    # nrb(l) = [4, 3] for MRB=4 and l_max=1.
    assert angular.values.shape == (3, 8 * 9 // 2 + 6 * 7 // 2)


def test_c00ps_mlff_uses_vasp_gaussian_radial_basis():
    batch = StructureBatch.from_ase([
        Atoms(
            "H2",
            positions=[[5.0, 5.0, 5.0], [6.0, 5.0, 5.0]],
            cell=np.diag([10.0, 10.0, 10.0]),
            pbc=True,
        )
    ])
    result = C00PSMLFF(
        species=[1],
        r_cut=3.0,
        n_radial=2,
        l_max=0,
        radial_sigma=0.5,
        include_angular=False,
        normalize_radial=False,
    ).compute(batch)

    # VASP 6.6.0's default ML_SION=0.5 gives WION=2 in RAD_FUNC.
    np.testing.assert_allclose(
        result.values[0],
        [0.13044015615446913, 0.086458272123694155],
        rtol=0.0,
        atol=2e-11,
    )


def test_c00ps_mlff_uses_vasp_off_diagonal_power_spectrum_weight():
    batch = StructureBatch.from_ase([
        Atoms(
            "H2",
            positions=[[5.0, 5.0, 5.0], [6.0, 5.0, 5.0]],
            cell=np.diag([10.0, 10.0, 10.0]),
            pbc=True,
        )
    ])
    result = C00PSMLFF(
        species=[1],
        r_cut=3.0,
        n_radial=2,
        l_max=0,
        include_radial=False,
        normalize_angular=False,
        exclude_self_interaction=False,
    ).compute(batch)

    diagonal_left, off_diagonal, diagonal_right = result.values[0]
    assert off_diagonal / np.sqrt(diagonal_left * diagonal_right) == pytest.approx(
        np.sqrt(2.0), rel=1e-12, abs=1e-12
    )


def test_c00ps_mlff_angular_descriptor_excludes_neighbor_self_terms():
    batch = StructureBatch.from_ase([
        Atoms(
            "HO",
            positions=[[5.0, 5.0, 5.0], [6.0, 5.0, 5.0]],
            cell=np.diag([10.0, 10.0, 10.0]),
            pbc=True,
        )
    ])
    result = C00PSMLFF(
        species=[1, 8],
        r_cut=3.0,
        n_radial=2,
        l_max=2,
        include_radial=False,
    ).compute(batch)

    np.testing.assert_allclose(result.values, 0.0, rtol=0.0, atol=1e-12)


def test_c00ps_mlff_higher_angular_channels_use_nonzero_bessel_basis():
    batch = StructureBatch.from_ase([
        Atoms(
            "H3",
            positions=[[5.0, 5.0, 5.0], [6.0, 5.0, 5.0], [5.8, 5.6, 5.0]],
            cell=np.diag([10.0, 10.0, 10.0]),
            pbc=True,
        )
    ])
    result = C00PSMLFF(
        species=[1],
        r_cut=3.0,
        n_radial=1,
        l_max=2,
        include_radial=False,
    ).compute(batch)

    assert abs(result.values[0, 1]) > 1e-8


def test_c00ps_mlff_honors_pre_cancelled_control():
    control = ComputeControl()
    control.reset(1)
    control.cancel()
    with pytest.raises(CancelledError):
        C00PSMLFF(species=[1, 8], r_cut=3.0, n_radial=1, l_max=1).compute(
            StructureBatch.from_ase([_system([
                [5.0, 5.0, 5.0], [5.8, 5.0, 5.0], [5.0, 5.9, 5.0]
            ])]),
            control=control,
        )
