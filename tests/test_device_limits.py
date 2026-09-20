"""Registry-declared device capacities and their public validation paths."""

from __future__ import annotations

import numpy as np
import pytest

from mdescriptor import (
    DescriptorConfigError,
    DescriptorInputError,
    ExecutionOptions,
    StructureBatch,
    describe_descriptor,
)
from mdescriptor.descriptors import (
    C00PSMLFF,
    LMBTR,
    MBTR,
    CoulombMatrix,
    EwaldSumMatrix,
    SineMatrix,
    ValleOganov,
)


def _batch(atom_count: int = 1) -> StructureBatch:
    return StructureBatch(
        np.ones(atom_count, dtype=np.int32),
        np.zeros((atom_count, 3), dtype=np.float64),
        np.eye(3, dtype=np.float64)[None, :, :] * 20.0,
        np.ones((1, 3), dtype=np.int32),
        np.array([0, atom_count], dtype=np.int64),
        ("one",),
    )


def _cuda() -> ExecutionOptions:
    return ExecutionOptions(device="cuda")


@pytest.mark.parametrize(
    ("descriptor", "options", "path"),
    (
        (CoulombMatrix, {"n_atoms_max": 257}, ["parameters", "n_atoms_max"]),
        (SineMatrix, {"n_atoms_max": 257}, ["parameters", "n_atoms_max"]),
        (EwaldSumMatrix, {"n_atoms_max": 257}, ["parameters", "n_atoms_max"]),
        (C00PSMLFF, {"species": [1], "l_max": 21}, ["parameters", "l_max"]),
        (
            MBTR,
            {"species": list(range(1, 66)), "normalization": "valle_oganov"},
            ["parameters", "species"],
        ),
        (ValleOganov, {"species": list(range(1, 66))}, ["parameters", "species"]),
    ),
)
def test_cuda_device_limits_fail_at_construction(descriptor, options, path):
    with pytest.raises(DescriptorConfigError) as caught:
        descriptor(execution=_cuda(), **options)

    assert list(caught.value.path or ()) == path
    assert caught.value.code == "invalid_parameter"
    assert caught.value.details["device"] == "cuda"


def test_cuda_matrix_boundary_and_dynamic_width_contract():
    descriptor = CoulombMatrix(n_atoms_max=256, execution=_cuda())
    descriptor.close()

    dynamic = CoulombMatrix(execution=_cuda())
    try:
        with pytest.raises(DescriptorInputError) as caught:
            dynamic.compute(_batch(257))
    finally:
        dynamic.close()

    assert caught.value.code == "unsupported_input"
    assert list(caught.value.path or ()) == ["input", "n_atoms_max"]
    assert caught.value.details == {
        "device": "cuda",
        "provided": 257,
        "maximum": 256,
    }


def test_conditional_cuda_species_limit_does_not_restrict_other_paths():
    descriptors = (
        MBTR(
            species=list(range(1, 66)),
            geometry={"function": "atomic_number"},
            normalization="valle_oganov",
            execution=_cuda(),
        ),
        MBTR(
            species=list(range(1, 66)),
            normalization="none",
            execution=_cuda(),
        ),
        ValleOganov(
            species=list(range(1, 66)),
            normalization="none",
            execution=_cuda(),
        ),
        ValleOganov(
            species=list(range(1, 66)),
            geometry={"function": "atomic_number"},
            n=2,
            r_cut=2.0,
            execution=_cuda(),
        ),
        LMBTR(
            species=list(range(1, 66)),
            normalization="valle_oganov",
            execution=_cuda(),
        ),
    )
    for descriptor in descriptors:
        descriptor.close()


def test_cuda_limits_are_inclusive_at_the_boundary():
    descriptors = (
        C00PSMLFF(species=[1], l_max=20, execution=_cuda()),
        MBTR(
            species=list(range(1, 65)),
            normalization="valle_oganov",
            grid={"min": 0.0, "max": 2.0, "n": 2, "sigma": 0.1},
            execution=_cuda(),
        ),
        ValleOganov(
            species=list(range(1, 65)),
            n=2,
            r_cut=2.0,
            execution=_cuda(),
        ),
    )
    for descriptor in descriptors:
        descriptor.close()


def test_cpu_keeps_wider_matrix_angular_and_species_ranges():
    cases = (
        (CoulombMatrix(n_atoms_max=257), 257 * 257),
        (C00PSMLFF(species=[1], l_max=21), None),
        (
            MBTR(
                species=list(range(1, 66)),
                grid={"min": 0.0, "max": 2.0, "n": 2, "sigma": 0.1},
                normalization="valle_oganov",
            ),
            None,
        ),
        (ValleOganov(species=list(range(1, 66)), n=2, r_cut=2.0), None),
    )
    try:
        for descriptor, expected_width in cases:
            result = descriptor.compute(_batch())
            assert result.values.shape[0] == 1
            if expected_width is not None:
                assert result.values.shape[1] == expected_width
            else:
                assert result.values.shape[1] > 0
    finally:
        for descriptor, _ in cases:
            descriptor.close()


def test_valle_oganov_atomic_geometry_keeps_the_wider_species_range():
    parameters = {
        "species": list(range(1, 66)),
        "geometry": {"function": "atomic_number"},
        "n": 2,
        "r_cut": 2.0,
    }
    descriptor = ValleOganov(**parameters)
    try:
        result = descriptor.compute(_batch())
        assert result.values.shape == (1, 130)
        assert np.isfinite(result.values).all()
    finally:
        descriptor.close()


def test_device_limits_are_static_gui_metadata():
    matrix_limits = describe_descriptor("CoulombMatrix")["execution"]["device_limits"]
    c00ps_limits = describe_descriptor("C00PSMLFF")["execution"]["device_limits"]
    mbtr_limits = describe_descriptor("MBTR")["execution"]["device_limits"]

    assert matrix_limits["cuda"]["parameter_limits"]["n_atoms_max"]["maximum"] == 256
    assert c00ps_limits["cuda"]["parameter_limits"]["l_max"]["maximum"] == 20
    assert (
        mbtr_limits["cuda"]["conditional_limits"]
        ["global_valle_oganov_non_atomic_species"]["maximum_items"]
        == 64
    )
