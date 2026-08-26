"""Accuracy regression for every standalone descriptor with OpenMP support."""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms

from tests._public import (
    EAD,
    LMBTR,
    MBTR,
    SNAP,
    SO3,
    SO4,
    AtomicComposition,
    CoulombMatrix,
    EwaldSumMatrix,
    ExecutionOptions,
    LBispectrum,
    NeighborList,
    SineMatrix,
    StructureBatch,
    ValleOganov,
    builtin_registry,
)

TARGETS = (
    "CoulombMatrix",
    "SineMatrix",
    "EwaldSumMatrix",
    "MBTR",
    "LMBTR",
    "ValleOganov",
    "AtomicComposition",
    "NeighborList",
    "EAD",
    "SO3",
    "SO4",
    "SNAP",
    "LBispectrum",
)


def test_only_excluded_dpa_descriptors_lack_thread_capability() -> None:
    unsupported = {
        spec.name for spec in builtin_registry if "num_threads" not in spec.capabilities
    }
    assert unsupported == {"DPA4"}


def _batch() -> StructureBatch:
    cell = np.asarray(
        [[8.0, 0.2, 0.1], [0.1, 8.2, 0.3], [0.2, 0.1, 8.1]],
        dtype=np.float64,
    )
    positions = np.asarray(
        [[0.1, 0.2, 0.3], [1.3, 1.1, 1.0], [2.1, 0.4, 2.3], [3.4, 2.2, 1.8]],
        dtype=np.float64,
    )
    systems = [
        Atoms("NaCl2Si", positions=positions + shift, cell=cell, pbc=True)
        for shift in (
            np.asarray([0.0, 0.0, 0.0]),
            np.asarray([0.15, -0.1, 0.2]),
            np.asarray([-0.2, 0.1, -0.15]),
        )
    ]
    return StructureBatch.from_ase(systems)


def _descriptor(name: str, num_threads: int):
    execution = ExecutionOptions(num_threads=num_threads)
    species = [11, 14, 17]
    if name == "CoulombMatrix":
        return CoulombMatrix(n_atoms_max=4, permutation="none", execution=execution)
    if name == "SineMatrix":
        return SineMatrix(n_atoms_max=4, permutation="none", execution=execution)
    if name == "EwaldSumMatrix":
        return EwaldSumMatrix(n_atoms_max=4, permutation="none", execution=execution)
    if name in {"MBTR", "LMBTR"}:
        descriptor_type = MBTR if name == "MBTR" else LMBTR
        return descriptor_type(
            species=species,
            geometry={"function": "angle"},
            grid={"min": 0.0, "max": 180.0, "n": 24, "sigma": 0.2},
            weighting={"function": "exp", "scale": 0.3, "threshold": 1e-3},
            execution=execution,
        )
    if name == "ValleOganov":
        return ValleOganov(
            species=species,
            function="angle",
            n=24,
            sigma=0.2,
            r_cut=3.5,
            execution=execution,
        )
    if name == "AtomicComposition":
        return AtomicComposition(species=species, execution=execution)
    if name == "NeighborList":
        return NeighborList(cutoff=3.5, execution=execution)
    if name == "EAD":
        return EAD(
            parameters={"L": 2, "eta": [0.05, 0.1], "Rs": [0.0, 0.5]},
            Rc=3.5,
            execution=execution,
        )
    if name == "SO3":
        return SO3(nmax=2, lmax=2, rcut=3.5, execution=execution)
    if name == "SO4":
        return SO4(lmax=2, rcut=3.5, execution=execution)
    if name == "SNAP":
        return SNAP(lmax=2, rcut=3.5, execution=execution)
    if name == "LBispectrum":
        return LBispectrum(twojmax=4, diagonal=2, rcut=3.5, execution=execution)
    raise AssertionError(f"unknown OpenMP target: {name}")


@pytest.mark.parametrize("name", TARGETS)
def test_openmp_target_matches_single_thread(name: str) -> None:
    batch = _batch()
    serial = _descriptor(name, 1)
    threaded = _descriptor(name, 4)
    try:
        serial_result = serial.compute(batch)
        threaded_result = threaded.compute(batch)
        np.testing.assert_allclose(
            threaded_result.values,
            serial_result.values,
            rtol=2e-11,
            atol=2e-12,
        )
        assert threaded_result.level == serial_result.level
        assert threaded_result.labels == serial_result.labels
        if serial_result.samples is not None:
            np.testing.assert_array_equal(threaded_result.samples, serial_result.samples)
        if serial_result.row_offsets is not None:
            np.testing.assert_array_equal(threaded_result.row_offsets, serial_result.row_offsets)
        assert np.isfinite(np.asarray(threaded_result.values)).all()
    finally:
        serial.close()
        threaded.close()
