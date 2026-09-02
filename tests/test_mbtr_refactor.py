"""Regression tests for the shared MBTR-family configuration seam."""

from __future__ import annotations

import numpy as np
from ase import Atoms

from tests._public import LMBTR, MBTR, ExecutionOptions, StructureBatch, ValleOganov


def _batch() -> StructureBatch:
    return StructureBatch.from_ase(
        [
            Atoms(
                numbers=[1, 8, 1],
                positions=[[0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [0.0, 1.2, 0.0]],
                cell=np.diag([8.0, 8.0, 8.0]),
                pbc=True,
            ),
            Atoms(
                numbers=[8, 1],
                positions=[[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]],
                cell=np.diag([8.0, 8.0, 8.0]),
                pbc=True,
            ),
        ]
    )


def test_valle_oganov_presets_match_generic_mbtr() -> None:
    batch = _batch()
    descriptors = (
        (
            ValleOganov(
                species=[1, 8], function="distance", n=24, sigma=0.08, r_cut=3.5
            ),
            MBTR(
                species=[1, 8],
                geometry={"function": "distance"},
                grid={"min": 0.0, "max": 3.5, "n": 24, "sigma": 0.08},
                weighting={"function": "inverse_square", "r_cut": 3.5},
                normalization="valle_oganov",
            ),
        ),
        (
            ValleOganov(
                species=[1, 8], function="angle", n=24, sigma=0.08, r_cut=3.5
            ),
            MBTR(
                species=[1, 8],
                geometry={"function": "angle"},
                grid={"min": 0.0, "max": 180.0, "n": 24, "sigma": 0.08},
                weighting={"function": "smooth_cutoff", "r_cut": 3.5},
                normalization="valle_oganov",
            ),
        ),
    )
    try:
        for valle, generic in descriptors:
            np.testing.assert_allclose(
                valle.compute(batch).values,
                generic.compute(batch).values,
                rtol=1e-12,
                atol=1e-12,
            )
    finally:
        for valle, generic in descriptors:
            valle.close()
            generic.close()


def test_valle_oganov_explicit_overrides_use_the_common_configuration() -> None:
    batch = _batch()
    parameters = {
        "species": [1, 8],
        "geometry": {"function": "inverse_distance"},
        "grid": {"min": 0.0, "max": 2.0, "n": 20, "sigma": 0.1},
        "weighting": {"function": "exp", "scale": 0.7, "threshold": 0.01},
        "normalization": "none",
    }
    valle = ValleOganov(
        function="distance",
        n=20,
        sigma=0.1,
        r_cut=3.5,
        **parameters,
    )
    generic = MBTR(**parameters)
    try:
        np.testing.assert_allclose(
            valle.compute(batch).values,
            generic.compute(batch).values,
            rtol=1e-12,
            atol=1e-12,
        )
    finally:
        valle.close()
        generic.close()


def test_mbtr_cuda_payload_contains_the_resolved_named_controls() -> None:
    descriptor = ValleOganov(
        species=[1, 8],
        function="distance",
        n=20,
        sigma=0.1,
        r_cut=3.5,
        geometry={"function": "inverse_distance"},
        grid={"min": 0.0, "max": 2.0, "n": 20, "sigma": 0.1},
        weighting={"function": "exp", "scale": 0.7, "threshold": 0.01},
        normalization="none",
        execution=ExecutionOptions(device="cuda"),
    )
    try:
        payload = descriptor._backend.options["_cuda_payload"]["mbtr_config"]
        assert payload["species"] == [1, 8]
        assert payload["geometry"] == 2
        assert payload["weighting"] == 1
        assert payload["normalization"] == 0
        assert payload["grid_n"] == 20
        assert payload["local"] is False
        assert "_cuda_payload" not in descriptor.configuration.to_dict()
    finally:
        descriptor.close()


def test_lmbtr_atomic_number_is_one_row_per_atom_and_one_central_channel() -> None:
    descriptor = LMBTR(
        species=[1, 8],
        geometry={"function": "atomic_number"},
        grid={"min": 0.0, "max": 10.0, "n": 20, "sigma": 0.1},
        weighting={"function": "unity"},
        normalization="none",
    )
    try:
        result = descriptor.compute(_batch())
        assert result.level == "atom"
        assert result.values.shape == (5, 40)
        np.testing.assert_array_equal(result.row_offsets, [0, 3, 5])
        channels = result.values.reshape(5, 2, 20)
        for row, number in enumerate([1, 8, 1, 8, 1]):
            central = 0 if number == 1 else 1
            other = 1 - central
            assert np.max(channels[row, central]) > 0.0
            np.testing.assert_array_equal(channels[row, other], 0.0)
    finally:
        descriptor.close()
