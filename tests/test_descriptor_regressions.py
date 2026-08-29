"""Regression tests for descriptor correctness issues found by adversarial review."""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap

import pytest
from ase import Atoms

import mdescriptor
from mdescriptor import DescriptorConfigError, DescriptorConfiguration, StructureBatch
from mdescriptor.descriptors import (
    SOAP,
    AtomicComposition,
    NeighborList,
    SortedDistances,
)


def test_neighbor_list_feature_count_matches_public_features() -> None:
    batch = StructureBatch.from_ase(
        [Atoms("H2", positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])]
    )

    descriptor = NeighborList(cutoff=2.0)
    result = descriptor.compute(batch)

    assert descriptor.feature_count == result.feature_count == 4
    assert len(result.labels) == result.values.shape[1] == 4


def test_ewald_matrix_empty_periodic_frame_does_not_abort_process() -> None:
    script = textwrap.dedent(
        """
        import json
        import numpy as np
        from mdescriptor import StructureBatch
        from mdescriptor.descriptors import EwaldSumMatrix

        batch = StructureBatch(
            np.array([1, 1], dtype=np.int32),
            np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            np.stack([np.eye(3) * 8.0, np.eye(3) * 8.0]),
            np.ones((2, 3), dtype=np.int32),
            np.array([0, 2, 2], dtype=np.int64),
            ("full", "empty"),
        )
        result = EwaldSumMatrix(n_atoms_max=4).compute(batch)
        print(json.dumps({"shape": list(result.values.shape), "empty": result.values[1].tolist()}))
        """
    )

    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["shape"] == [2, 16]
    assert payload["empty"] == [0.0] * 16


def test_lode_schema_matches_native_exponent_range() -> None:
    schema = mdescriptor.describe_descriptor("LodeSphericalExpansion")["parameters"][
        "exponent"
    ]
    assert schema["minimum"] == 1
    assert schema["maximum"] == 9

    for exponent in (0, 10):
        configuration = DescriptorConfiguration(
            1,
            "LodeSphericalExpansion",
            {"species": [1], "exponent": exponent},
        )
        with pytest.raises(DescriptorConfigError, match="violates"):
            mdescriptor.create_descriptor(configuration)


@pytest.mark.parametrize(
    ("descriptor", "kwargs"),
    [
        (SOAP, {"species": [1], "r_cut": 2.0, "n_max": 1.5}),
        (AtomicComposition, {"species": [1], "per_system": 1}),
        (SortedDistances, {"species": [1], "max_neighbors": 1.5}),
    ],
)
def test_direct_constructors_reject_schema_invalid_parameter_types(
    descriptor: type, kwargs: dict[str, object]
) -> None:
    with pytest.raises(DescriptorConfigError, match="does not match type"):
        descriptor(**kwargs)
