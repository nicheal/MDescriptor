import json
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms

from tests._public import MTP, ExecutionOptions, StructureBatch


def _system():
    return Atoms(
        "OHH",
        positions=[[4.0, 4.0, 4.0], [4.8, 4.2, 4.0], [3.7, 4.6, 4.3]],
        cell=np.diag([10.0, 10.0, 10.0]),
        pbc=True,
    )


def test_mtp_is_invariant_to_rigid_rotation_and_atom_order():
    system = _system()
    rotation = np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    center = np.asarray([5.0, 5.0, 5.0])
    rotated = system.copy()
    rotated.positions = (system.positions - center) @ rotation.T + center
    permuted = system[[2, 0, 1]]

    calculator = MTP(
        species=[1, 8],
        min_dist=0.1,
        max_dist=4.0,
        radial_basis_size=2,
        max_rank=2,
    )
    first_result = calculator.compute(StructureBatch.from_ase([system]))
    first = first_result.values
    second = calculator.compute(StructureBatch.from_ase([rotated])).values
    third = calculator.compute(StructureBatch.from_ase([permuted])).values

    np.testing.assert_allclose(first, second, rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(first, third[[1, 2, 0]], rtol=1e-11, atol=1e-12)
    assert first.shape == (3, calculator.feature_count)
    assert len(first_result.labels) == calculator.feature_count
    assert np.isfinite(first).all()


def test_mtp_native_mlip4_json_matches_official_radial_prefix():
    # Extracted from mlip-4-main/test/example_combined_pot.ipynb.  This is a
    # current MLIP-4 fixture, rather than a benchmark result from an earlier
    # run.
    potential = Path(__file__).parent / "data" / "mlip4_test_mtp.json"
    system = Atoms(
        numbers=[13, 14],
        positions=[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        cell=np.diag([10.0, 10.0, 10.0]),
        pbc=True,
    )
    calculator = MTP(
        species=[13, 14], model=potential, execution=ExecutionOptions(num_threads=1)
    )
    result = calculator.compute(StructureBatch.from_ase([system]))

    assert calculator.feature_count == 5
    assert result.values.shape == (2, 5)
    assert result.labels == tuple(f"mlip4:basis={i}" for i in range(5))
    assert result.metadata["details"]["official_format"] == "MLIP-4"
    assert result.metadata["details"]["official_mlip4"] is True
    np.testing.assert_allclose(
        result.values[0, :2],
        [0.655498644654072, -0.782935732461145],
        rtol=1e-12,
        atol=1e-13,
    )


def test_mtp_reloads_replaced_model_at_same_path(tmp_path):
    source = Path(__file__).parent / "data" / "mlip4_test_mtp.json"
    potential = json.loads(source.read_text(encoding="utf-8"))
    path = tmp_path / "potential.json"
    path.write_text(json.dumps(potential), encoding="utf-8")
    system = Atoms(
        numbers=[13, 14],
        positions=[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        cell=np.diag([10.0, 10.0, 10.0]),
        pbc=True,
    )
    batch = StructureBatch.from_ase([system])

    first = MTP(species=[13, 14], model=path)
    try:
        first_values = first.compute(batch).values.copy()

        potential[1]["PairDescriptorPot"]["params"][20] += 1.0
        path.write_text(json.dumps(potential), encoding="utf-8")

        second = MTP(species=[13, 14], model=path)
        try:
            second_values = second.compute(batch).values
        finally:
            second.close()
    finally:
        first.close()

    assert not np.allclose(first_values, second_values)


@pytest.mark.parametrize(
    ("radial_type", "radial_config"),
    [
        ("RadialBasisVdw", {"basis_size": 8, "mindist": 1.0, "maxdist": 5.0}),
        (
            "RadialBasisVdwDamped",
            {"basis_size": 2, "cutoff": 5.0, "params": [-0.33528, 2.86229, 1.001, 1.452]},
        ),
    ],
)
def test_mtp_native_mlip4_additional_radial_basis_classes(
    tmp_path, radial_type, radial_config
):
    source = Path(__file__).parent / "data" / "mlip4_test_mtp.json"
    potential = json.loads(source.read_text(encoding="utf-8"))
    pair = potential[1]["PairDescriptorPot"]
    pair["radial_basis"] = [radial_type, radial_config]
    radial_size = int(radial_config["basis_size"])
    pair["params"] = [
        (index + 1) / 100.0 for index in range(2 * 2 * 2 * radial_size + 2 + 5)
    ]
    path = tmp_path / f"{radial_type}.json"
    path.write_text(json.dumps(potential), encoding="utf-8")
    system = Atoms(
        numbers=[13, 14],
        positions=[[0.0, 0.0, 0.0], [2.2, 0.0, 0.0]],
        cell=np.diag([10.0, 10.0, 10.0]),
        pbc=True,
    )
    result = MTP(
        species=[13, 14], model=path, execution=ExecutionOptions(num_threads=1)
    ).compute(StructureBatch.from_ase([system]))
    assert result.values.shape == (2, 5)
    assert np.isfinite(result.values).all()
