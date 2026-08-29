"""Pinned deepmd-kit 3.2.0 comparisons for the bundled DPA4 descriptors.

deepmd-kit exposes DPA4 through ``eval_descriptor``.  DPA4C is graph-native in
deepmd-kit 3.2, so its official graph descriptor path is used here instead of
the dense sentinel-capacity path.  Both paths are exercised with periodic and
non-periodic inputs; the latter is represented by ``cells=None`` in DeepMD's
API, matching MDescriptor's all-zero cell/non-periodic contract.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from scripts.deepmd_reference import evaluate_batch

from mdescriptor import StructureBatch
from mdescriptor.descriptors import DPA4, DPA4C
from mdescriptor.models import DPA4_MODEL, DPA4C_MODEL

pytestmark = [pytest.mark.reference, pytest.mark.deepmd]
ROOT = Path(__file__).resolve().parents[2]


def _water() -> Atoms:
    return Atoms(
        "OHH",
        positions=[[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]],
        cell=np.diag([8.0, 8.0, 8.0]),
        pbc=True,
    )


def _water_nonperiodic() -> Atoms:
    return Atoms(
        "OHH",
        positions=[[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]],
        cell=np.zeros((3, 3), dtype=np.float64),
        pbc=False,
    )


def _reference_values(name: str, system: Atoms) -> np.ndarray:
    try:
        import deepmd  # noqa: F401
        import torch  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised in misconfigured CI
        pytest.fail(
            "DeepMD reference job requires deepmd-kit[torch]==3.2.0; "
            f"import failed: {exc}",
            pytrace=False,
        )
    model = DPA4_MODEL if name == "DPA4" else DPA4C_MODEL
    return evaluate_batch(name, model, StructureBatch.from_ase(system))


@pytest.mark.parametrize(
    ("name", "descriptor"),
    [("DPA4", DPA4), ("DPA4C", DPA4C)],
)
@pytest.mark.parametrize(
    "system_factory",
    [_water, _water_nonperiodic],
    ids=["periodic", "nonperiodic"],
)
def test_dpa_descriptors_match_deepmd_kit(name, descriptor, system_factory):
    system = system_factory()
    expected = _reference_values(name, system)

    native = descriptor(model=DPA4_MODEL if name == "DPA4" else DPA4C_MODEL)
    try:
        actual = np.asarray(
            native.compute(StructureBatch.from_ase(system)).values,
            dtype=np.float64,
        )
    finally:
        native.close()

    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=1e-5)


@pytest.mark.parametrize("name", ["DPA4", "DPA4C"])
def test_dpa_golden_nonperiodic_rows_match_deepmd_kit(name):
    """Keep the committed non-periodic golden rows tied to DeepMD output."""

    try:
        import deepmd  # noqa: F401
        import torch  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised in misconfigured CI
        pytest.fail(
            "DeepMD reference job requires deepmd-kit[torch]==3.2.0; "
            f"import failed: {exc}",
            pytrace=False,
        )
    fixture_dir = ROOT / "tests" / "golden" / name.lower()
    with np.load(fixture_dir / "input.npz") as arrays:
        batch = StructureBatch(
            np.asarray(arrays["numbers"], dtype=np.int32),
            np.asarray(arrays["positions"], dtype=np.float64),
            np.asarray(arrays["cells"], dtype=np.float64),
            np.asarray(arrays["pbc"], dtype=np.int32),
            np.asarray(arrays["offsets"], dtype=np.int64),
            ("hea32-periodic", "water3-nonperiodic"),
        )
    with np.load(fixture_dir / "expected_output.npz") as arrays:
        expected = np.asarray(arrays["values"], dtype=np.float64)

    model = DPA4_MODEL if name == "DPA4" else DPA4C_MODEL
    actual = evaluate_batch(name, model, batch)
    nonperiodic_begin = int(batch.offsets[1])
    np.testing.assert_allclose(
        expected[nonperiodic_begin:],
        actual[nonperiodic_begin:],
        rtol=2e-5,
        atol=1e-5,
    )
