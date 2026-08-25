"""Official DPA4 and DPA4C checkpoint goldens in one schema."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from mdescriptor import StructureBatch
from mdescriptor.descriptors import DPA4, DPA4C
from mdescriptor.models import DPA4_MODEL, DPA4C_MODEL

pytestmark = pytest.mark.model

ROOT = Path(__file__).parents[1]
_MODEL_TOLERANCES = {
    # DPA checkpoints carry float32 weights.  BLAS/NumPy implementations can
    # differ by a few ulps between supported Python/OS wheels, so retain the
    # relative gate while allowing the observed platform-level absolute drift.
    "DPA4": {"rtol": 2e-5, "atol": 1e-5},
    "DPA4C": {"rtol": 2e-5, "atol": 1e-5},
}


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize(
    ("name", "descriptor_type", "model", "fixture"),
    [
        ("DPA4", DPA4, DPA4_MODEL, "dpa4_air_h2o_golden.json"),
        ("DPA4C", DPA4C, DPA4C_MODEL, "dpa4c_air_h2o_golden.json"),
    ],
)
def test_official_model_golden_schema_and_values(name, descriptor_type, model, fixture):
    payload = json.loads((ROOT / "tests" / "data" / fixture).read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["descriptor"] == name
    assert payload["model"]["sha256"] == _digest(model)
    batch = StructureBatch(
        np.asarray(payload["numbers"], dtype=np.int32),
        np.asarray(payload["positions"], dtype=np.float64),
        np.asarray(payload["cells"], dtype=np.float64),
        np.asarray(payload["pbc"], dtype=np.int32),
        np.asarray(payload["offsets"], dtype=np.int64),
        tuple(payload["ids"]),
    )
    result = descriptor_type(model=model).compute(batch)
    expected = np.asarray(payload["values"], dtype=np.float64)
    np.testing.assert_allclose(result.values, expected, **_MODEL_TOLERANCES[name])
    np.testing.assert_array_equal(result.samples, payload["samples"])
    assert result.level.value == payload["level"]
    assert result.feature_count == payload["feature_count"]
    assert result.labels == tuple(payload["labels"])
