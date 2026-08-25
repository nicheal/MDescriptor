"""Hard numerical and result-layout gate for the frozen reference evidence."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from mdescriptor import DescriptorConfiguration, StructureBatch, create_descriptor, list_descriptors

pytestmark = pytest.mark.model

ROOT = Path(__file__).parents[1]
BASELINE_DIR = ROOT / "tests" / "data" / "numerical_baselines"


def _restore_paths(value):
    if isinstance(value, str) and value.startswith("${PROJECT_ROOT}/"):
        return str(ROOT / value.removeprefix("${PROJECT_ROOT}/"))
    if isinstance(value, dict):
        return {key: _restore_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_restore_paths(item) for item in value]
    return value


def _batch(payload: dict) -> StructureBatch:
    return StructureBatch(
        np.asarray(payload["numbers"], dtype=np.int32),
        np.asarray(payload["positions"], dtype=np.float64),
        np.asarray(payload["cells"], dtype=np.float64),
        np.asarray(payload["pbc"], dtype=np.int32),
        np.asarray(payload["offsets"], dtype=np.int64),
        tuple(payload["ids"]),
    )


def test_frozen_reference_manifest_covers_all_descriptors_and_mtp_modes():
    manifest = json.loads((BASELINE_DIR / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1
    assert manifest["source_commit"] == "60dccbb"
    cases = manifest["cases"]
    assert len(cases) == 28
    assert {case["name"] for case in cases if case["name"] != "MTP-MLIP4"} == set(
        list_descriptors()
    )

    for case in cases:
        configuration = _restore_paths(case["configuration"])
        descriptor = create_descriptor(DescriptorConfiguration.from_dict(configuration))
        result = descriptor.compute(_batch(case["input"]))
        with np.load(BASELINE_DIR / case["values"]) as arrays:
            expected_values = arrays["values"]
            expected_samples = arrays["samples"]
        tolerance = case["tolerance"]
        np.testing.assert_allclose(
            result.values,
            expected_values,
            rtol=tolerance["rtol"],
            atol=tolerance["atol"],
            err_msg=case["name"],
        )
        np.testing.assert_array_equal(result.samples, expected_samples)
        assert result.level.value == case["level"]
        assert result.feature_count == case["feature_count"]
        assert result.labels == tuple(case["labels"])
        assert result.structure_ids == tuple(case["structure_ids"])
        if case["row_offsets"] is None:
            assert result.row_offsets is None
        else:
            np.testing.assert_array_equal(result.row_offsets, case["row_offsets"])
        descriptor.close()
