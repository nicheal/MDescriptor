"""Shared loader for descriptor-specific, benchmark-independent goldens."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import mdescriptor
from mdescriptor import DescriptorConfiguration, StructureBatch, create_descriptor

ROOT = Path(__file__).parents[1]
PACKAGE_ROOT = Path(mdescriptor.__file__).resolve().parent
GOLDEN_ROOT = ROOT / "tests" / "golden"


def _restore_paths(value: Any) -> Any:
    if isinstance(value, str):
        if value.startswith("${PACKAGE_ROOT}/"):
            return str(PACKAGE_ROOT / value.removeprefix("${PACKAGE_ROOT}/"))
        if value.startswith("${PROJECT_ROOT}/"):
            return str(ROOT / value.removeprefix("${PROJECT_ROOT}/"))
    if isinstance(value, dict):
        return {key: _restore_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_restore_paths(item) for item in value]
    return value


def _batch_from_npz(path: Path, ids: tuple[str, ...]) -> StructureBatch:
    with np.load(path) as arrays:
        return StructureBatch(
            np.asarray(arrays["numbers"], dtype=np.int32),
            np.asarray(arrays["positions"], dtype=np.float64),
            np.asarray(arrays["cells"], dtype=np.float64),
            np.asarray(arrays["pbc"], dtype=np.int32),
            np.asarray(arrays["offsets"], dtype=np.int64),
            ids,
        )


def _single_structure(batch: StructureBatch, index: int) -> StructureBatch:
    begin = int(batch.offsets[index])
    end = int(batch.offsets[index + 1])
    return StructureBatch(
        batch.numbers[begin:end],
        batch.positions[begin:end],
        batch.cells[index : index + 1],
        batch.pbc[index : index + 1],
        np.asarray([0, end - begin], dtype=np.int64),
        (batch.ids[index],),
    )


def _descriptor(manifest: dict[str, Any]):
    configuration = DescriptorConfiguration.from_dict(_restore_paths(manifest["configuration"]))
    return create_descriptor(configuration)


def _assert_process_abort(manifest: dict[str, Any], batch: StructureBatch, match: str) -> None:
    nonperiodic = _single_structure(batch, 1)
    payload = {
        "numbers": nonperiodic.numbers.tolist(),
        "positions": nonperiodic.positions.tolist(),
        "cells": nonperiodic.cells.tolist(),
        "pbc": nonperiodic.pbc.tolist(),
        "offsets": nonperiodic.offsets.tolist(),
        "ids": list(nonperiodic.ids),
    }
    script = (
        "import json, sys, numpy as np\n"
        "from mdescriptor import DescriptorConfiguration, StructureBatch, create_descriptor\n"
        "configuration = DescriptorConfiguration.from_dict(json.loads(sys.argv[1]))\n"
        "data = json.loads(sys.argv[2])\n"
        "batch = StructureBatch(np.asarray(data['numbers'], dtype=np.int32), "
        "np.asarray(data['positions'], dtype=np.float64), "
        "np.asarray(data['cells'], dtype=np.float64), "
        "np.asarray(data['pbc'], dtype=np.int32), "
        "np.asarray(data['offsets'], dtype=np.int64), tuple(data['ids']))\n"
        "create_descriptor(configuration).compute(batch)\n"
    )
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            json.dumps(_restore_paths(manifest["configuration"])),
            json.dumps(payload),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert probe.returncode != 0
    assert match in probe.stderr


def _assert_result(result: Any, expected: dict[str, Any], arrays: Any, tolerance: dict[str, float]) -> None:
    np.testing.assert_allclose(
        result.values,
        arrays["values"],
        rtol=tolerance["rtol"],
        atol=tolerance["atol"],
    )
    np.testing.assert_array_equal(result.samples, arrays["samples"])
    assert result.level.value == expected["level"]
    assert result.feature_count == expected["feature_count"]
    assert result.labels == tuple(expected["labels"])
    assert result.structure_ids == tuple(expected["structure_ids"])
    if expected["row_offsets"] is None:
        assert result.row_offsets is None
    else:
        np.testing.assert_array_equal(result.row_offsets, expected["row_offsets"])
    assert result.metadata == _restore_paths(expected["metadata"])


def _assert_nonperiodic_contract(name: str, manifest: dict[str, Any], batch: StructureBatch) -> None:
    policy = manifest["nonperiodic"]
    if policy["mode"] == "output":
        return
    if policy["mode"] == "contract_rejection":
        configuration = _restore_paths(manifest["configuration"])
        parameters = dict(configuration["parameters"])
        parameters["periodic"] = False
        invalid = dict(configuration)
        invalid["parameters"] = parameters
        with pytest.raises(ValueError, match=policy["match"]):
            create_descriptor(DescriptorConfiguration.from_dict(invalid))
        return
    if policy["mode"] == "error":
        if policy["type"] == "process_abort":
            _assert_process_abort(manifest, batch, policy["match"])
            return
        descriptor = _descriptor(manifest)
        try:
            with pytest.raises(Exception, match=policy["match"]):
                descriptor.compute(_single_structure(batch, 1))
        finally:
            descriptor.close()
        return
    raise AssertionError(f"unknown non-periodic policy: {policy}")


def assert_descriptor_golden_at(
    name: str,
    fixture_dir: Path,
    manifest_name: str = "manifest.json",
) -> None:
    """Assert one descriptor golden using an explicitly selected manifest."""

    manifest = json.loads((fixture_dir / manifest_name).read_text(encoding="utf-8"))
    assert manifest["descriptor"] == name
    expected = manifest["result"]
    ids = tuple(manifest["input_ids"])
    batch = _batch_from_npz(fixture_dir / manifest["input"], ids)
    periodic_only = manifest["nonperiodic"]["mode"] != "output"
    compute_batch = _single_structure(batch, 0) if periodic_only else batch
    descriptor = _descriptor(manifest)
    try:
        with np.load(fixture_dir / manifest["expected_output"]) as arrays:
            _assert_result(descriptor.compute(compute_batch), expected, arrays, manifest["tolerance"])
    finally:
        descriptor.close()
    _assert_nonperiodic_contract(name, manifest, batch)


def assert_descriptor_golden(name: str) -> None:
    """Assert the retained project snapshot golden."""

    assert_descriptor_golden_at(name, GOLDEN_ROOT / name.lower())


def assert_descriptor_external_static_golden(name: str) -> None:
    """Assert the provider-generated static numerical sidecar."""

    assert_descriptor_golden_at(
        name,
        GOLDEN_ROOT / name.lower(),
        manifest_name="external_manifest.json",
    )
