"""Run the controlled CPU benchmark from independent golden fixtures."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

# Keep native and BLAS kernels single-threaded before importing NumPy.
for _thread_env in (
    "OMP_NUM_THREADS",
    "OMP_DYNAMIC",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_env] = "FALSE" if _thread_env == "OMP_DYNAMIC" else "1"

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # noqa: E402
import mdescriptor  # noqa: E402
from external_reference import (  # noqa: E402
    _batch_from_npz,
    _restore_paths,
    _single_structure,
)
from mdescriptor import DescriptorConfiguration, create_descriptor  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GOLDEN_ROOT = ROOT / "tests" / "golden"


def _accuracy(result: Any, manifest: dict[str, Any], arrays: Any) -> dict[str, Any]:
    actual = np.asarray(result.values)
    expected = np.asarray(arrays["values"])
    tolerance = manifest["tolerance"]
    if actual.shape != expected.shape:
        return {
            "shape": {"actual": list(actual.shape), "expected": list(expected.shape)},
            "pass": False,
        }
    error = np.abs(actual - expected)
    nonzero = expected != 0
    expected_result = manifest["result"]
    offsets = expected_result["row_offsets"]
    contract = (
        result.level.value == expected_result["level"]
        and result.feature_count == expected_result["feature_count"]
        and result.labels == tuple(expected_result["labels"])
        and result.structure_ids == tuple(expected_result["structure_ids"])
        and (
            offsets is None
            if result.row_offsets is None
            else np.array_equal(result.row_offsets, offsets)
        )
        and np.array_equal(result.samples, arrays["samples"])
    )
    values_pass = bool(
        np.allclose(actual, expected, rtol=tolerance["rtol"], atol=tolerance["atol"])
    )
    return {
        "shape": list(actual.shape),
        "values": int(actual.size),
        "max_abs_error": float(error.max(initial=0.0)),
        "max_rel_error_nonzero_reference": float(
            (error[nonzero] / np.abs(expected[nonzero])).max(initial=0.0)
        ),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "tolerance": tolerance,
        "values_pass": values_pass,
        "result_contract_pass": contract,
        "pass": values_pass and contract,
    }


def _configuration(manifest: dict[str, Any]) -> DescriptorConfiguration:
    value = _restore_paths(manifest["configuration"])
    parameters = dict(value["parameters"])
    spec = mdescriptor.builtin_registry.get(manifest["descriptor"])
    execution = dict(parameters.get("execution", {}))
    execution["device"] = "cpu"
    execution["num_threads"] = 1 if "num_threads" in spec.capabilities else None
    parameters["execution"] = execution
    return DescriptorConfiguration(value["schema_version"], value["descriptor"], parameters)


def _cases(root: Path) -> list[tuple[Path, dict[str, Any]]]:
    return [
        (path.parent, json.loads(path.read_text(encoding="utf-8")))
        for path in sorted(root.glob("*/manifest.json"))
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--golden-root", type=Path, default=DEFAULT_GOLDEN_ROOT)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=5)
    args = parser.parse_args(argv)
    if args.warmup < 0 or args.repeat <= 0:
        raise SystemExit("warmup must be non-negative and repeat must be positive")

    measurements = []
    for fixture_dir, manifest in _cases(args.golden_root):
        batch = _batch_from_npz(fixture_dir / manifest["input"], tuple(manifest["input_ids"]))
        compute_batch = (
            _single_structure(batch, 0) if manifest["nonperiodic"]["mode"] != "output" else batch
        )
        descriptor = create_descriptor(_configuration(manifest))
        try:
            for _ in range(args.warmup):
                descriptor.compute(compute_batch)
            elapsed = []
            for _ in range(args.repeat):
                started = time.perf_counter()
                result = descriptor.compute(compute_batch)
                elapsed.append(time.perf_counter() - started)
            accuracy_manifest_path = fixture_dir / "external_manifest.json"
            accuracy_manifest = (
                json.loads(accuracy_manifest_path.read_text(encoding="utf-8"))
                if accuracy_manifest_path.is_file()
                else manifest
            )
            accuracy_result = result
            if accuracy_manifest is not manifest:
                accuracy_batch = _batch_from_npz(
                    fixture_dir / accuracy_manifest["input"],
                    tuple(accuracy_manifest["input_ids"]),
                )
                accuracy_batch = (
                    _single_structure(accuracy_batch, 0)
                    if accuracy_manifest["nonperiodic"]["mode"] != "output"
                    else accuracy_batch
                )
                accuracy_descriptor = create_descriptor(_configuration(accuracy_manifest))
                try:
                    accuracy_result = accuracy_descriptor.compute(accuracy_batch)
                finally:
                    accuracy_descriptor.close()
            with np.load(fixture_dir / accuracy_manifest["expected_output"]) as arrays:
                accuracy = _accuracy(accuracy_result, accuracy_manifest, arrays)
            measurements.append(
                {
                    "name": manifest["descriptor"],
                    "level": result.level.value,
                    "rows": int(result.values.shape[0]),
                    "features": int(result.values.shape[1]),
                    "nonperiodic_mode": manifest["nonperiodic"]["mode"],
                    "raw_seconds": elapsed,
                    "median_seconds": float(np.median(elapsed)),
                    "p95_seconds": float(np.percentile(elapsed, 95)),
                    "accuracy_reference": accuracy_manifest.get(
                        "numeric_baseline", accuracy_manifest.get("reference", {})
                    ),
                    "accuracy_dataset_sha256": accuracy_manifest.get("dataset", {}).get("sha256"),
                    "accuracy": accuracy,
                }
            )
        finally:
            descriptor.close()
    output = {
        "schema_version": 2,
        "package": "MDescriptor",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "execution": {"device": "cpu", "num_threads": 1},
        "thread_limits": {"openmp": 1, "blas": 1},
        "cases": measurements,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.output} ({len(measurements)} descriptors)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
