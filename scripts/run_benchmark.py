"""Run the controlled, registry-derived descriptor benchmark."""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np

import mdescriptor
from mdescriptor import DescriptorConfiguration, StructureBatch, create_descriptor

ROOT = Path(__file__).resolve().parents[1]
BASELINE_MANIFEST = ROOT / "tests" / "data" / "numerical_baselines" / "manifest.json"


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


def _configuration(case: dict) -> DescriptorConfiguration:
    value = _restore_paths(case["configuration"])
    parameters = dict(value["parameters"])
    name = case["descriptor"]
    spec = mdescriptor.builtin_registry.get(name)
    execution = dict(parameters.get("execution", {}))
    execution["device"] = "cpu"
    if "num_threads" in spec.capabilities:
        execution["num_threads"] = 1
    else:
        execution["num_threads"] = None
    parameters["execution"] = execution
    return DescriptorConfiguration(value["schema_version"], value["descriptor"], parameters)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=5)
    args = parser.parse_args(argv)
    if args.warmup < 0 or args.repeat <= 0:
        raise SystemExit("warmup must be non-negative and repeat must be positive")

    manifest = json.loads(BASELINE_MANIFEST.read_text(encoding="utf-8"))
    cases = [case for case in manifest["cases"] if case["name"] != "MTP-MLIP4"]
    measurements = []
    for case in cases:
        descriptor = create_descriptor(_configuration(case))
        batch = _batch(case["input"])
        for _ in range(args.warmup):
            descriptor.compute(batch)
        elapsed = []
        for _ in range(args.repeat):
            start = time.perf_counter()
            result = descriptor.compute(batch)
            elapsed.append(time.perf_counter() - start)
        measurements.append(
            {
                "name": case["name"],
                "level": result.level.value,
                "rows": int(result.values.shape[0]),
                "features": int(result.values.shape[1]),
                "median_seconds": float(np.median(elapsed)),
                "p95_seconds": float(np.percentile(elapsed, 95)),
            }
        )
        descriptor.close()
    output = {
        "schema_version": 1,
        "package": "MDescriptor",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "execution": {"device": "cpu", "num_threads": 1},
        "cases": measurements,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.output} ({len(measurements)} descriptors)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
