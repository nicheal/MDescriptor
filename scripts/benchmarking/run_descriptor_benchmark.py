"""Run the controlled CPU benchmark from independent golden fixtures."""

from __future__ import annotations

import argparse
import json
import os
import platform
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

import mdescriptor  # noqa: E402
from mdescriptor import DescriptorConfiguration, StructureBatch, create_descriptor  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GOLDEN_ROOT = ROOT / "tests" / "golden"


def _restore_paths(value: Any) -> Any:
    if isinstance(value, str) and value.startswith("${PROJECT_ROOT}/"):
        return str(ROOT / value.removeprefix("${PROJECT_ROOT}/"))
    if isinstance(value, dict):
        return {key: _restore_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_restore_paths(item) for item in value]
    return value


def _batch(path: Path, ids: tuple[str, ...]) -> StructureBatch:
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
        batch = _batch(fixture_dir / manifest["input"], tuple(manifest["input_ids"]))
        compute_batch = (
            _single_structure(batch, 0)
            if manifest["nonperiodic"]["mode"] != "output"
            else batch
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
