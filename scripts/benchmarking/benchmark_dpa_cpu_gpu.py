"""Benchmark the bundled DPA4/DPA4C models on CPU and CUDA.

The GPU measurement is the first synchronous ``compute`` call.  It includes
lazy CUDA plugin loading, context creation, and device model upload because
those costs are part of the first public CUDA call.  CPU gets one first call
and a small steady-state sample for comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
from ase.io import read

import mdescriptor
from mdescriptor import ExecutionOptions, StructureBatch
from mdescriptor.descriptors import DPA4, DPA4C
from mdescriptor.models import DPA4_MODEL, DPA4C_MODEL

ROOT = Path(__file__).resolve().parents[2]
TWO_DATASET = ROOT / "benchmarks/_datasets/two-structure-v1-2a727a880fef"
CARBON_DATASET = ROOT / "benchmarks/_datasets/legacy/carbon_dataset_pbc.xyz"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_cuda_plugin() -> None:
    """Load the source-tree CUDA extension without touching the driver early."""

    candidates: list[Path] = []
    configured = os.environ.get("MDESCRIPTOR_CUDA_PLUGIN_DIR")
    if configured:
        candidates.append(Path(configured))
    candidates.append(ROOT / "build-cuda")
    for candidate in candidates:
        if not any(candidate.glob("_cuda*.so")):
            continue
        candidate_text = str(candidate)
        if candidate_text not in mdescriptor.__path__:
            mdescriptor.__path__.insert(0, candidate_text)
        try:
            importlib.import_module("mdescriptor._cuda")
        except (ImportError, OSError):
            continue
        return
    raise RuntimeError("CUDA plugin is not available in this checkout")


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


def _two_structure_batch() -> StructureBatch:
    with np.load(TWO_DATASET / "structures.npz") as arrays:
        return StructureBatch(
            np.asarray(arrays["numbers"], dtype=np.int32),
            np.asarray(arrays["positions"], dtype=np.float64),
            np.asarray(arrays["cells"], dtype=np.float64),
            np.asarray(arrays["pbc"], dtype=np.int32),
            np.asarray(arrays["offsets"], dtype=np.int64),
            ("hea32-periodic", "water3-nonperiodic"),
        )


def _carbon_frame(index: int) -> StructureBatch:
    structure = read(CARBON_DATASET, index=index)
    return StructureBatch.from_ase([structure], ids=[f"carbon-{index:04d}"])


def _workloads() -> list[dict[str, Any]]:
    two = _two_structure_batch()
    return [
        {
            "name": "two-structure-v1/hea32-periodic",
            "source": str(TWO_DATASET / "structures.npz"),
            "source_sha256": _sha256(TWO_DATASET / "structures.npz"),
            "batch": _single_structure(two, 0),
        },
        {
            "name": "two-structure-v1/water3-nonperiodic",
            "source": str(TWO_DATASET / "structures.npz"),
            "source_sha256": _sha256(TWO_DATASET / "structures.npz"),
            "batch": _single_structure(two, 1),
        },
        {
            "name": "carbon_dataset_pbc/frame-0358-C4",
            "source": str(CARBON_DATASET),
            "source_sha256": _sha256(CARBON_DATASET),
            "batch": _carbon_frame(358),
        },
    ]


def _accuracy(actual: Any, expected: Any) -> dict[str, Any]:
    actual_values = np.asarray(actual.values, dtype=np.float64)
    expected_values = np.asarray(expected.values, dtype=np.float64)
    difference = np.abs(actual_values - expected_values)
    denominator = np.maximum(np.abs(expected_values), 1.0e-12)
    tolerance = 1.0e-5 + 2.0e-5 * np.abs(expected_values)
    return {
        "shape": list(actual_values.shape),
        "max_abs_error": float(np.max(difference)),
        "mae": float(np.mean(difference)),
        "rmse": float(np.sqrt(np.mean(np.square(difference)))),
        "max_relative_error": float(np.max(difference / denominator)),
        "allclose": bool(np.all(difference <= tolerance)),
        "rtol": 2.0e-5,
        "atol": 1.0e-5,
    }


def _metadata(workload: dict[str, Any]) -> dict[str, Any]:
    batch = workload["batch"]
    return {
        "name": workload["name"],
        "source": workload["source"],
        "source_sha256": workload["source_sha256"],
        "frames": len(batch.offsets) - 1,
        "atoms": int(batch.numbers.size),
        "species": sorted({int(value) for value in batch.numbers}),
        "pbc": batch.pbc.tolist(),
    }


def _measure(
    descriptor_type: type[Any],
    model: Path,
    workload: dict[str, Any],
    cpu_threads: int,
    cpu_repeat: int,
) -> dict[str, Any]:
    batch = workload["batch"]
    workload_name = workload["name"]
    print(f"[{descriptor_type.__name__}] {workload_name}: CPU", flush=True)
    cpu_started = time.perf_counter()
    cpu = descriptor_type(
        model=model,
        execution=ExecutionOptions(device="cpu", num_threads=cpu_threads),
    )
    cpu_construct_seconds = time.perf_counter() - cpu_started
    try:
        started = time.perf_counter()
        expected = cpu.compute(batch)
        cpu_first_seconds = time.perf_counter() - started
        cpu_samples: list[float] = []
        for _ in range(cpu_repeat):
            started = time.perf_counter()
            cpu.compute(batch)
            cpu_samples.append(time.perf_counter() - started)
    finally:
        cpu.close()

    print(f"[{descriptor_type.__name__}] {workload_name}: GPU", flush=True)
    gpu_started = time.perf_counter()
    gpu = descriptor_type(
        model=model,
        execution=ExecutionOptions(device="cuda"),
    )
    gpu_construct_seconds = time.perf_counter() - gpu_started
    try:
        started = time.perf_counter()
        actual = gpu.compute(batch)
        gpu_first_seconds = time.perf_counter() - started
    finally:
        gpu.close()

    accuracy = _accuracy(actual, expected)
    result = {
        "descriptor": descriptor_type.__name__,
        "dataset": _metadata(workload),
        "feature_count": int(actual.feature_count),
        "cpu": {
            "threads": cpu_threads,
            "construct_seconds": cpu_construct_seconds,
            "first_compute_seconds": cpu_first_seconds,
            "steady_samples_seconds": cpu_samples,
            "steady_median_seconds": float(np.median(cpu_samples)),
        },
        "gpu": {
            "construct_seconds": gpu_construct_seconds,
            "first_compute_seconds": gpu_first_seconds,
            "includes_lazy_setup": True,
        },
        "speedup_vs_cpu_first": cpu_first_seconds / gpu_first_seconds,
        "speedup_vs_cpu_steady": float(np.median(cpu_samples)) / gpu_first_seconds,
        "accuracy": accuracy,
    }
    print(
        f"[{result['descriptor']}] {workload_name}: "
        f"CPU first={cpu_first_seconds:.6f}s, "
        f"GPU first={gpu_first_seconds:.6f}s, "
        f"max_abs={accuracy['max_abs_error']:.6e}, "
        f"allclose={accuracy['allclose']}",
        flush=True,
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--cpu-repeat", type=int, default=3)
    parser.add_argument(
        "--descriptors",
        default="DPA4C,DPA4",
        help="comma-separated descriptors; default runs DPA4C then DPA4",
    )
    args = parser.parse_args(argv)
    if args.cpu_threads < 1 or args.cpu_repeat < 1:
        raise SystemExit("cpu-threads and cpu-repeat must be positive")
    descriptor_specs = {
        "DPA4": (DPA4, DPA4_MODEL),
        "DPA4C": (DPA4C, DPA4C_MODEL),
    }
    names = [name.strip() for name in args.descriptors.split(",") if name.strip()]
    if not names or any(name not in descriptor_specs for name in names):
        raise SystemExit("descriptors must contain only DPA4 and DPA4C")

    _load_cuda_plugin()
    workloads = _workloads()
    measurements = []
    for name in names:
        descriptor_type, model = descriptor_specs[name]
        for workload in workloads:
            measurements.append(
                _measure(
                    descriptor_type,
                    model,
                    workload,
                    args.cpu_threads,
                    args.cpu_repeat,
                )
            )
    result = {
        "schema_version": 1,
        "package": "MDescriptor",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_threads": args.cpu_threads,
        "cpu_repeat": args.cpu_repeat,
        "measurement_definition": {
            "gpu": "first synchronous compute; includes lazy CUDA setup",
            "cpu_first": "first synchronous compute after construction",
            "cpu_steady": "median of subsequent repeated computes",
        },
        "measurements": measurements,
    }
    encoded = json.dumps(result, indent=2, sort_keys=True)
    if args.output is None:
        print(encoded)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
