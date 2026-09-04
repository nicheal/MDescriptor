"""Compare CUDA NEP with CPU NEP and NEPAdapters on bundled datasets.

This is a host-GPU benchmark.  It intentionally reports both CPU/GPU parity
and NEPAdapters parity because the CPU implementation and NEPAdapters use
different floating-point/neighbor accumulation paths.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.io import read

from mdescriptor import ExecutionOptions, StructureBatch
from mdescriptor._cuda_loader import load_cuda_plugin
from mdescriptor.descriptors import NEP
from mdescriptor.models import NEP_MODEL

ROOT = Path(__file__).resolve().parents[2]
TWO_DATASET = ROOT / "benchmarks/_datasets/two-structure-v1-2a727a880fef/structures.npz"
CARBON_DATASET = ROOT / "benchmarks/_datasets/legacy/carbon_dataset_pbc.xyz"
SOAP_DATASET = ROOT / "benchmarks/_datasets/legacy/soap_diverse_dataset_300.xyz"
NEP_FIXTURE = ROOT / "tests/golden/nep/external_input.npz"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _npz_batch(path: Path, ids: tuple[str, ...]) -> StructureBatch:
    with np.load(path) as arrays:
        return StructureBatch(
            np.asarray(arrays["numbers"], dtype=np.int32),
            np.asarray(arrays["positions"], dtype=np.float64),
            np.asarray(arrays["cells"], dtype=np.float64),
            np.asarray(arrays["pbc"], dtype=np.int32),
            np.asarray(arrays["offsets"], dtype=np.int64),
            ids,
        )


def _single_structure(batch: StructureBatch, index: int, name: str) -> StructureBatch:
    begin = int(batch.offsets[index])
    end = int(batch.offsets[index + 1])
    return StructureBatch(
        batch.numbers[begin:end],
        batch.positions[begin:end],
        batch.cells[index : index + 1],
        batch.pbc[index : index + 1],
        np.asarray([0, end - begin], dtype=np.int64),
        (name,),
    )


def _random_periodic() -> StructureBatch:
    rng = np.random.default_rng(20260831)
    structures = []
    for _index in range(8):
        cell_length = 16.0
        positions = rng.random((64, 3)) * cell_length
        numbers = np.resize(np.asarray([1, 6, 8, 14], dtype=np.int32), 64)
        structures.append(
            Atoms(
                numbers=numbers,
                positions=positions,
                cell=np.eye(3) * cell_length,
                pbc=True,
            )
        )
    return StructureBatch.from_ase(structures, ids=[f"random-{index}" for index in range(8)])


def _cases(include_full_carbon: bool = False) -> list[tuple[str, Path, StructureBatch]]:
    nep_fixture = _npz_batch(NEP_FIXTURE, ("water3-periodic",))
    two = _npz_batch(TWO_DATASET, ("hea32-periodic", "water3-nonperiodic"))
    carbon_frame = StructureBatch.from_ase(
        [read(CARBON_DATASET, index=34)], ids=["carbon-pbc-frame34"]
    )
    carbon_first64 = StructureBatch.from_ase(
        list(read(CARBON_DATASET, index=slice(0, 64))),
        ids=[f"carbon-pbc-{index:02d}" for index in range(64)],
    )
    soap = StructureBatch.from_ase(
        list(read(SOAP_DATASET, index=slice(None))),
        ids=[f"soap-diverse-{index:03d}" for index in range(300)],
    )
    cases = [
        ("water3-periodic", NEP_FIXTURE, nep_fixture),
        ("hea32-periodic", TWO_DATASET, _single_structure(two, 0, "hea32-periodic")),
        ("two-structure-mixed", TWO_DATASET, two),
        ("carbon-pbc-frame34", CARBON_DATASET, carbon_frame),
        ("carbon-pbc-first64", CARBON_DATASET, carbon_first64),
        ("soap-diverse-300", SOAP_DATASET, soap),
        ("random-periodic-8x64", ROOT / "scripts/benchmarking/benchmark_cuda_nep_datasets.py", _random_periodic()),
    ]
    if include_full_carbon:
        carbon_all_frames = list(read(CARBON_DATASET, index=":"))
        carbon_all = StructureBatch.from_ase(
            carbon_all_frames,
            ids=[f"carbon-pbc-{index:03d}" for index in range(len(carbon_all_frames))],
        )
        cases.append(("carbon-pbc-all", CARBON_DATASET, carbon_all))
    return cases


def _measure(operation: Any, warmup: int, repeat: int) -> tuple[list[float], Any]:
    result: Any = None
    for _ in range(warmup):
        result = operation()
    samples: list[float] = []
    for _ in range(repeat):
        started = time.perf_counter()
        result = operation()
        samples.append(time.perf_counter() - started)
    return samples, result


def _accuracy(actual: Any, expected: Any) -> dict[str, Any]:
    actual_array = np.asarray(getattr(actual, "values", actual), dtype=np.float64)
    expected_array = np.asarray(getattr(expected, "values", expected), dtype=np.float64)
    if actual_array.shape != expected_array.shape:
        return {
            "pass": False,
            "shape": {"actual": list(actual_array.shape), "expected": list(expected_array.shape)},
        }
    difference = np.abs(actual_array - expected_array)
    tolerance = 1.0e-7 + 1.0e-6 * np.abs(expected_array)
    denominator = np.maximum(np.abs(expected_array), np.finfo(np.float64).tiny)
    return {
        "pass": bool(np.all(difference <= tolerance)),
        "shape": list(actual_array.shape),
        "max_abs_error": float(np.max(difference, initial=0.0)),
        "rmse": float(np.sqrt(np.mean(np.square(difference)))),
        "max_relative_error": float(np.max(difference / denominator, initial=0.0)),
        "max_tolerance_ratio": float(np.max(difference / tolerance, initial=0.0)),
        "rtol": 1.0e-6,
        "atol": 1.0e-7,
    }


def _timing(samples: list[float]) -> dict[str, float]:
    return {
        "median_seconds": float(np.median(np.asarray(samples, dtype=np.float64))),
        "p95_seconds": float(np.percentile(np.asarray(samples, dtype=np.float64), 95.0)),
    }


def _metadata(name: str, source: Path, batch: StructureBatch) -> dict[str, Any]:
    return {
        "name": name,
        "source": str(source),
        "source_sha256": _sha256(source) if source.is_file() else None,
        "frames": int(batch.offsets.size - 1),
        "atoms": int(batch.numbers.size),
        "species": sorted({int(value) for value in batch.numbers}),
        "pbc": batch.pbc.tolist(),
    }


def _run_case(name: str, source: Path, batch: StructureBatch, warmup: int, repeat: int) -> dict[str, Any]:
    print(f"running {name}: {batch.numbers.size} atoms / {batch.offsets.size - 1} frames", flush=True)
    cpu_results: dict[int, Any] = {}
    cpu_times: dict[str, dict[str, float]] = {}
    for threads in (1, 16):
        descriptor = NEP(model=NEP_MODEL, execution=ExecutionOptions(device="cpu", num_threads=threads))
        try:
            samples, result = _measure(
                lambda active_descriptor=descriptor: active_descriptor.compute(batch),
                warmup,
                repeat,
            )
        finally:
            descriptor.close()
        cpu_results[threads] = result
        cpu_times[str(threads)] = _timing(samples)

    gpu = NEP(model=NEP_MODEL, execution=ExecutionOptions(device="cuda"))
    try:
        gpu_samples, gpu_result = _measure(lambda: gpu.compute(batch), warmup, repeat)
    finally:
        gpu.close()

    row: dict[str, Any] = {
        "dataset": _metadata(name, source, batch),
        "cpu": {"threads_1": cpu_times["1"], "threads_16": cpu_times["16"]},
        "gpu": _timing(gpu_samples),
        "speedup_gpu_vs_cpu1": cpu_times["1"]["median_seconds"] / _timing(gpu_samples)["median_seconds"],
        "speedup_gpu_vs_cpu16": cpu_times["16"]["median_seconds"] / _timing(gpu_samples)["median_seconds"],
        "accuracy_gpu_vs_cpu1": _accuracy(gpu_result, cpu_results[1]),
        "accuracy_cpu16_vs_cpu1": _accuracy(cpu_results[16], cpu_results[1]),
    }

    if bool(np.all(batch.pbc == 1)):
        from nep_adapters import NEPCalculator

        reference = NEPCalculator(str(NEP_MODEL), backend="cuda")
        try:
            reference_samples, reference_values = _measure(
                lambda: reference.predict_descriptors(
                    [
                        Atoms(
                            numbers=batch.numbers[int(batch.offsets[index]) : int(batch.offsets[index + 1])],
                            positions=batch.positions[int(batch.offsets[index]) : int(batch.offsets[index + 1])],
                            cell=batch.cells[index],
                            pbc=batch.pbc[index].astype(bool),
                        )
                        for index in range(batch.offsets.size - 1)
                    ]
                ),
                warmup,
                repeat,
            )
        finally:
            reference.close()
        row["nepadapters"] = _timing(reference_samples)
        row["speedup_mdescriptor_vs_nepadapters"] = row["nepadapters"]["median_seconds"] / row["gpu"]["median_seconds"]
        row["accuracy_gpu_vs_nepadapters"] = _accuracy(gpu_result, reference_values)
        row["accuracy_cpu1_vs_nepadapters"] = _accuracy(cpu_results[1], reference_values)
    else:
        row["nepadapters"] = None
        row["speedup_mdescriptor_vs_nepadapters"] = None
        row["accuracy_gpu_vs_nepadapters"] = None
        row["accuracy_cpu1_vs_nepadapters"] = None
    return row


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument(
        "--carbon-all",
        action="store_true",
        help="also benchmark all frames in carbon_dataset_pbc.xyz",
    )
    args = parser.parse_args(argv)
    if args.warmup < 0 or args.repeat <= 0:
        raise SystemExit("warmup must be non-negative and repeat must be positive")
    load_cuda_plugin(ROOT / "build-cuda")
    result = {
        "schema_version": 1,
        "package": "MDescriptor",
        "model": str(NEP_MODEL.resolve()),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "warmup": args.warmup,
        "repeat": args.repeat,
        "tolerance": {"rtol": 1.0e-6, "atol": 1.0e-7},
        "measurements": [
            _run_case(name, source, batch, args.warmup, args.repeat)
            for name, source, batch in _cases(args.carbon_all)
        ],
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
