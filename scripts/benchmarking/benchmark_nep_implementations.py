"""Compare the project NEP paths with NEPAdapters and official GPUMD NEP.

The benchmark uses one model and one ASE structure batch for every backend.
MDescriptor and NEPAdapters are measured through their public in-process
descriptor calls.  GPUMD's standalone ``nep`` executable is measured through
its reported prediction time and its official ``descriptor.out`` output.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import re
import shutil
import subprocess
import tempfile
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.io import read, write

import mdescriptor
from mdescriptor import ExecutionOptions, StructureBatch
from mdescriptor.descriptors import NEP
from mdescriptor.models import NEP_MODEL

ROOT = Path(__file__).resolve().parents[2]
CARBON_DATASET = ROOT / "benchmarks/_datasets/legacy/carbon_dataset_pbc.xyz"
TOLERANCE = {"rtol": 1.0e-6, "atol": 1.0e-7}
PREDICTION_RE = re.compile(r"Time used for predicting = ([0-9.eE+-]+) s\.")


def _load_cuda_plugin() -> None:
    configured = os.environ.get("MDESCRIPTOR_CUDA_PLUGIN_DIR")
    candidates = [Path(configured)] if configured else []
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


def _carbon_batch(frame_count: int | None) -> tuple[list[Atoms], StructureBatch]:
    structures = list(read(CARBON_DATASET, index=":"))
    if frame_count is not None:
        structures = structures[:frame_count]
    if not structures:
        raise ValueError("carbon dataset selection is empty")
    return structures, StructureBatch.from_ase(
        structures,
        ids=[f"carbon-pbc-{index:03d}" for index in range(len(structures))],
    )


def _measure(
    operation: Callable[[], Any],
    warmup: int,
    repeat: int,
) -> tuple[list[float], Any]:
    result: Any = None
    for _ in range(warmup):
        result = operation()
    samples: list[float] = []
    for _ in range(repeat):
        started = time.perf_counter()
        result = operation()
        samples.append(time.perf_counter() - started)
    return samples, result


def _timing(samples: Sequence[float]) -> dict[str, float]:
    values = np.asarray(samples, dtype=np.float64)
    return {
        "median_seconds": float(np.median(values)),
        "p95_seconds": float(np.percentile(values, 95.0)),
    }


def _values(value: Any) -> np.ndarray:
    return np.asarray(getattr(value, "values", value), dtype=np.float64)


def _accuracy(actual: Any, expected: Any) -> dict[str, Any]:
    actual_array = _values(actual)
    expected_array = _values(expected)
    if actual_array.shape != expected_array.shape:
        return {
            "pass": False,
            "shape": {"actual": list(actual_array.shape), "expected": list(expected_array.shape)},
        }
    difference = np.abs(actual_array - expected_array)
    tolerance = TOLERANCE["atol"] + TOLERANCE["rtol"] * np.abs(expected_array)
    denominator = np.maximum(np.abs(expected_array), np.finfo(np.float64).tiny)
    return {
        "pass": bool(np.all(difference <= tolerance)),
        "shape": list(actual_array.shape),
        "max_abs_error": float(np.max(difference, initial=0.0)),
        "rmse": float(np.sqrt(np.mean(np.square(difference)))),
        "max_relative_error": float(np.max(difference / denominator, initial=0.0)),
        "max_tolerance_ratio": float(np.max(difference / tolerance, initial=0.0)),
        **TOLERANCE,
    }


def _round_like_gpumd_descriptor_output(value: Any, significant_digits: int) -> np.ndarray:
    """Apply GPUMD's descriptor output significant-digit format."""

    array = _values(value)
    formatter = np.vectorize(
        lambda item: float(f"{item:.{significant_digits}g}"), otypes=[np.float64]
    )
    return formatter(array)


def _run_in_process_backends(
    structures: list[Atoms],
    batch: StructureBatch,
    warmup: int,
    repeat: int,
) -> dict[str, Any]:
    project_results: dict[str, Any] = {}
    timings: dict[str, dict[str, float]] = {}

    for threads in (1, 16):
        descriptor = NEP(
            model=NEP_MODEL,
            execution=ExecutionOptions(device="cpu", num_threads=threads),
        )
        try:
            samples, result = _measure(
                lambda active_descriptor=descriptor: active_descriptor.compute(batch),
                warmup,
                repeat,
            )
        finally:
            descriptor.close()
        project_results[f"mdescriptor_cpu{threads}"] = result
        timings[f"mdescriptor_cpu{threads}"] = _timing(samples)

    project_gpu = NEP(model=NEP_MODEL, execution=ExecutionOptions(device="cuda"))
    try:
        samples, result = _measure(lambda: project_gpu.compute(batch), warmup, repeat)
    finally:
        project_gpu.close()
    project_results["mdescriptor_gpu"] = result
    timings["mdescriptor_gpu"] = _timing(samples)

    from nep_adapters import NEPCalculator

    for backend, label in (("cpu", "cpu"), ("cuda", "gpu")):
        reference = NEPCalculator(str(NEP_MODEL), backend=backend)
        try:
            samples, result = _measure(
                lambda active_reference=reference: active_reference.predict_descriptors(structures),
                warmup,
                repeat,
            )
        finally:
            reference.close()
        project_results[f"nepadapters_{label}"] = result
        timings[f"nepadapters_{label}"] = _timing(samples)

    return {"results": project_results, "timings": timings}


def _gpumd_nep_input(output_descriptor: int) -> str:
    header = NEP_MODEL.read_text(encoding="utf-8").splitlines()
    model_tokens = header[0].split()
    species = model_tokens[2:]
    return "\n".join(
        [
            "type " + str(len(species)) + " " + " ".join(species),
            "version 4",
            "zbl 2",
            "cutoff 6 5",
            "n_max 4 4",
            "basis_size 8 8",
            "l_max 4 2 1",
            "neuron 80",
            "prediction 1",
            f"output_descriptor {output_descriptor}",
            "",
        ]
    )


def _run_gpumd_once(
    gpumd_nep: Path,
    structures: list[Atoms],
    output_descriptor: int,
) -> tuple[float, np.ndarray | None]:
    with tempfile.TemporaryDirectory(prefix="mdescriptor-gpumd-") as temporary:
        workdir = Path(temporary)
        shutil.copy2(NEP_MODEL, workdir / "nep.txt")
        (workdir / "nep.in").write_text(
            _gpumd_nep_input(output_descriptor), encoding="utf-8"
        )
        if len(structures) == 450:
            (workdir / "train.xyz").symlink_to(CARBON_DATASET)
        else:
            write(workdir / "train.xyz", structures, format="extxyz")
        completed = subprocess.run(
            [str(gpumd_nep.resolve())],
            cwd=workdir,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"GPUMD nep failed with exit code {completed.returncode}:\n"
                + (completed.stdout + "\n" + completed.stderr)[-8000:]
            )
        match = PREDICTION_RE.search(completed.stdout)
        if match is None:
            raise RuntimeError(
                "GPUMD nep did not report prediction time:\n" + completed.stdout[-4000:]
            )
        values = None
        if output_descriptor:
            values = np.loadtxt(workdir / "descriptor.out", dtype=np.float64)
            values = np.atleast_2d(values)
        return float(match.group(1)), values


def _run_gpumd(
    gpumd_nep: Path | None,
    structures: list[Atoms],
    warmup: int,
    repeat: int,
    output_significant_digits: int,
) -> dict[str, Any] | None:
    if gpumd_nep is None:
        return None
    if not gpumd_nep.is_file():
        raise FileNotFoundError(gpumd_nep)
    for _ in range(warmup):
        _run_gpumd_once(gpumd_nep, structures, 0)
    samples = [_run_gpumd_once(gpumd_nep, structures, 0)[0] for _ in range(repeat)]
    accuracy_time, values = _run_gpumd_once(gpumd_nep, structures, 2)
    return {
        "timing": _timing(samples),
        "reported_prediction_time_seconds": accuracy_time,
        "descriptor_values": values,
        "descriptor_output": f"GPUMD descriptor.out (%.{output_significant_digits}g text output)",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpumd-nep", type=Path, required=True)
    parser.add_argument("--frames", type=int, help="number of initial carbon frames; default is all")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--gpumd-output-significant-digits", type=int, default=6)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.frames is not None and args.frames <= 0:
        raise SystemExit("frames must be positive")
    if args.warmup < 0 or args.repeat <= 0:
        raise SystemExit("warmup must be non-negative and repeat must be positive")
    if args.gpumd_output_significant_digits <= 0:
        raise SystemExit("gpumd-output-significant-digits must be positive")

    _load_cuda_plugin()
    structures, batch = _carbon_batch(args.frames)
    print(
        f"running carbon: {batch.numbers.size} atoms / {batch.offsets.size - 1} frames",
        flush=True,
    )
    in_process = _run_in_process_backends(structures, batch, args.warmup, args.repeat)
    gpumd = _run_gpumd(
        args.gpumd_nep,
        structures,
        args.warmup,
        args.repeat,
        args.gpumd_output_significant_digits,
    )

    results = in_process["results"]
    timings = in_process["timings"]
    accuracy: dict[str, Any] = {}
    for left, right in (
        ("mdescriptor_cpu1", "nepadapters_cpu"),
        ("mdescriptor_cpu16", "nepadapters_cpu"),
        ("mdescriptor_gpu", "nepadapters_gpu"),
        ("mdescriptor_gpu", "mdescriptor_cpu1"),
        ("nepadapters_gpu", "nepadapters_cpu"),
    ):
        accuracy[f"{left}_vs_{right}"] = _accuracy(results[left], results[right])
    speedup = {
        "mdescriptor_cpu1_vs_nepadapters_cpu": timings["nepadapters_cpu"]["median_seconds"]
        / timings["mdescriptor_cpu1"]["median_seconds"],
        "mdescriptor_cpu16_vs_nepadapters_cpu": timings["nepadapters_cpu"]["median_seconds"]
        / timings["mdescriptor_cpu16"]["median_seconds"],
        "mdescriptor_gpu_vs_nepadapters_gpu": timings["nepadapters_gpu"]["median_seconds"]
        / timings["mdescriptor_gpu"]["median_seconds"],
        "mdescriptor_gpu_vs_mdescriptor_cpu1": timings["mdescriptor_cpu1"]["median_seconds"]
        / timings["mdescriptor_gpu"]["median_seconds"],
    }
    if gpumd is not None:
        accuracy["mdescriptor_cpu1_vs_gpumd_nep"] = _accuracy(
            results["mdescriptor_cpu1"], gpumd["descriptor_values"]
        )
        accuracy["mdescriptor_gpu_vs_gpumd_nep"] = _accuracy(
            results["mdescriptor_gpu"], gpumd["descriptor_values"]
        )
        accuracy["nepadapters_cpu_vs_gpumd_nep"] = _accuracy(
            results["nepadapters_cpu"], gpumd["descriptor_values"]
        )
        accuracy["nepadapters_gpu_vs_gpumd_nep"] = _accuracy(
            results["nepadapters_gpu"], gpumd["descriptor_values"]
        )
        accuracy["mdescriptor_gpu_quantized_vs_gpumd_nep"] = _accuracy(
            _round_like_gpumd_descriptor_output(
                results["mdescriptor_gpu"], args.gpumd_output_significant_digits
            ),
            gpumd["descriptor_values"],
        )
        accuracy["nepadapters_gpu_quantized_vs_gpumd_nep"] = _accuracy(
            _round_like_gpumd_descriptor_output(
                results["nepadapters_gpu"], args.gpumd_output_significant_digits
            ),
            gpumd["descriptor_values"],
        )
        timings["gpumd_nep"] = gpumd["timing"]
        speedup["mdescriptor_gpu_vs_gpumd_nep"] = (
            gpumd["timing"]["median_seconds"] / timings["mdescriptor_gpu"]["median_seconds"]
        )

    output = {
        "schema_version": 1,
        "package": "MDescriptor",
        "model": str(NEP_MODEL.resolve()),
        "dataset": {
            "source": str(CARBON_DATASET.resolve()),
            "frames": int(batch.offsets.size - 1),
            "atoms": int(batch.numbers.size),
            "feature_count": 35,
        },
        "python": platform.python_version(),
        "platform": platform.platform(),
        "warmup": args.warmup,
        "repeat": args.repeat,
        "tolerance": TOLERANCE,
        "timings": timings,
        "speedup": speedup,
        "accuracy": accuracy,
        "gpumd_nep": None
        if gpumd is None
        else {
            "reported_prediction_time_seconds": gpumd["reported_prediction_time_seconds"],
            "descriptor_output": gpumd["descriptor_output"],
            "executable": str(args.gpumd_nep.resolve()),
        },
    }
    encoded = json.dumps(output, indent=2, sort_keys=True)
    if args.output is None:
        print(encoded)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
