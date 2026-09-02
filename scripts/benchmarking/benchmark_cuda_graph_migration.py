"""Compare the pre- and post-migration CUDA graph paths.

Build the parent revision and the working tree into separate CUDA plugin
directories, then run this script with both directories.  Each child process
loads exactly one ``_cuda`` extension, which avoids Python module caching and
makes the before/after comparison reproducible.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms

import mdescriptor
from mdescriptor import ExecutionOptions, MDescriptorError, StructureBatch
from mdescriptor.descriptors import (
    NeighborList,
    SoapPowerSpectrum,
    SoapRadialSpectrum,
    SphericalExpansion,
    SphericalExpansionByPair,
)

ROOT = Path(__file__).resolve().parents[2]

DESCRIPTORS: dict[str, tuple[type[Any], dict[str, Any]]] = {
    "NeighborList": (NeighborList, {"cutoff": 3.5}),
    "SphericalExpansion": (
        SphericalExpansion,
        {
            "species": [1, 6, 8],
            "cutoff": 3.5,
            "density_width": 0.6,
            "max_radial": 2,
            "max_angular": 2,
        },
    ),
    "SoapRadialSpectrum": (
        SoapRadialSpectrum,
        {
            "species": [1, 6, 8],
            "cutoff": 3.5,
            "density_width": 0.6,
            "max_radial": 2,
            "max_angular": 2,
        },
    ),
    "SoapPowerSpectrum": (
        SoapPowerSpectrum,
        {
            "species": [1, 6, 8],
            "cutoff": 3.5,
            "density_width": 0.6,
            "max_radial": 2,
            "max_angular": 2,
        },
    ),
    "SphericalExpansionByPair": (
        SphericalExpansionByPair,
        {
            "species": [1, 6, 8],
            "cutoff": 3.5,
            "density_width": 0.6,
            "max_radial": 2,
            "max_angular": 2,
        },
    ),
}


def _batch(periodic_repeats: int, isolated_repeats: int) -> StructureBatch:
    periodic_base = np.asarray(
        [
            [0.15, 0.25, 0.35],
            [1.35, 0.35, 0.25],
            [3.7, 2.1, 1.4],
        ],
        dtype=np.float64,
    )
    periodic_offsets = np.asarray(
        [
            [index % 4 * 0.3, index // 4 * 0.25, index % 2 * 0.15]
            for index in range(periodic_repeats)
        ],
        dtype=np.float64,
    )
    return StructureBatch.from_ase(
        [
            Atoms(
                numbers=[1, 8, 6] * periodic_repeats,
                positions=(periodic_base[None, :, :] + periodic_offsets[:, None, :]).reshape(-1, 3),
                cell=[
                    [5.0, 0.0, 0.0],
                    [0.35, 4.8, 0.0],
                    [0.2, 0.25, 5.4],
                ],
                pbc=True,
            ),
            Atoms(
                numbers=[1, 1, 8] * isolated_repeats,
                positions=np.asarray(
                    [
                        [-0.3, 0.0, 0.1],
                        [1.15, 0.2, -0.15],
                        [0.4, 1.3, 0.0],
                    ]
                    * isolated_repeats,
                    dtype=np.float64,
                ),
            ),
        ],
        ids=["periodic", "isolated"],
    )


def _load_cuda_plugin(plugin_dir: Path) -> None:
    if not any(plugin_dir.glob("_cuda*.so")):
        raise RuntimeError(f"CUDA plugin directory has no _cuda*.so: {plugin_dir}")
    plugin_text = str(plugin_dir.resolve())
    os.environ["MDESCRIPTOR_CUDA_PLUGIN_DIR"] = plugin_text
    if plugin_text not in mdescriptor.__path__:
        mdescriptor.__path__.insert(0, plugin_text)
    importlib.import_module("mdescriptor._cuda")


def _measure(operation: Any, warmup: int, repeat: int) -> tuple[float, list[float], Any]:
    started = time.perf_counter()
    result = operation()
    first = time.perf_counter() - started
    for _ in range(warmup):
        result = operation()
    samples: list[float] = []
    for _ in range(repeat):
        started = time.perf_counter()
        result = operation()
        samples.append(time.perf_counter() - started)
    return float(first), samples, result


def _accuracy(actual: Any, expected: Any) -> dict[str, Any]:
    actual_values = np.asarray(actual.values, dtype=np.float64)
    expected_values = np.asarray(expected.values, dtype=np.float64)
    if actual_values.shape != expected_values.shape:
        return {
            "allclose": False,
            "shape": {"cuda": list(actual_values.shape), "cpu": list(expected_values.shape)},
            "max_abs_error": None,
            "max_relative_error": None,
            "samples_equal": False,
            "row_offsets_equal": False,
        }
    difference = np.abs(actual_values - expected_values)
    denominator = np.maximum(np.abs(expected_values), 1.0e-12)
    tolerance = 1.0e-10 + 1.0e-8 * np.abs(expected_values)
    return {
        "allclose": bool(np.all(difference <= tolerance)),
        "shape": list(actual_values.shape),
        "max_abs_error": float(np.max(difference, initial=0.0)),
        "mae": float(np.mean(difference)),
        "rmse": float(np.sqrt(np.mean(np.square(difference)))),
        "max_relative_error": float(np.max(difference / denominator, initial=0.0)),
        "samples_equal": bool(np.array_equal(actual.samples, expected.samples)),
        "row_offsets_equal": bool(np.array_equal(actual.row_offsets, expected.row_offsets)),
        "rtol": 1.0e-8,
        "atol": 1.0e-10,
    }


def _single(args: argparse.Namespace) -> int:
    base: dict[str, Any] = {
        "schema_version": 1,
        "label": args.label,
        "plugin_dir": str(args.plugin_dir.resolve()),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "warmup": args.warmup,
        "repeat": args.repeat,
        "periodic_repeats": args.periodic_repeats,
        "isolated_repeats": args.isolated_repeats,
    }
    try:
        _load_cuda_plugin(args.plugin_dir)
    except Exception as error:
        print(json.dumps({**base, "status": "unavailable", "reason": str(error)}))
        return 2

    batch = _batch(args.periodic_repeats, args.isolated_repeats)
    measurements: list[dict[str, Any]] = []
    for name, (descriptor_type, parameters) in DESCRIPTORS.items():
        cpu = descriptor_type(
            **parameters,
            execution=ExecutionOptions(device="cpu", num_threads=1),
        )
        gpu = descriptor_type(
            **parameters,
            execution=ExecutionOptions(device="cuda"),
        )
        try:
            cpu_first_started = time.perf_counter()
            expected = cpu.compute(batch)
            cpu_first = time.perf_counter() - cpu_first_started
            def cpu_operation(descriptor=cpu):
                return descriptor.compute(batch)

            _, cpu_samples, _ = _measure(cpu_operation, args.warmup, args.repeat)
            try:
                def gpu_operation(descriptor=gpu):
                    return descriptor.compute(batch)

                gpu_first, gpu_samples, actual = _measure(
                    gpu_operation, args.warmup, args.repeat
                )
            except MDescriptorError as error:
                if error.code == "device_unavailable":
                    print(
                        json.dumps(
                            {
                                **base,
                                "status": "unavailable",
                                "reason": f"{name}: {error}",
                            }
                        )
                    )
                    return 2
                raise
            measurements.append(
                {
                    "descriptor": name,
                    "feature_count": int(actual.feature_count),
                    "cpu": {
                        "first_seconds": cpu_first,
                        "steady_median_seconds": float(np.median(cpu_samples)),
                        "samples_seconds": cpu_samples,
                    },
                    "cuda": {
                        "first_seconds": gpu_first,
                        "steady_median_seconds": float(np.median(gpu_samples)),
                        "samples_seconds": gpu_samples,
                    },
                    "speedup_vs_cpu_steady": float(
                        np.median(cpu_samples) / np.median(gpu_samples)
                    ),
                    "accuracy": _accuracy(actual, expected),
                }
            )
        finally:
            cpu.close()
            gpu.close()
    print(json.dumps({**base, "status": "ok", "measurements": measurements}))
    return 0


def _comparison(args: argparse.Namespace) -> int:
    records: dict[str, dict[str, Any]] = {}
    for label, plugin_dir in (("before", args.before_plugin_dir), ("after", args.after_plugin_dir)):
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--single",
            "--label",
            label,
            "--plugin-dir",
            str(plugin_dir),
            "--warmup",
            str(args.warmup),
            "--repeat",
            str(args.repeat),
            "--periodic-repeats",
            str(args.periodic_repeats),
            "--isolated-repeats",
            str(args.isolated_repeats),
        ]
        completed = subprocess.run(
            command,
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.stdout.strip():
            records[label] = json.loads(completed.stdout.strip().splitlines()[-1])
        else:
            records[label] = {
                "status": "unavailable",
                "label": label,
                "reason": completed.stderr.strip() or f"child exited {completed.returncode}",
            }

    comparison: list[dict[str, Any]] = []
    before = records["before"]
    after = records["after"]
    if before.get("status") == "ok" and after.get("status") == "ok":
        before_by_name = {item["descriptor"]: item for item in before["measurements"]}
        after_by_name = {item["descriptor"]: item for item in after["measurements"]}
        for name in DESCRIPTORS:
            before_item = before_by_name[name]
            after_item = after_by_name[name]
            before_time = before_item["cuda"]["steady_median_seconds"]
            after_time = after_item["cuda"]["steady_median_seconds"]
            comparison.append(
                {
                    "descriptor": name,
                    "before_cuda_median_seconds": before_time,
                    "after_cuda_median_seconds": after_time,
                    "speedup_after_vs_before": float(before_time / after_time),
                    "before_accuracy": before_item["accuracy"],
                    "after_accuracy": after_item["accuracy"],
                }
            )
    result = {
        "schema_version": 1,
        "package": "MDescriptor",
        "measurement": {
            "periodic_repeats": args.periodic_repeats,
            "isolated_repeats": args.isolated_repeats,
            "workload": "mixed triclinic periodic plus isolated batch",
            "steady_state": "median of synchronous compute calls after warmup",
            "accuracy_reference": "CPU descriptor with num_threads=1",
        },
        "before": before,
        "after": after,
        "comparison": comparison,
    }
    encoded = json.dumps(result, indent=2, sort_keys=True)
    if args.output is None:
        print(encoded)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
        print(f"wrote {args.output}")
    return 0 if before.get("status") == "ok" and after.get("status") == "ok" else 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--single", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--label", default="current", help=argparse.SUPPRESS)
    parser.add_argument("--plugin-dir", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--before-plugin-dir", type=Path)
    parser.add_argument("--after-plugin-dir", type=Path)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument(
        "--periodic-repeats",
        type=int,
        default=8,
        help="number of three-atom motifs in the periodic structure",
    )
    parser.add_argument(
        "--isolated-repeats",
        type=int,
        default=4,
        help="number of three-atom motifs in the isolated structure",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.warmup < 0 or args.repeat <= 0:
        raise SystemExit("warmup must be non-negative and repeat must be positive")
    if args.single:
        if args.plugin_dir is None:
            raise SystemExit("--plugin-dir is required with --single")
        if args.periodic_repeats <= 0 or args.isolated_repeats <= 0:
            raise SystemExit("repeat counts must be positive")
        return _single(args)
    if args.before_plugin_dir is None or args.after_plugin_dir is None:
        raise SystemExit("--before-plugin-dir and --after-plugin-dir are required")
    if args.periodic_repeats <= 0 or args.isolated_repeats <= 0:
        raise SystemExit("repeat counts must be positive")
    return _comparison(args)


if __name__ == "__main__":
    raise SystemExit(main())
