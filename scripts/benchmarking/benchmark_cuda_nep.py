"""Compare the CUDA NEP descriptor with NEPAdapters.

The benchmark measures the public ``compute()``/``predict_descriptors()`` call
after warmup, so model upload and one-time CUDA context creation do not hide
the steady-state cost.  It is an explicit gate: the command exits non-zero if
the requested tolerance or minimum speedup is not met.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms

from mdescriptor import ExecutionOptions, MDescriptorError, StructureBatch
from mdescriptor._cuda_loader import load_cuda_plugin
from mdescriptor.descriptors import NEP
from mdescriptor.models import NEP_MODEL

ROOT = Path(__file__).resolve().parents[2]


def _batch(
    structures: int,
    atoms_per_structure: int,
) -> tuple[list[Atoms], StructureBatch]:
    rng = np.random.default_rng(20260831)
    # Scale the cell with N so the benchmark keeps a roughly constant density
    # and neighbor count as --atoms changes.
    cell_length = max(12.0, float(atoms_per_structure) ** (1.0 / 3.0) * 4.0)
    cell = np.diag([cell_length, cell_length, cell_length])
    species = np.asarray([1, 6, 8, 14], dtype=np.int32)
    systems: list[Atoms] = []
    for _index in range(structures):
        positions = rng.random((atoms_per_structure, 3)) * cell_length
        numbers = np.resize(species, atoms_per_structure)
        systems.append(Atoms(numbers=numbers, positions=positions, cell=cell, pbc=True))
    batch = StructureBatch.from_ase(
        systems,
        ids=[f"nep-benchmark-{index}" for index in range(structures)],
    )
    return systems, batch


def _measure(
    operation: Callable[[], Any],
    warmup: int,
    repeat: int,
) -> tuple[list[float], Any]:
    result: Any = None
    for _ in range(warmup):
        result = operation()
    elapsed: list[float] = []
    for _ in range(repeat):
        started = time.perf_counter()
        result = operation()
        elapsed.append(time.perf_counter() - started)
    if result is None:  # pragma: no cover - repeat is validated by main
        raise RuntimeError("benchmark produced no result")
    return elapsed, result


def _percentile(values: list[float], percentile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def _accuracy(actual: Any, expected: Any) -> dict[str, Any]:
    actual_array = np.asarray(actual, dtype=np.float64)
    expected_array = np.asarray(expected, dtype=np.float64)
    if actual_array.shape != expected_array.shape:
        return {
            "pass": False,
            "shape": {"mdescriptor": list(actual_array.shape), "nepadapters": list(expected_array.shape)},
            "max_abs_error": None,
            "max_relative_error": None,
            "max_tolerance_ratio": None,
        }
    difference = np.abs(actual_array - expected_array)
    tolerance = 1.0e-7 + 1.0e-6 * np.abs(expected_array)
    denominator = np.maximum(np.abs(expected_array), np.finfo(np.float64).tiny)
    return {
        "pass": bool(np.all(difference <= tolerance)),
        "shape": list(actual_array.shape),
        "max_abs_error": float(np.max(difference, initial=0.0)),
        "max_relative_error": float(np.max(difference / denominator, initial=0.0)),
        "max_tolerance_ratio": float(np.max(difference / tolerance, initial=0.0)),
        "rtol": 1.0e-6,
        "atol": 1.0e-7,
    }


def _write_or_print(value: dict[str, Any], output: Path | None) -> None:
    encoded = json.dumps(value, indent=2, sort_keys=True)
    if output is None:
        print(encoded)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(encoded + "\n", encoding="utf-8")
    print(f"wrote {output}")
    print(encoded)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=NEP_MODEL)
    parser.add_argument("--structures", type=int, default=8)
    parser.add_argument("--atoms", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=7)
    parser.add_argument(
        "--min-speedup",
        type=float,
        default=1.0,
        help="minimum NEPAdapters/MDescriptor median runtime ratio (default: 1.0)",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.structures <= 0 or args.atoms <= 0 or args.warmup < 0 or args.repeat <= 0:
        raise SystemExit("structures and atoms must be positive; warmup must be non-negative; repeat must be positive")
    if not np.isfinite(args.min_speedup) or args.min_speedup < 0.0:
        raise SystemExit("min-speedup must be finite and non-negative")

    base = {
        "schema_version": 1,
        "package": "MDescriptor",
        "model": str(args.model.resolve()),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "warmup": args.warmup,
        "repeat": args.repeat,
        "structures": args.structures,
        "atoms_per_structure": args.atoms,
        "minimum_speedup": args.min_speedup,
    }
    try:
        load_cuda_plugin(ROOT / "build-cuda")
    except Exception as error:
        _write_or_print(
            {**base, "status": "unavailable", "reason": f"CUDA plugin: {error}"},
            args.output,
        )
        return 2

    systems, batch = _batch(args.structures, args.atoms)
    descriptor: NEP | None = None
    reference: Any = None
    try:
        descriptor = NEP(
            model=args.model,
            execution=ExecutionOptions(device="cuda"),
        )
        try:
            from nep_adapters import NEPCalculator

            reference = NEPCalculator(str(args.model), backend="cuda")
        except Exception as error:
            _write_or_print(
                {**base, "status": "unavailable", "reason": f"NEPAdapters CUDA: {error}"},
                args.output,
            )
            return 2

        try:
            mdescriptor_times, mdescriptor_result = _measure(
                lambda: descriptor.compute(batch), args.warmup, args.repeat
            )
            reference_times, reference_values = _measure(
                lambda: reference.predict_descriptors(systems), args.warmup, args.repeat
            )
        except MDescriptorError as error:
            if error.code == "device_unavailable":
                _write_or_print(
                    {**base, "status": "unavailable", "reason": f"MDescriptor CUDA: {error}"},
                    args.output,
                )
                return 2
            raise

        accuracy = _accuracy(mdescriptor_result.values, reference_values)
        mdescriptor_median = float(np.median(np.asarray(mdescriptor_times)))
        reference_median = float(np.median(np.asarray(reference_times)))
        speedup = reference_median / mdescriptor_median
        speed = {
            "pass": bool(speedup >= args.min_speedup),
            "mdescriptor_median_seconds": mdescriptor_median,
            "mdescriptor_p95_seconds": _percentile(mdescriptor_times, 95.0),
            "nepadapters_median_seconds": reference_median,
            "nepadapters_p95_seconds": _percentile(reference_times, 95.0),
            "speedup": speedup,
        }
        result = {
            **base,
            "status": "pass" if accuracy["pass"] and speed["pass"] else "fail",
            "feature_count": int(mdescriptor_result.feature_count),
            "accuracy": accuracy,
            "speed": speed,
        }
        _write_or_print(result, args.output)
        return 0 if result["status"] == "pass" else 1
    finally:
        if reference is not None:
            reference.close()
        if descriptor is not None:
            descriptor.close()


if __name__ == "__main__":
    raise SystemExit(main())
