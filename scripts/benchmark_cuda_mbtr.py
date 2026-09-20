"""Measure the CUDA MBTR family with an exact native/plugin build.

The script runs each build in a fresh child process so an editable install or a
previously imported extension cannot silently provide the measured code.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

_CHILD = r'''
from __future__ import annotations

import hashlib
import importlib
import importlib.machinery
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np


def _load_extensions() -> tuple[Path, Path]:
    root = Path(os.environ["MDESCRIPTOR_BUILD"]).resolve()
    suffixes = importlib.machinery.EXTENSION_SUFFIXES
    native_path = next(
        (root / f"_native{suffix}" for suffix in suffixes
         if (root / f"_native{suffix}").is_file()),
        None,
    )
    cuda_path = next(
        (root / f"_cuda{suffix}" for suffix in suffixes
         if (root / f"_cuda{suffix}").is_file()),
        None,
    )
    if native_path is None or cuda_path is None:
        raise ImportError(f"build {root} lacks exact _native/_cuda extensions")

    import mdescriptor

    spec = importlib.util.spec_from_file_location("mdescriptor._native", native_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load native extension from {native_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["mdescriptor._native"] = module
    spec.loader.exec_module(module)
    if Path(module.__file__).resolve() != native_path:
        raise ImportError(f"native extension loaded from {module.__file__}")
    mdescriptor.__dict__["_native"] = module

    from mdescriptor._cuda_loader import load_cuda_plugin

    load_cuda_plugin(root)
    cuda = importlib.import_module("mdescriptor._cuda")
    if Path(cuda.__file__).resolve() != cuda_path:
        raise ImportError(f"CUDA extension loaded from {cuda.__file__}")
    return native_path, cuda_path


def _batch():
    from mdescriptor import StructureBatch

    structures = 8
    atoms_per_structure = 192
    rng = np.random.default_rng(20260920)
    cell = np.eye(3, dtype=np.float64) * 18.0
    numbers = np.resize(np.asarray([1, 6, 8, 14], dtype=np.int32), atoms_per_structure)
    positions = np.concatenate(
        [rng.random((atoms_per_structure, 3)) * 18.0 for _ in range(structures)]
    )
    return StructureBatch(
        np.tile(numbers, structures),
        positions,
        np.tile(cell, (structures, 1, 1)),
        np.ones((structures, 3), dtype=np.int32),
        np.arange(structures + 1, dtype=np.int64) * atoms_per_structure,
        tuple(f"mbtr-{index}" for index in range(structures)),
    )


def _cases():
    from mdescriptor import ExecutionOptions
    from mdescriptor.descriptors import LMBTR, MBTR, ValleOganov

    serial = ExecutionOptions(device="cpu", num_threads=1)
    cuda = ExecutionOptions(device="cuda")
    common_distance = {
        "species": [1, 6, 8, 14],
        "geometry": {"function": "distance"},
        "grid": {"min": 0.0, "max": 6.0, "n": 24, "sigma": 0.2},
        "weighting": {"function": "smooth_cutoff", "r_cut": 4.0, "sharpness": 2.0},
        "normalization": "none",
    }
    common_angle = {
        "species": [1, 6, 8, 14],
        "geometry": {"function": "angle"},
        "grid": {"min": 0.0, "max": 180.0, "n": 24, "sigma": 0.2},
        "weighting": {"function": "smooth_cutoff", "r_cut": 4.0, "sharpness": 2.0},
        "normalization": "none",
    }
    return [
        ("MBTR-distance", MBTR, common_distance),
        ("MBTR-angle", MBTR, common_angle),
        ("LMBTR-distance", LMBTR, common_distance),
        ("LMBTR-angle", LMBTR, common_angle),
        ("ValleOganov-distance", ValleOganov, {
            "species": [1, 6, 8, 14], "function": "distance", "n": 24,
            "sigma": 0.2, "r_cut": 4.0,
        }),
        ("ValleOganov-angle", ValleOganov, {
            "species": [1, 6, 8, 14], "function": "angle", "n": 24,
            "sigma": 0.2, "r_cut": 4.0,
        }),
    ], serial, cuda


def _measure(descriptor, batch, warmup: int, repeat: int):
    for _ in range(warmup):
        descriptor.compute(batch)
    samples = []
    digests = []
    result = None
    for _ in range(repeat):
        started = time.perf_counter()
        result = descriptor.compute(batch)
        samples.append((time.perf_counter() - started) * 1e3)
        digests.append(hashlib.sha256(np.asarray(result.values).tobytes()).hexdigest())
    assert result is not None
    return (
        float(np.median(samples)),
        np.asarray(result.values, dtype=np.float64),
        samples,
        len(set(digests)) == 1,
    )


def main() -> None:
    native_path, cuda_path = _load_extensions()
    batch = _batch()
    cases, serial_options, cuda_options = _cases()
    warmup = int(os.environ["MDESCRIPTOR_BENCH_WARMUP"])
    repeat = int(os.environ["MDESCRIPTOR_BENCH_REPEAT"])
    records = []
    for name, descriptor_type, parameters in cases:
        cpu = descriptor_type(**parameters, execution=serial_options)
        gpu = descriptor_type(**parameters, execution=cuda_options)
        try:
            cpu_ms, cpu_values, cpu_samples, cpu_stable = _measure(
                cpu, batch, warmup, repeat
            )
            gpu_ms, gpu_values, gpu_samples, gpu_stable = _measure(
                gpu, batch, warmup, repeat
            )
        finally:
            cpu.close()
            gpu.close()
        delta = np.abs(gpu_values - cpu_values)
        records.append({
            "name": name,
            "rows": int(gpu_values.shape[0]),
            "features": int(gpu_values.shape[1]),
            "cpu_median_ms": cpu_ms,
            "gpu_median_ms": gpu_ms,
            "gpu_over_cpu": gpu_ms / cpu_ms,
            "cpu_samples_ms": cpu_samples,
            "gpu_samples_ms": gpu_samples,
            "max_abs_cpu_gpu": float(np.max(delta, initial=0.0)),
            "finite": bool(np.isfinite(gpu_values).all()),
            "cpu_repeat_stable": cpu_stable,
            "gpu_repeat_stable": gpu_stable,
            "sha256": hashlib.sha256(gpu_values.tobytes()).hexdigest(),
        })
    print("MBTR_BUILD=" + json.dumps({
        "native": str(native_path), "cuda": str(cuda_path),
        "warmup": warmup, "repeat": repeat, "records": records,
    }, sort_keys=True))


main()
'''


def _run(build: Path, warmup: int, repeat: int) -> dict[str, object]:
    environment = os.environ.copy()
    build = build.expanduser().resolve()
    environment.update({
        "MDESCRIPTOR_BUILD": str(build),
        "MDESCRIPTOR_CUDA_PLUGIN_DIR": str(build),
        "MDESCRIPTOR_NATIVE_PLUGIN_DIR": str(build),
        "MDESCRIPTOR_EXPECTED_NATIVE_PLUGIN_DIR": str(build),
        "MDESCRIPTOR_BENCH_WARMUP": str(warmup),
        "MDESCRIPTOR_BENCH_REPEAT": str(repeat),
        "OMP_NUM_THREADS": "1",
        "OMP_DYNAMIC": "FALSE",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "BLIS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    })
    result = subprocess.run(
        [sys.executable, "-c", _CHILD], cwd=ROOT, env=environment,
        capture_output=True, text=True, timeout=300, check=False,
    )
    if result.returncode:
        raise RuntimeError(f"benchmark failed for {build}:\n{result.stdout}\n{result.stderr}")
    for line in reversed(result.stdout.splitlines()):
        if line.startswith("MBTR_BUILD="):
            return json.loads(line.split("=", 1)[1])
    raise RuntimeError(f"benchmark produced no result for {build}:\n{result.stdout}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.warmup < 0 or args.repeat <= 0:
        parser.error("warmup must be non-negative and repeat must be positive")
    payload = _run(args.build, args.warmup, args.repeat)
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
