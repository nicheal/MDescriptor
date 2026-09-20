"""Compare CUDA output materialization before and after the host-copy change."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import cast

ROOT = Path(__file__).resolve().parents[1]

_CHILD = r'''
from __future__ import annotations

import hashlib
import importlib
import importlib.machinery
import importlib.util
import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np


def _load_native() -> None:
    root = Path(os.environ["MDESCRIPTOR_NATIVE_PLUGIN_DIR"]).resolve()
    extension = next(
        (
            root / f"_native{suffix}"
            for suffix in importlib.machinery.EXTENSION_SUFFIXES
            if (root / f"_native{suffix}").is_file()
        ),
        None,
    )
    if extension is None:
        raise ImportError(f"no exact _native extension in {root}")
    import mdescriptor

    spec = importlib.util.spec_from_file_location("mdescriptor._native", extension)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load _native from {extension}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["mdescriptor._native"] = module
    spec.loader.exec_module(module)
    mdescriptors_native_path = getattr(module, "__file__", None)
    if not mdescriptors_native_path or Path(mdescriptors_native_path).resolve() != extension:
        raise ImportError("native extension did not report the selected path")
    mdescriptor.__dict__["_native"] = module


def _batch():
    structures = 16
    atoms_per_structure = 256
    coordinates = np.indices((8, 8, 4), dtype=np.float64).reshape(3, -1).T
    coordinates *= 1.4
    positions = np.tile(coordinates, (structures, 1))
    numbers = np.tile(np.asarray([1, 8, 14, 16], dtype=np.int32), positions.shape[0] // 4)
    cells = np.tile(np.eye(3, dtype=np.float64) * 30.0, (structures, 1, 1))
    pbc = np.zeros((structures, 3), dtype=np.int32)
    offsets = np.arange(structures + 1, dtype=np.int64) * atoms_per_structure
    from mdescriptor import StructureBatch

    return StructureBatch(numbers, positions, cells, pbc, offsets, tuple(map(str, range(structures))))


def main() -> None:
    _load_native()
    from mdescriptor._cuda_loader import load_cuda_plugin

    plugin_dir = Path(os.environ["MDESCRIPTOR_CUDA_PLUGIN_DIR"]).resolve()
    load_cuda_plugin(plugin_dir)
    cuda = importlib.import_module("mdescriptor._cuda")
    cuda_path = Path(cuda.__file__).resolve()
    if cuda_path.parent != plugin_dir:
        raise ImportError(f"CUDA extension loaded from {cuda_path}")
    from mdescriptor import ComputeControl

    batch = _batch()
    backend = cuda.CudaBackend(
        "CoulombMatrix", {"n_atoms_max": 256, "permutation": "none"}
    )
    warmup = int(os.environ["MDESCRIPTOR_BENCH_WARMUP"])
    repeat = int(os.environ["MDESCRIPTOR_BENCH_REPEAT"])
    for _ in range(warmup):
        backend.compute(batch, ComputeControl())
    timings = []
    digest = ""
    shape = None
    output_bytes = 0
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    for _ in range(repeat):
        started = time.perf_counter()
        result = backend.compute(batch, ComputeControl())
        timings.append((time.perf_counter() - started) * 1e3)
        values = np.asarray(result["values"])
        shape = list(values.shape)
        output_bytes = int(values.nbytes)
        digest = hashlib.sha256(values.tobytes()).hexdigest()
        del values, result
    rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    backend.close()
    print(
        "OUTPUT_COPY_RESULT="
        + json.dumps(
            {
                "cuda": str(cuda_path),
                "digest": digest,
                "median_ms": float(np.median(timings)),
                "maxrss_kib": int(rss_after),
                "maxrss_delta_kib": int(rss_after - rss_before),
                "output_bytes": output_bytes,
                "shape": shape,
            },
            sort_keys=True,
        )
    )


main()
'''


def _run_build(path: Path, warmup: int, repeat: int) -> dict[str, object]:
    environment = os.environ.copy()
    build = path.expanduser().resolve()
    environment.update(
        {
            "MDESCRIPTOR_CUDA_PLUGIN_DIR": str(build),
            "MDESCRIPTOR_NATIVE_PLUGIN_DIR": str(build),
            "MDESCRIPTOR_BENCH_WARMUP": str(warmup),
            "MDESCRIPTOR_BENCH_REPEAT": str(repeat),
        }
    )
    completed = subprocess.run(
        [sys.executable, "-c", _CHILD],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"benchmark failed for {build}:\n{completed.stdout}\n{completed.stderr}"
        )
    for line in reversed(completed.stdout.splitlines()):
        if line.startswith("OUTPUT_COPY_RESULT="):
            return json.loads(line.split("=", 1)[1])
    raise RuntimeError(f"benchmark produced no result for {build}:\n{completed.stdout}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=ROOT / "build/review-model-integrity")
    parser.add_argument("--optimized", type=Path, default=ROOT / "build/review-output-copy")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.warmup < 0 or args.repeat <= 0:
        parser.error("warmup must be non-negative and repeat must be positive")

    baseline = _run_build(args.baseline, args.warmup, args.repeat)
    optimized = _run_build(args.optimized, args.warmup, args.repeat)
    for key in ("digest", "shape", "output_bytes"):
        if baseline[key] != optimized[key]:
            raise RuntimeError(
                f"baseline and optimized outputs differ for {key}: "
                f"{baseline[key]!r} != {optimized[key]!r}"
            )
    result = {
        "baseline": baseline,
        "optimized": optimized,
        "same_output": True,
        "timing_ratio_optimized_over_baseline": cast(float, optimized["median_ms"])
        / cast(float, baseline["median_ms"]),
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
