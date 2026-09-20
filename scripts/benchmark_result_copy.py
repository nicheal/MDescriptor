"""Measure the dense Python result seam with and without the private handoff."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import cast

import numpy as np

from mdescriptor.core.result import DescriptorResult, _owned_dense_values


def _child(mode: str, warmup: int, repeat: int) -> int:
    rows, columns = 4096, 1024
    half = rows // 2
    parts = [
        np.ones((half, columns), dtype=np.float64),
        np.full((rows - half, columns), 2.0, dtype=np.float64),
    ]
    structure_ids = tuple(str(index) for index in range(rows))
    labels = tuple(f"x{index}" for index in range(columns))

    def compute() -> DescriptorResult:
        combined = np.concatenate(parts, axis=0)
        values = _owned_dense_values(combined) if mode == "optimized" else combined
        return DescriptorResult(
            values,
            "structure",
            structure_ids,
            None,
            labels,
            {"descriptor": "result-copy-benchmark", "backend": "python"},
        )

    for _ in range(warmup):
        result = compute()
        del result
        gc.collect()
    timings: list[float] = []
    digest = ""
    output_bytes = 0
    for _ in range(repeat):
        started = time.perf_counter()
        result = compute()
        timings.append((time.perf_counter() - started) * 1e3)
        output = np.asarray(result.values)
        output_bytes = int(output.nbytes)
        digest = hashlib.sha256(output.tobytes()).hexdigest()
        del output, result
        gc.collect()
    print(
        "RESULT_COPY_CHILD="
        + json.dumps(
            {
                "mode": mode,
                "rows": rows,
                "columns": columns,
                "output_bytes": output_bytes,
                "median_ms": float(np.median(timings)),
                "samples_ms": timings,
                "maxrss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
                "digest": digest,
            },
            sort_keys=True,
        )
    )
    return 0


def _run_child(mode: str, warmup: int, repeat: int) -> dict[str, object]:
    completed = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--child", mode,
         "--warmup", str(warmup), "--repeat", str(repeat)],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"result-copy benchmark failed for {mode}:\n{completed.stderr}")
    for line in reversed(completed.stdout.splitlines()):
        if line.startswith("RESULT_COPY_CHILD="):
            return json.loads(line.split("=", 1)[1])
    raise RuntimeError(f"result-copy benchmark produced no result for {mode}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", choices=("baseline", "optimized"))
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=7)
    args = parser.parse_args(argv)
    if args.warmup < 0 or args.repeat <= 0:
        parser.error("warmup must be non-negative and repeat must be positive")
    if args.child:
        return _child(args.child, args.warmup, args.repeat)

    baseline = _run_child("baseline", args.warmup, args.repeat)
    optimized = _run_child("optimized", args.warmup, args.repeat)
    for key in ("digest", "rows", "columns", "output_bytes"):
        if baseline[key] != optimized[key]:
            raise RuntimeError(f"benchmark outputs differ for {key}")
    print(
        json.dumps(
            {
                "baseline": baseline,
                "optimized": optimized,
                "same_output": True,
                "eliminated_dense_copy_bytes": cast(int, optimized["output_bytes"]),
                "timing_ratio_optimized_over_baseline": (
                    cast(float, optimized["median_ms"])
                    / cast(float, baseline["median_ms"])
                ),
                "maxrss_delta_kib_optimized_over_baseline": (
                    cast(int, optimized["maxrss_kib"])
                    - cast(int, baseline["maxrss_kib"])
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
