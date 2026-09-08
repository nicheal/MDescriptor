"""Compare DPA4 CPU/CUDA builds using identical real structures and warm calls.

Run each build in a fresh process. JSON stores timings; NPZ stores outputs for
cross-build parity. The dataset uses the same 512-atom frames as the launch test.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, help="CPU extension file to test")
    parser.add_argument("--plugin", type=Path, help="CUDA plugin directory")
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--frames", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--dataset", type=Path, default=Path(".deps/train.xyz"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repeat < 1 or any(frame < 1 or frame > 32 for frame in args.frames):
        parser.error("repeat must be positive and frames must be between 1 and 32")

    if args.native:
        spec = importlib.util.spec_from_file_location("mdescriptor._native", args.native)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)

    from benchmark_dpa4_launch_fix import load_batch

    from mdescriptor import ExecutionOptions
    from mdescriptor._cuda_loader import load_cuda_plugin
    from mdescriptor.core.backends import _slice_structure_batch
    from mdescriptor.descriptors import DPA4

    if args.plugin:
        load_cuda_plugin(args.plugin)
    batch = load_batch(args.dataset)
    report = {"device": args.device, "threads": args.threads, "cases": {}}
    arrays = {}
    for frames in args.frames:
        sample = _slice_structure_batch(batch, 0, frames)
        descriptor = DPA4(
            execution=ExecutionOptions(
                device=args.device,
                num_threads=args.threads if args.device == "cpu" else None,
            )
        )
        timings = []
        try:
            for iteration in range(args.repeat + 1):
                start = time.perf_counter()
                values = descriptor.compute(sample).values
                elapsed = time.perf_counter() - start
                if iteration:
                    timings.append(elapsed)
                else:
                    arrays[str(frames)] = values.copy()
                assert np.isfinite(values).all()
                np.testing.assert_array_equal(values, arrays[str(frames)])
        finally:
            descriptor.close()
        report["cases"][str(frames)] = {
            "atoms": int(sample.numbers.size),
            "seconds": timings,
            "median_seconds": float(np.median(timings)),
        }
        print(frames, report["cases"][str(frames)], flush=True)
        args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
        np.savez(args.output.with_suffix(".npz"), **arrays)


if __name__ == "__main__":
    main()
