"""Compare DPA4 CUDA builds on the train.xyz launch regression.

Run each plugin in a fresh process; output includes arrays for cross-build parity.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np

from mdescriptor import ExecutionOptions, StructureBatch
from mdescriptor._cuda_loader import load_cuda_plugin
from mdescriptor.core.backends import _slice_structure_batch
from mdescriptor.descriptors import DPA4


def load_batch(path: Path) -> StructureBatch:
    positions, cells, offsets = [], [], [0]
    with path.open() as source:
        for frame in range(4128):
            count = int(source.readline())
            comment = source.readline()
            coordinates = []
            for _ in range(count):
                fields = source.readline().split()
                if frame >= 4096:
                    assert fields[0] == "C"
                    coordinates.append([float(value) for value in fields[1:4]])
            if frame >= 4096:
                match = re.search(r'Lattice="([^"]+)"', comment)
                assert match is not None
                cells.append([float(value) for value in match.group(1).split()])
                positions.extend(coordinates)
                offsets.append(len(positions))
    return StructureBatch(
        np.full(len(positions), 6, dtype=np.int32), np.asarray(positions),
        np.asarray(cells).reshape(-1, 3, 3), np.ones((32, 3), dtype=np.int32),
        np.asarray(offsets), tuple(str(i) for i in range(4096, 4128)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plugin", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=Path(".deps/train.xyz"))
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    plugin = load_cuda_plugin(args.plugin)
    batch = load_batch(args.dataset)
    report = {"plugin": str(plugin), "cases": {}}
    arrays = {}
    descriptor = DPA4(execution=ExecutionOptions(device="cuda"))
    try:
        for frames in (1, 4, 32):
            sample = _slice_structure_batch(batch, 0, frames)
            timings = []
            for iteration in range(args.repeat + 1):
                started = time.perf_counter()
                result = descriptor.compute(sample)
                elapsed = time.perf_counter() - started
                assert np.isfinite(result.values).all()
                if iteration:
                    timings.append(elapsed)
                else:
                    cold = elapsed
                    arrays[str(frames)] = result.values.copy()
                np.testing.assert_array_equal(result.values, arrays[str(frames)])
            report["cases"][str(frames)] = {
                "cold_seconds": cold, "seconds": timings,
                "median_seconds": float(np.median(timings)),
            }
            print(frames, report["cases"][str(frames)], flush=True)
    finally:
        descriptor.close()
    if args.cpu:
        cpu = DPA4(execution=ExecutionOptions(device="cpu", num_threads=4))
        try:
            arrays["cpu"] = cpu.compute(_slice_structure_batch(batch, 0, 1)).values
        finally:
            cpu.close()
        delta = np.abs(arrays["1"] - arrays["cpu"])
        report["cpu_parity"] = {"max_abs": float(delta.max()),
            "allclose": bool(np.allclose(arrays["1"], arrays["cpu"], atol=1e-5, rtol=2e-5))}
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    np.savez(args.output.with_suffix(".npz"), **arrays)


if __name__ == "__main__":
    main()
