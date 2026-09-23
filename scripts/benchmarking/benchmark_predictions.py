"""Compare prediction latency with the pinned NEPAdapters/DeepMD references.

Run CPU and CUDA as separate processes so DeepMD chooses the requested device
before importing Torch. This script is diagnostic: precision is enforced by
the reference tests, while timing is reported rather than gated by hardware.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Any


def _median_seconds(call: Any, warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        call()
    samples = []
    for _ in range(repeats):
        start = perf_counter()
        call()
        samples.append(perf_counter() - start)
    return median(samples)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=("nep", "dpa4c"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--structures", type=int, default=8)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument(
        "--native-dir", type=Path, help="directory containing the built _native extension"
    )
    parser.add_argument(
        "--plugin-dir", type=Path, help="directory containing the built _cuda extension"
    )
    args = parser.parse_args()
    if min(args.structures, args.threads, args.repeats) <= 0 or args.warmup < 0:
        parser.error("structures, threads, repeats must be positive; warmup cannot be negative")
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["DP_INTRA_OP_PARALLELISM_THREADS"] = str(args.threads)
    os.environ["DP_INTER_OP_PARALLELISM_THREADS"] = "1"
    if args.plugin_dir is not None:
        os.environ["MDESCRIPTOR_CUDA_PLUGIN_DIR"] = str(args.plugin_dir.resolve())

    if args.native_dir is not None:
        import mdescriptor

        extensions = sorted(args.native_dir.resolve().glob("_native*.so"))
        if len(extensions) != 1:
            parser.error("--native-dir must contain exactly one _native extension")
        spec = importlib.util.spec_from_file_location("mdescriptor._native", extensions[0])
        if spec is None or spec.loader is None:
            parser.error("could not load _native extension")
        module = importlib.util.module_from_spec(spec)
        sys.modules["mdescriptor._native"] = module
        spec.loader.exec_module(module)
        mdescriptor._native = module
    if args.plugin_dir is not None:
        from mdescriptor._cuda_loader import load_cuda_plugin

        load_cuda_plugin(args.plugin_dir.resolve())

    import numpy as np

    from mdescriptor import ExecutionOptions, StructureBatch
    from mdescriptor.models import DPA4C_MODEL, NEP_MODEL
    from mdescriptor.predictors import DPA4C, NEP

    positions = np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]])
    cells = np.broadcast_to(np.eye(3) * 8.0, (args.structures, 3, 3)).copy()
    batch = StructureBatch(
        numbers=np.tile(np.array([8, 1, 1], dtype=np.int32), args.structures),
        positions=np.tile(positions, (args.structures, 1)),
        cells=cells,
        pbc=np.ones((args.structures, 3), dtype=np.int32),
        offsets=np.arange(args.structures + 1, dtype=np.int64) * 3,
        ids=tuple(f"water-{index}" for index in range(args.structures)),
    )
    predictor_type = NEP if args.model == "nep" else DPA4C
    predictor = predictor_type(
        execution=ExecutionOptions(device=args.device, num_threads=args.threads)
    )
    try:
        if args.model == "nep":
            from ase.data import chemical_symbols
            from nep_adapters import NEPCalculator

            reference = NEPCalculator(str(NEP_MODEL), backend=args.device)
            reference_types = np.asarray(
                [reference.type_dict[chemical_symbols[int(number)]] for number in batch.numbers],
                dtype=np.int32,
            )

            def run_reference() -> Any:
                return reference.predict_arrays(
                    reference_types,
                    batch.positions,
                    batch.cells.reshape(args.structures, 9),
                    np.full(args.structures, 3, dtype=np.int32),
                    pbc=batch.pbc,
                )

        else:
            from deepmd.infer import DeepPot

            reference = DeepPot(str(DPA4C_MODEL), neighbor_graph_method="ase")
            type_map = {symbol: index for index, symbol in enumerate(reference.get_type_map())}
            atom_types = np.array([type_map[symbol] for symbol in ("O", "H", "H")], dtype=np.int32)

            def run_reference() -> Any:
                return reference.eval(
                    batch.positions.reshape(args.structures, 3, 3),
                    batch.cells.reshape(args.structures, 9),
                    atom_types,
                    atomic=True,
                )

        try:
            reference_seconds = _median_seconds(run_reference, args.warmup, args.repeats)
            predicted_seconds = _median_seconds(
                lambda: predictor.predict(batch), args.warmup, args.repeats
            )
        finally:
            close = getattr(reference, "close", None)
            if callable(close):
                close()
    finally:
        predictor.close()
    print(
        json.dumps(
            {
                "model": args.model,
                "device": args.device,
                "structures": args.structures,
                "atoms": batch.atoms,
                "threads": args.threads,
                "reference_median_ms": round(reference_seconds * 1000, 3),
                "mdescriptor_median_ms": round(predicted_seconds * 1000, 3),
                "ratio_to_reference": round(predicted_seconds / reference_seconds, 3),
            }
        )
    )


if __name__ == "__main__":
    main()
