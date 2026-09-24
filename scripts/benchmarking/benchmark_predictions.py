"""Compare prediction accuracy and latency with pinned NEPAdapters/DeepMD references.

Run CPU and CUDA as separate processes so DeepMD chooses the requested device
before importing Torch. Timings are observations, and accuracy uses the tolerances
from the corresponding external-reference checks.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import sys
from importlib.metadata import version as package_version
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Any


def _measure(call: Any, warmup: int, repeats: int) -> tuple[list[float], Any]:
    for _ in range(warmup):
        call()
    samples = []
    result = None
    for _ in range(repeats):
        start = perf_counter()
        result = call()
        samples.append(perf_counter() - start)
    return samples, result


def _prediction_arrays(result: Any) -> dict[str, np.ndarray]:
    return {
        "energy": np.asarray(result.energy),
        "atom_energy": np.asarray(result.atom_energy),
        "forces": np.asarray(result.forces),
    }


def _reference_arrays(model: str, result: Any) -> dict[str, np.ndarray]:
    if model == "nep":
        return {
            "energy": np.asarray(result.energy),
            "atom_energy": np.asarray(result.potential),
            "forces": np.asarray(result.forces),
        }
    return {
        "energy": np.asarray(result[0]).reshape(-1),
        "atom_energy": np.asarray(result[3]).reshape(-1),
        "forces": np.asarray(result[1]).reshape(-1, 3),
    }


def _accuracy(
    actual: dict[str, np.ndarray],
    expected: dict[str, np.ndarray],
    tolerances: dict[str, tuple[float, float]],
) -> dict[str, Any]:
    fields = {}
    for name, values in actual.items():
        reference = expected[name]
        if values.shape != reference.shape:
            fields[name] = {
                "shape": {"mdescriptor": list(values.shape), "reference": list(reference.shape)},
                "pass": False,
            }
            continue
        error = np.abs(values - reference)
        nonzero = reference != 0
        rtol, atol = tolerances[name]
        fields[name] = {
            "shape": list(values.shape),
            "max_abs_error": float(error.max(initial=0.0)),
            "max_rel_error_nonzero_reference": float(
                (error[nonzero] / np.abs(reference[nonzero])).max(initial=0.0)
            ),
            "rmse": float(np.sqrt(np.mean(np.square(error)))),
            "values": int(values.size),
            "pass": bool(np.allclose(values, reference, rtol=rtol, atol=atol)),
            "rtol": rtol,
            "atol": atol,
        }
    return {"fields": fields, "pass": all(v["pass"] for v in fields.values())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=("nep", "dpa4c"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--structures", type=int, default=8)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--output", type=Path, help="write the full comparison JSON here")
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

    global np
    import numpy as np

    import mdescriptor

    if args.native_dir is not None:
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

            reference_version = package_version("nep-adapters")
            if reference_version != "1.0.2":
                parser.error("NEP reference requires nep-adapters==1.0.2")
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

            reference_version = package_version("deepmd-kit")
            if reference_version != "3.2.0":
                parser.error("DPA4C reference requires deepmd-kit==3.2.0")
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
            reference_samples, reference_result = _measure(run_reference, args.warmup, args.repeats)
            predicted_samples, predicted_result = _measure(
                lambda: predictor.predict(batch), args.warmup, args.repeats
            )
        finally:
            close = getattr(reference, "close", None)
            if callable(close):
                close()
    finally:
        predictor.close()
    if args.model == "dpa4c":
        field_tolerances = {
            "energy": (2e-5, 1e-5),
            "atom_energy": (2e-5, 1e-5),
            "forces": (2e-4, 1e-4),
        }
    elif args.device == "cuda":
        field_tolerances = {name: (1e-5, 1e-4) for name in ("energy", "atom_energy", "forces")}
    else:
        field_tolerances = {name: (1e-6, 1e-6) for name in ("energy", "atom_energy", "forces")}
    actual = _prediction_arrays(predicted_result)
    expected = _reference_arrays(args.model, reference_result)
    accuracy = _accuracy(actual, expected, field_tolerances)
    reference_median = median(reference_samples)
    predicted_median = median(predicted_samples)
    output = {
        "model": args.model,
        "device": args.device,
        "structures": args.structures,
        "atoms": batch.atoms,
        "threads": args.threads,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "mdescriptor_version": mdescriptor.__version__,
        "reference_version": reference_version,
        "mdescriptor_raw_ms": [value * 1000 for value in predicted_samples],
        "reference_raw_ms": [value * 1000 for value in reference_samples],
        "mdescriptor_median_ms": predicted_median * 1000,
        "reference_median_ms": reference_median * 1000,
        "speedup_mdescriptor_vs_reference": reference_median / predicted_median,
        "accuracy": accuracy,
        "pass": accuracy["pass"],
    }
    serialized = json.dumps(output, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)
    if not output["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
