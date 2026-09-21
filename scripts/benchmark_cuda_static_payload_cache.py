"""Measure CUDA static-payload reuse for ACE, MTP, and C00PSMLFF."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from benchmark_ace_cuda_cache import _load_extensions

ROOT = Path(__file__).resolve().parents[1]
_ACE_ARRAYS = (
    "base_species",
    "base_radial",
    "base_angular",
    "base_magnetic",
    "radial_a",
    "radial_b",
    "radial_c",
    "center_feature_offsets",
    "feature_term_offsets",
    "term_channel_offsets",
    "term_channels",
    "term_coefficients",
)


def _batch(numbers: list[int]):
    from ase import Atoms

    from mdescriptor import StructureBatch

    positions = np.zeros((len(numbers), 3), dtype=np.float64)
    positions[:, 0] = np.arange(len(numbers), dtype=np.float64) * 1.2
    return StructureBatch.from_ase(
        [
            Atoms(
                numbers=numbers,
                positions=positions,
                cell=np.diag([10.0, 10.0, 10.0]),
                pbc=True,
            )
        ]
    )


def _descriptor(name: str):
    from mdescriptor import ExecutionOptions
    from mdescriptor.descriptors import ACE, C00PSMLFF, MTP

    execution = ExecutionOptions(device="cuda")
    if name == "ACE":
        return ACE(species=[1, 8], N=3, maxdeg=4, rcut=3.5, execution=execution)
    if name == "MTP":
        return MTP(
            species=[13, 14],
            model=ROOT / "tests/data/mlip4_test_mtp.json",
            execution=execution,
        )
    return C00PSMLFF(
        species=[1, 8],
        r_cut=3.0,
        n_radial=3,
        l_max=2,
        execution=execution,
    )


def _array_bytes(value: object) -> int:
    if isinstance(value, np.ndarray):
        return int(value.nbytes)
    if isinstance(value, Mapping):
        return sum(_array_bytes(child) for child in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_array_bytes(child) for child in value)
    return 0


def _static_payload_bytes(descriptor, name: str) -> int:
    backend = descriptor._kernel
    options = backend.options
    payload = options["_cuda_payload"]
    if name == "ACE":
        payload_bytes = sum(np.asarray(payload[key]).nbytes for key in _ACE_ARRAYS)
    elif name == "MTP":
        # model_species validates the payload on the host but is not uploaded
        # by the MLIP-4 CUDA kernel.
        payload_bytes = sum(
            _array_bytes(value)
            for key, value in payload.items()
            if key != "model_species"
        )
    else:
        payload_bytes = _array_bytes(payload)
    payload_bytes += np.asarray(options["species"], dtype=np.int32).nbytes
    if name == "C00PSMLFF":
        count = len(np.asarray(payload["radial_counts"]))
        # The CUDA path flattens four offset vectors generated from the basis.
        payload_bytes += 4 * count * np.dtype(np.int64).itemsize
    return int(payload_bytes)


def _measure(descriptor, batch, warmup: int, repeat: int) -> dict[str, object]:
    def compute() -> np.ndarray:
        return np.asarray(descriptor.compute(batch).values, dtype=np.float64)

    started = time.perf_counter()
    first = compute()
    first_ms = (time.perf_counter() - started) * 1e3
    for _ in range(warmup):
        compute()
    samples: list[float] = []
    digests: list[str] = []
    values = first
    for _ in range(repeat):
        started = time.perf_counter()
        values = compute()
        samples.append((time.perf_counter() - started) * 1e3)
        digests.append(hashlib.sha256(values.tobytes()).hexdigest())
    return {
        "first_ms": first_ms,
        "repeat_samples_ms": samples,
        "repeat_median_ms": float(np.median(samples)),
        "output_shape": list(values.shape),
        "output_bytes": int(values.nbytes),
        "repeat_stable": len(set(digests)) == 1,
        "sha256": hashlib.sha256(values.tobytes()).hexdigest(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--descriptor", choices=("ACE", "MTP", "C00PSMLFF"), required=True)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=10)
    args = parser.parse_args(argv)
    if args.warmup < 0 or args.repeat <= 0:
        parser.error("warmup must be non-negative and repeat must be positive")

    native_path, cuda_path = _load_extensions(args.build)
    numbers = [13, 14, 13, 14] if args.descriptor == "MTP" else [1, 8, 1, 8]
    batch = _batch(numbers)
    descriptor = _descriptor(args.descriptor)
    try:
        result = _measure(descriptor, batch, args.warmup, args.repeat)
        result.update(
            {
                "descriptor": args.descriptor,
                "native": str(native_path),
                "cuda": str(cuda_path),
                "static_upload_bytes_per_compute": _static_payload_bytes(
                    descriptor, args.descriptor
                ),
                "warmup": args.warmup,
                "repeat": args.repeat,
            }
        )
        print("STATIC_CACHE=" + json.dumps(result, sort_keys=True))
    finally:
        descriptor.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
