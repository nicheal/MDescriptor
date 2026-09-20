"""Measure ACE CUDA first-call and repeated-call costs from the ACE golden fixture."""

from __future__ import annotations

import argparse
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

ROOT = Path(__file__).resolve().parents[1]
_STATIC_ARRAYS = (
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


def _extension(directory: Path, name: str) -> Path:
    for suffix in importlib.machinery.EXTENSION_SUFFIXES:
        candidate = directory / f"{name}{suffix}"
        if candidate.is_file():
            return candidate
    raise ImportError(f"no exact {name} extension in {directory}")


def _load_extensions(directory: Path) -> tuple[Path, Path]:
    directory = directory.expanduser().resolve()
    native_path = _extension(directory, "_native")
    cuda_path = _extension(directory, "_cuda")
    import mdescriptor

    spec = importlib.util.spec_from_file_location("mdescriptor._native", native_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load native extension from {native_path}")
    native = importlib.util.module_from_spec(spec)
    sys.modules["mdescriptor._native"] = native
    spec.loader.exec_module(native)
    mdescriptor.__dict__["_native"] = native
    os.environ["MDESCRIPTOR_CUDA_PLUGIN_DIR"] = str(directory)
    os.environ["MDESCRIPTOR_NATIVE_PLUGIN_DIR"] = str(directory)
    os.environ["MDESCRIPTOR_EXPECTED_NATIVE_PLUGIN_DIR"] = str(directory)
    from mdescriptor._cuda_loader import load_cuda_plugin

    load_cuda_plugin(directory)
    cuda = importlib.import_module("mdescriptor._cuda")
    if Path(native.__file__).resolve() != native_path:
        raise ImportError(f"native extension loaded from {native.__file__}")
    if Path(cuda.__file__).resolve() != cuda_path:
        raise ImportError(f"CUDA extension loaded from {cuda.__file__}")
    return native_path, cuda_path


def _fixture_batch():
    from mdescriptor import StructureBatch

    with np.load(ROOT / "tests/golden/ace/input.npz") as arrays:
        return StructureBatch(
            np.asarray(arrays["numbers"], dtype=np.int32),
            np.asarray(arrays["positions"], dtype=np.float64),
            np.asarray(arrays["cells"], dtype=np.float64),
            np.asarray(arrays["pbc"], dtype=np.int32),
            np.asarray(arrays["offsets"], dtype=np.int64),
            ("ace-periodic", "ace-isolated"),
        )


def _cuda_descriptor():
    from mdescriptor import DescriptorConfiguration, create_descriptor

    manifest = json.loads(
        (ROOT / "tests/golden/ace/manifest.json").read_text(encoding="utf-8")
    )
    parameters = dict(manifest["configuration"]["parameters"])
    parameters["execution"] = {"device": "cuda", "num_threads": None}
    configuration = DescriptorConfiguration.from_dict(
        {
            "schema_version": manifest["configuration"]["schema_version"],
            "descriptor": "ACE",
            "parameters": parameters,
        }
    )
    return create_descriptor(configuration)


def _static_upload_bytes(descriptor) -> int:
    backend = descriptor._kernel
    payload = backend.options["_cuda_payload"]
    payload_bytes = sum(np.asarray(payload[name]).nbytes for name in _STATIC_ARRAYS)
    species_bytes = np.asarray(backend.options["species"], dtype=np.int32).nbytes
    return int(payload_bytes + species_bytes)


def _timed_compute(descriptor, batch):
    started = time.perf_counter()
    result = descriptor.compute(batch)
    elapsed_ms = (time.perf_counter() - started) * 1e3
    values = np.asarray(result.values, dtype=np.float64)
    return elapsed_ms, values


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--phase", choices=("timing", "first", "repeat"), default="timing")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=5)
    args = parser.parse_args(argv)
    if args.warmup < 0 or args.repeat <= 0:
        parser.error("warmup must be non-negative and repeat must be positive")

    native_path, cuda_path = _load_extensions(args.build)
    batch = _fixture_batch()
    descriptor = _cuda_descriptor()
    try:
        static_bytes = _static_upload_bytes(descriptor)
        first_ms = None
        first_values = None
        repeat_samples: list[float] = []
        values = None
        if args.phase in {"timing", "first"}:
            first_ms, values = _timed_compute(descriptor, batch)
            first_values = values
        if args.phase in {"timing", "repeat"}:
            for _ in range(args.warmup):
                _timed_compute(descriptor, batch)
            for _ in range(args.repeat):
                elapsed_ms, values = _timed_compute(descriptor, batch)
                repeat_samples.append(elapsed_ms)
        if values is None:
            raise RuntimeError("ACE benchmark produced no result")
        print("ACE_BUILD=" + json.dumps({
            "native": str(native_path),
            "cuda": str(cuda_path),
            "phase": args.phase,
            "warmup": args.warmup,
            "repeat": args.repeat,
            "structures": batch.structures,
            "atoms": batch.atoms,
            "offsets": batch.offsets.tolist(),
            "output_shape": list(values.shape),
            "output_bytes": int(values.nbytes),
            "static_upload_bytes_per_compute": static_bytes,
            "first_ms": first_ms,
            "repeat_samples_ms": repeat_samples,
            "repeat_median_ms": float(np.median(repeat_samples)) if repeat_samples else None,
            "first_sha256": hashlib.sha256(first_values.tobytes()).hexdigest()
            if first_values is not None else None,
            "last_sha256": hashlib.sha256(values.tobytes()).hexdigest(),
        }, sort_keys=True))
    finally:
        descriptor.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
