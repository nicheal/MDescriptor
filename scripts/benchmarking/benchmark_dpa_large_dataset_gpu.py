"""Benchmark DPA4/DPA4C on the complete bundled carbon dataset.

This is a large-batch host-GPU benchmark.  The complete dataset is processed
in structure-preserving chunks so the measurement covers every frame without
requiring one descriptor workspace larger than the available GPU memory.  It
reports both a first pass and a warm pass after reusing the same descriptor
object.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from ase.io import read
from benchmark_dpa_deepmd_project_gpu import (
    DeepMDReference,
    _accuracy,
    _sha256,
)

from mdescriptor import ExecutionOptions, StructureBatch
from mdescriptor._cuda_loader import load_cuda_plugin
from mdescriptor.descriptors import DPA4, DPA4C
from mdescriptor.models import DPA4_MODEL, DPA4C_MODEL

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "benchmarks/_datasets/legacy/carbon_dataset_pbc.xyz"


def _load_chunks(chunk_frames: int) -> tuple[list[StructureBatch], int, int]:
    structures = read(DATASET, index=":")
    chunks = [
        StructureBatch.from_ase(
            structures[start : start + chunk_frames],
            ids=[
                f"carbon-pbc-frame-{index:03d}"
                for index in range(start, min(start + chunk_frames, len(structures)))
            ],
        )
        for start in range(0, len(structures), chunk_frames)
    ]
    total_atoms = sum(int(chunk.numbers.size) for chunk in chunks)
    max_chunk_atoms = max(int(chunk.numbers.size) for chunk in chunks)
    return chunks, total_atoms, max_chunk_atoms


def _timed_compute(descriptor: object, batch: StructureBatch) -> tuple[float, np.ndarray]:
    started = time.perf_counter()
    result = descriptor.compute(batch)  # type: ignore[attr-defined]
    values = np.asarray(
        result.values if hasattr(result, "values") else result,
        dtype=np.float64,
    )
    return time.perf_counter() - started, values


def _run_pass(
    descriptor: object,
    chunks: list[StructureBatch],
    label: str,
) -> tuple[float, list[np.ndarray]]:
    started = time.perf_counter()
    values: list[np.ndarray] = []
    for index, chunk in enumerate(chunks):
        _, result = _timed_compute(descriptor, chunk)
        values.append(result)
        if index == 0 or (index + 1) % 16 == 0 or index + 1 == len(chunks):
            print(f"[{label}] chunks={index + 1}/{len(chunks)}", flush=True)
    return time.perf_counter() - started, values


def _measure(
    name: str,
    descriptor_type: type[object],
    model: Path,
    chunks: list[StructureBatch],
) -> dict[str, object]:
    print(f"[{name}] DeepMD-kit: constructing", flush=True)
    started = time.perf_counter()
    reference = DeepMDReference(name, model)
    deepmd_construct = time.perf_counter() - started
    try:
        print(f"[{name}] DeepMD-kit: {reference.device}, first pass", flush=True)
        deepmd_first, expected_chunks = _run_pass(reference, chunks, f"{name}/deepmd-first")
        print(f"[{name}] DeepMD-kit: warm pass", flush=True)
        deepmd_warm, expected_warm_chunks = _run_pass(
            reference, chunks, f"{name}/deepmd-warm"
        )
        deepmd_device = str(reference.device)
        torch_version = reference.torch.__version__
        torch_cuda = reference.torch.version.cuda
    finally:
        reference.close()

    print(f"[{name}] MDescriptor CUDA: constructing", flush=True)
    started = time.perf_counter()
    project = descriptor_type(model=model, execution=ExecutionOptions(device="cuda"))
    project_construct = time.perf_counter() - started
    try:
        print(f"[{name}] MDescriptor CUDA: first pass", flush=True)
        project_first, actual_chunks = _run_pass(project, chunks, f"{name}/project-first")
        print(f"[{name}] MDescriptor CUDA: warm pass", flush=True)
        project_warm, actual_warm_chunks = _run_pass(
            project, chunks, f"{name}/project-warm"
        )
    finally:
        project.close()

    expected = np.concatenate(expected_chunks, axis=0)
    expected_warm = np.concatenate(expected_warm_chunks, axis=0)
    actual = np.concatenate(actual_chunks, axis=0)
    actual_warm = np.concatenate(actual_warm_chunks, axis=0)
    first_accuracy = _accuracy(actual, expected)
    warm_accuracy = _accuracy(actual_warm, expected_warm)
    result = {
        "descriptor": name,
        "deepmd_kit": {
            "device": deepmd_device,
            "construct_seconds": deepmd_construct,
            "first_pass_seconds": deepmd_first,
            "warm_pass_seconds": deepmd_warm,
            "torch": torch_version,
            "torch_cuda": torch_cuda,
        },
        "project_gpu": {
            "device": "cuda",
            "construct_seconds": project_construct,
            "first_pass_seconds": project_first,
            "warm_pass_seconds": project_warm,
            "includes_lazy_setup_in_first": True,
        },
        "first_pass_project_over_deepmd": project_first / deepmd_first,
        "warm_pass_project_over_deepmd": project_warm / deepmd_warm,
        "first_accuracy_vs_deepmd_kit": first_accuracy,
        "warm_accuracy_vs_deepmd_kit": warm_accuracy,
    }
    print(
        f"[{name}] first: deepmd={deepmd_first:.6f}s, project={project_first:.6f}s; "
        f"warm: deepmd={deepmd_warm:.6f}s, project={project_warm:.6f}s; "
        f"first_max_abs={first_accuracy['max_abs_error']:.6e}, "
        f"warm_max_abs={warm_accuracy['max_abs_error']:.6e}, "
        f"allclose={first_accuracy['allclose'] and warm_accuracy['allclose']}",
        flush=True,
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-frames", type=int, default=8)
    parser.add_argument("--descriptors", default="DPA4,DPA4C")
    args = parser.parse_args(argv)
    if args.chunk_frames <= 0:
        raise SystemExit("chunk-frames must be positive")
    model_paths = {"DPA4": (DPA4, DPA4_MODEL), "DPA4C": (DPA4C, DPA4C_MODEL)}
    names = [name.strip() for name in args.descriptors.split(",") if name.strip()]
    if not names or any(name not in model_paths for name in names):
        raise SystemExit("descriptors must contain only DPA4 and DPA4C")

    load_cuda_plugin(ROOT / "build-cuda")
    chunks, total_atoms, max_chunk_atoms = _load_chunks(args.chunk_frames)
    print(
        f"dataset={DATASET}, frames={sum(chunk.structures for chunk in chunks)}, "
        f"atoms={total_atoms}, chunks={len(chunks)}, chunk_frames={args.chunk_frames}, "
        f"max_chunk_atoms={max_chunk_atoms}",
        flush=True,
    )
    measurements = [
        _measure(name, *model_paths[name], chunks)
        for name in names
    ]
    import torch
    from deepmd import __version__ as deepmd_version

    result = {
        "schema_version": 2,
        "comparison": "deepmd-kit GPU vs MDescriptor CUDA on complete carbon dataset",
        "dataset": {
            "path": str(DATASET),
            "sha256": _sha256(DATASET),
            "frames": sum(int(chunk.structures) for chunk in chunks),
            "atoms": total_atoms,
            "chunks": len(chunks),
            "chunk_frames": args.chunk_frames,
            "max_chunk_atoms": max_chunk_atoms,
            "species": sorted({int(value) for chunk in chunks for value in chunk.numbers}),
        },
        "deepmd_kit_version": deepmd_version,
        "runtime": {
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "torch_cuda_available": bool(torch.cuda.is_available()),
            "torch_cuda_device_count": int(torch.cuda.device_count()),
        },
        "measurement_definition": {
            "first_pass": "all chunks computed once after descriptor construction",
            "warm_pass": "all chunks computed a second time using the same descriptor object",
            "accuracy": "elementwise comparison with atol=1e-5 and rtol=2e-5",
        },
        "measurements": measurements,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
