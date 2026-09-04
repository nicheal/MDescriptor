"""Compare DeepMD-kit reference inference with the project's CUDA DPA kernels.

The installed DeepMD-kit stack and selected execution device are reported
explicitly in the output, so CPU and CUDA timings are never mislabeled.

Each workload uses the same checkpoint and one independent descriptor object.
The measured interval is the first synchronous public descriptor call after
construction; for MDescriptor CUDA this includes lazy plugin/context/model
initialization.  Accuracy is measured against DeepMD-kit's descriptor values.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import platform
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
TWO_DATASET = ROOT / "benchmarks/_datasets/two-structure-v1-2a727a880fef"
CARBON_DATASET = ROOT / "benchmarks/_datasets/legacy/carbon_dataset_pbc.xyz"


def _set_thread_limits() -> int:
    raw = os.environ.get("DPA_BENCH_THREADS", "1")
    threads = int(raw)
    if threads < 1:
        raise ValueError("DPA_BENCH_THREADS must be positive")
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "DP_INTRA_OP_PARALLELISM_THREADS",
        "DP_INTER_OP_PARALLELISM_THREADS",
    ):
        os.environ[name] = str(threads)
    os.environ.setdefault("OMP_DYNAMIC", "FALSE")
    return threads


THREADS = _set_thread_limits()

import numpy as np  # noqa: E402
from ase.data import atomic_numbers  # noqa: E402
from ase.io import read  # noqa: E402

from mdescriptor import ExecutionOptions, StructureBatch  # noqa: E402
from mdescriptor._cuda_loader import load_cuda_plugin  # noqa: E402
from mdescriptor.descriptors import DPA4, DPA4C  # noqa: E402
from mdescriptor.models import DPA4_MODEL, DPA4C_MODEL  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _single_structure(batch: StructureBatch, index: int) -> StructureBatch:
    begin = int(batch.offsets[index])
    end = int(batch.offsets[index + 1])
    return StructureBatch(
        batch.numbers[begin:end],
        batch.positions[begin:end],
        batch.cells[index : index + 1],
        batch.pbc[index : index + 1],
        np.asarray([0, end - begin], dtype=np.int64),
        (batch.ids[index],),
    )


def _workloads() -> list[dict[str, Any]]:
    with np.load(TWO_DATASET / "structures.npz") as arrays:
        two = StructureBatch(
            np.asarray(arrays["numbers"], dtype=np.int32),
            np.asarray(arrays["positions"], dtype=np.float64),
            np.asarray(arrays["cells"], dtype=np.float64),
            np.asarray(arrays["pbc"], dtype=np.int32),
            np.asarray(arrays["offsets"], dtype=np.int64),
            ("hea32-periodic", "water3-nonperiodic"),
        )
    two_sha = _sha256(TWO_DATASET / "structures.npz")
    carbon = StructureBatch.from_ase([read(CARBON_DATASET, index=358)], ids=["carbon-0358-C4"])
    carbon_sha = _sha256(CARBON_DATASET)
    return [
        {
            "name": "two-structure-v1/hea32-periodic",
            "source": str(TWO_DATASET / "structures.npz"),
            "source_sha256": two_sha,
            "batch": _single_structure(two, 0),
        },
        {
            "name": "two-structure-v1/water3-nonperiodic",
            "source": str(TWO_DATASET / "structures.npz"),
            "source_sha256": two_sha,
            "batch": _single_structure(two, 1),
        },
        {
            "name": "carbon_dataset_pbc/frame-0358-C4",
            "source": str(CARBON_DATASET),
            "source_sha256": carbon_sha,
            "batch": carbon,
        },
    ]


def _torch_graph(graph: Any, torch: Any, device: Any) -> Any:
    values: dict[str, Any] = {}
    for field in dataclasses.fields(graph):
        value = getattr(graph, field.name)
        if value is None or field.name == "destination_sorted":
            values[field.name] = value
        elif field.name == "edge_vec":
            values[field.name] = torch.as_tensor(value, dtype=torch.float64, device=device)
        elif field.name == "edge_mask":
            values[field.name] = torch.as_tensor(value, dtype=torch.bool, device=device)
        else:
            values[field.name] = torch.as_tensor(value, dtype=torch.int64, device=device)
    return dataclasses.replace(graph, **values)


class DeepMDReference:
    """Keep one DeepMD-kit model and evaluate one-frame batches."""

    def __init__(self, descriptor: str, model: Path) -> None:
        import torch
        from deepmd.infer import DeepPot
        from deepmd.pt_expt.utils.env import DEVICE

        self.descriptor = descriptor
        self.torch = torch
        self.device = DEVICE
        kwargs = {"neighbor_graph_method": "ase"} if descriptor == "DPA4C" else {}
        self.deep_pot = DeepPot(str(model), **kwargs)
        self.deep_eval = self.deep_pot.deep_eval
        self.type_indices = {
            int(atomic_numbers[symbol]): index
            for index, symbol in enumerate(self.deep_pot.get_type_map())
        }
        self.graph_descriptor = None
        self.type_embedding = None
        if descriptor == "DPA4C":
            self.deep_eval._dpmodel.eval()
            self.graph_descriptor = self.deep_eval._dpmodel.get_dp_atomic_model().descriptor
            self.type_embedding = self.graph_descriptor.type_embedding.call()

    def compute(self, batch: StructureBatch) -> np.ndarray:
        rows: list[np.ndarray] = []
        context = self.torch.no_grad() if self.descriptor == "DPA4C" else nullcontext()
        with context:
            for frame in range(batch.structures):
                begin = int(batch.offsets[frame])
                end = int(batch.offsets[frame + 1])
                numbers = np.asarray(batch.numbers[begin:end], dtype=np.int32)
                positions = np.asarray(batch.positions[begin:end], dtype=np.float64)[None, :, :]
                cells = (
                    np.asarray(batch.cells[frame], dtype=np.float64).reshape(1, 9)
                    if bool(np.all(batch.pbc[frame] == 1))
                    else None
                )
                atom_types = np.asarray(
                    [[self.type_indices[int(number)] for number in numbers]], dtype=np.int32
                )
                if self.descriptor == "DPA4":
                    value = self.deep_eval.eval_descriptor(positions, cells, atom_types)
                    rows.append(np.asarray(value, dtype=np.float64).reshape(len(numbers), -1))
                    continue
                graph = self.deep_eval._build_eval_graph(
                    positions, atom_types, cells, self.device
                )
                graph = _torch_graph(graph, self.torch, self.device)
                output, _ = self.graph_descriptor.call_graph(  # type: ignore[union-attr]
                    graph,
                    self.torch.as_tensor(
                        atom_types.reshape(-1), dtype=self.torch.int64, device=self.device
                    ),
                    type_embedding=self.type_embedding,
                )
                rows.append(
                    output.detach().cpu().numpy().astype(np.float64, copy=False).reshape(
                        len(numbers), -1
                    )
                )
        if not rows:
            return np.empty((0, 0), dtype=np.float64)
        return np.concatenate(rows, axis=0)

    def close(self) -> None:
        close = getattr(self.deep_pot, "close", None)
        if close is not None:
            close()


def _project_measure(
    descriptor: str, model: Path, batch: StructureBatch
) -> tuple[float, float, np.ndarray]:
    descriptor_type = DPA4 if descriptor == "DPA4" else DPA4C
    started = time.perf_counter()
    kernel = descriptor_type(model=model, execution=ExecutionOptions(device="cuda"))
    construct_seconds = time.perf_counter() - started
    try:
        started = time.perf_counter()
        values = np.asarray(kernel.compute(batch).values, dtype=np.float64)
        compute_seconds = time.perf_counter() - started
    finally:
        kernel.close()
    return construct_seconds, compute_seconds, values


def _accuracy(actual: np.ndarray, expected: np.ndarray) -> dict[str, Any]:
    if actual.shape != expected.shape:
        raise ValueError(f"descriptor shape mismatch: project={actual.shape}, deepmd={expected.shape}")
    difference = np.abs(actual - expected)
    denominator = np.maximum(np.abs(expected), 1.0e-12)
    tolerance = 2.0e-5 * np.abs(expected) + 1.0e-5
    return {
        "shape": [int(value) for value in actual.shape],
        "max_abs_error": float(np.max(difference)),
        "mae": float(np.mean(difference)),
        "rmse": float(np.sqrt(np.mean(np.square(difference)))),
        "max_relative_error": float(np.max(difference / denominator)),
        "allclose": bool(np.all(difference <= tolerance)),
        "rtol": 2.0e-5,
        "atol": 1.0e-5,
    }


def _dataset_metadata(workload: dict[str, Any]) -> dict[str, Any]:
    batch = workload["batch"]
    return {
        "name": workload["name"],
        "source": workload["source"],
        "source_sha256": workload["source_sha256"],
        "frames": int(batch.structures),
        "atoms": int(batch.numbers.size),
        "species": sorted({int(value) for value in batch.numbers}),
        "pbc": batch.pbc.tolist(),
    }


def _measure(descriptor: str, model: Path, workload: dict[str, Any]) -> dict[str, Any]:
    batch = workload["batch"]
    print(f"[{descriptor}] {workload['name']}: deepmd-kit", flush=True)
    deepmd_started = time.perf_counter()
    reference = DeepMDReference(descriptor, model)
    deepmd_construct_seconds = time.perf_counter() - deepmd_started
    print(f"[{descriptor}] {workload['name']}: deepmd-kit {reference.device}", flush=True)
    try:
        started = time.perf_counter()
        expected = reference.compute(batch)
        deepmd_compute_seconds = time.perf_counter() - started
        deepmd_device = str(reference.device)
        torch_version = reference.torch.__version__
        torch_cuda = reference.torch.version.cuda
        torch_cuda_available = bool(reference.torch.cuda.is_available())
    finally:
        reference.close()

    print(f"[{descriptor}] {workload['name']}: MDescriptor CUDA", flush=True)
    model_path = model
    project_construct_seconds, project_compute_seconds, actual = _project_measure(
        descriptor, model_path, batch
    )
    accuracy = _accuracy(actual, expected)
    result = {
        "descriptor": descriptor,
        "dataset": _dataset_metadata(workload),
        "feature_count": int(actual.shape[1]),
        "deepmd_kit": {
            "device": deepmd_device,
            "construct_seconds": deepmd_construct_seconds,
            "compute_seconds": deepmd_compute_seconds,
            "torch": torch_version,
            "torch_cuda": torch_cuda,
            "torch_cuda_available": torch_cuda_available,
        },
        "project_gpu": {
            "device": "cuda",
            "construct_seconds": project_construct_seconds,
            "compute_seconds": project_compute_seconds,
            "includes_lazy_setup": True,
        },
        "deepmd_gpu_seconds_per_atom_us": 1.0e6
        * deepmd_compute_seconds
        / int(batch.numbers.size),
        "project_gpu_seconds_per_atom_us": 1.0e6
        * project_compute_seconds
        / int(batch.numbers.size),
        "project_gpu_speedup_vs_deepmd_gpu": deepmd_compute_seconds / project_compute_seconds,
        "accuracy_vs_deepmd_kit": accuracy,
    }
    print(
        f"[{descriptor}] {workload['name']}: "
        f"deepmd={deepmd_compute_seconds:.6f}s, "
        f"project_gpu={project_compute_seconds:.6f}s, "
        f"speedup={result['project_gpu_speedup_vs_deepmd_gpu']:.6f}x, "
        f"max_abs={accuracy['max_abs_error']:.6e}, "
        f"allclose={accuracy['allclose']}",
        flush=True,
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--descriptors", default="DPA4C,DPA4")
    args = parser.parse_args(argv)
    names = [name.strip() for name in args.descriptors.split(",") if name.strip()]
    model_paths = {"DPA4": DPA4_MODEL, "DPA4C": DPA4C_MODEL}
    if not names or any(name not in model_paths for name in names):
        raise SystemExit("descriptors must contain only DPA4 and DPA4C")

    load_cuda_plugin(ROOT / "build-cuda")
    measurements = []
    for descriptor in names:
        model = Path(model_paths[descriptor])
        for workload in _workloads():
            measurements.append(_measure(descriptor, model, workload))

    import torch
    from deepmd import __version__ as deepmd_version
    from deepmd.pt_expt.utils.env import DEVICE

    result = {
        "schema_version": 1,
        "comparison": "deepmd-kit reference vs MDescriptor CUDA",
        "package": "MDescriptor",
        "deepmd_kit_version": deepmd_version,
        "deepmd_kit_device": str(DEVICE),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "threads": THREADS,
        "measurement_definition": {
            "deepmd_kit": "first synchronous compute after model construction",
            "project_gpu": "first synchronous public compute after model construction; includes lazy CUDA setup",
            "speedup": "deepmd-kit GPU compute seconds divided by MDescriptor CUDA compute seconds; greater than 1 means MDescriptor CUDA is faster",
            "accuracy": "MDescriptor CUDA values compared elementwise with deepmd-kit values using atol=1e-5 and rtol=2e-5",
        },
        "runtime_check": {
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "torch_cuda_available": bool(torch.cuda.is_available()),
            "torch_cuda_device_count": int(torch.cuda.device_count()),
        },
        "models": {
            "DPA4": {"path": str(DPA4_MODEL), "sha256": _sha256(Path(DPA4_MODEL))},
            "DPA4C": {"path": str(DPA4C_MODEL), "sha256": _sha256(Path(DPA4C_MODEL))},
        },
        "measurements": measurements,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
