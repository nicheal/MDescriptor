"""Run the SOAPTurbo comparison against the source in ``.deps``.

The upstream Fortran routine accepts a pre-built neighbor list.  We therefore
build the reference inputs once with the same native neighbor-list builder
used by MDescriptor, keep that preparation outside the upstream timer, and
record the timing scope explicitly in the result.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import platform
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

# Keep BLAS single-threaded.  MDescriptor's explicit OpenMP option controls
# its native kernel at each scaling point.
for _name in (
    "OMP_DYNAMIC",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_name] = "FALSE" if _name == "OMP_DYNAMIC" else "1"

import numpy as np  # noqa: E402
from ase.io import read  # noqa: E402

from mdescriptor import ExecutionOptions, StructureBatch  # noqa: E402
from mdescriptor.descriptors import SOAPTurbo  # noqa: E402
from mdescriptor.descriptors._kernels.core import _cpp  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
TWO_STRUCTURE = ROOT / "benchmarks/_datasets/two-structure-v1-2a727a880fef/structures.npz"
CARBON = ROOT / "benchmarks/_datasets/legacy/carbon_dataset_pbc.xyz"
I32_POINTER = ctypes.POINTER(ctypes.c_int)
F64_POINTER = ctypes.POINTER(ctypes.c_double)
THREADS = (1, 2, 4, 8, 16)
ATOL = 1e-11
RTOL = 1e-9


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_npz_batch(path: Path) -> StructureBatch:
    with np.load(path) as arrays:
        return StructureBatch(
            np.asarray(arrays["numbers"], dtype=np.int32),
            np.asarray(arrays["positions"], dtype=np.float64),
            np.asarray(arrays["cells"], dtype=np.float64),
            np.asarray(arrays["pbc"], dtype=np.int32),
            np.asarray(arrays["offsets"], dtype=np.int64),
            ("hea32-periodic", "water3-nonperiodic"),
        )


def _load_carbon_batch(path: Path) -> StructureBatch:
    structures = read(path, index=":")
    return StructureBatch.from_ase(structures)


def _reference_inputs(
    batch: StructureBatch, species: list[int], cutoff: float
) -> list[dict[str, np.ndarray]]:
    species_index = {value: index + 1 for index, value in enumerate(species)}
    result: list[dict[str, np.ndarray]] = []
    for structure in range(batch.structures):
        start, stop = map(int, batch.offsets[structure : structure + 2])
        numbers = np.ascontiguousarray(batch.numbers[start:stop], dtype=np.int32)
        positions = np.ascontiguousarray(batch.positions[start:stop], dtype=np.float64)
        graph = _cpp.build_neighbor_graph(
            numbers,
            positions,
            np.ascontiguousarray(batch.cells[structure], dtype=np.float64),
            np.ascontiguousarray(batch.pbc[structure], dtype=np.int32),
            cutoff,
        )
        offsets, atoms, shifts, displacements, distance2 = graph
        neighbor_counts: list[int] = []
        center_species = np.asarray(
            [species_index[int(value)] for value in numbers], dtype=np.int32
        )
        neighbor_species_parts: list[np.ndarray] = []
        rjs_parts: list[np.ndarray] = []
        theta_parts: list[np.ndarray] = []
        phi_parts: list[np.ndarray] = []
        for center in range(len(numbers)):
            begin, end = int(offsets[center]), int(offsets[center + 1])
            indices = np.arange(begin, end, dtype=np.int64)
            exact = (atoms[indices] == center) & np.all(shifts[indices] == 0, axis=1)
            if not np.any(exact):
                raise RuntimeError(
                    f"missing exact self pair for structure {structure}, atom {center}"
                )
            # The upstream routine expects the central atom first.  The native
            # graph is otherwise reused without changing neighbor vectors.
            ordered = np.concatenate((indices[exact], indices[~exact]))
            vectors = np.asarray(displacements[ordered], dtype=np.float64)
            distances = np.sqrt(
                np.maximum(np.asarray(distance2[ordered], dtype=np.float64), 0.0)
            )
            neighbor_counts.append(len(ordered))
            neighbor_species_parts.append(
                np.asarray(
                    [species_index[int(numbers[int(atom)])] for atom in atoms[ordered]],
                    dtype=np.int32,
                )
            )
            rjs_parts.append(distances)
            cosine = np.divide(
                vectors[:, 2],
                distances,
                out=np.zeros_like(distances),
                where=distances > 0.0,
            )
            theta_parts.append(np.arccos(np.clip(cosine, -1.0, 1.0)))
            phi_parts.append(np.arctan2(vectors[:, 1], vectors[:, 0]))
        result.append(
            {
                "center_species": center_species,
                "neighbor_species": np.ascontiguousarray(
                    np.concatenate(neighbor_species_parts), dtype=np.int32
                ),
                "n_neigh": np.asarray(neighbor_counts, dtype=np.int32),
                "rjs": np.ascontiguousarray(np.concatenate(rjs_parts), dtype=np.float64),
                "thetas": np.ascontiguousarray(
                    np.concatenate(theta_parts), dtype=np.float64
                ),
                "phis": np.ascontiguousarray(np.concatenate(phi_parts), dtype=np.float64),
            }
        )
    return result


def _official_function(library: Path):
    function = ctypes.CDLL(str(library.resolve())).soap_turbo_reference
    function.restype = None
    function.argtypes = [ctypes.c_int] * 5 + [
        I32_POINTER,
        F64_POINTER,
        ctypes.c_int,
        ctypes.c_int,
        F64_POINTER,
    ]
    return function


class OfficialSoapTurbo:
    def __init__(self, function: Any, species: list[int], config: dict[str, Any]):
        self.function = function
        self.species = species
        self.config = config
        count = len(species)
        alpha_max = config["alpha_max"]
        if isinstance(alpha_max, list):
            self.alpha_max = np.ascontiguousarray(alpha_max, dtype=np.int32)
        else:
            self.alpha_max = np.full(count, int(alpha_max), dtype=np.int32)
        self.feature_count = int(self.alpha_max.sum())
        self.feature_count = (
            self.feature_count * (self.feature_count + 1) // 2 * (config["l_max"] + 1)
        )
        self.rcut_hard = np.full(count, config["rcut_hard"], dtype=np.float64)
        self.rcut_soft = np.full(count, config["rcut_soft"], dtype=np.float64)
        self.nf = np.full(count, config["nf"], dtype=np.float64)
        self.global_scaling = np.ones(count, dtype=np.float64)
        self.atom_sigma_r = np.full(count, config["atom_sigma_r"], dtype=np.float64)
        self.atom_sigma_r_scaling = np.full(
            count, config["atom_sigma_r_scaling"], dtype=np.float64
        )
        self.atom_sigma_t = np.full(count, config["atom_sigma_t"], dtype=np.float64)
        self.atom_sigma_t_scaling = np.full(
            count, config["atom_sigma_t_scaling"], dtype=np.float64
        )
        self.amplitude_scaling = np.full(
            count, config["amplitude_scaling"], dtype=np.float64
        )
        self.central_weight = np.full(count, config["central_weight"], dtype=np.float64)

    @staticmethod
    def _compression_id(mode: str | None) -> int:
        if not mode:
            return 0
        if mode == "trivial":
            return 1
        return 2 + int(mode[0]) * 3 + int(mode[2])

    def compute(self, inputs: list[dict[str, np.ndarray]]) -> np.ndarray:
        rows: list[np.ndarray] = []
        for item in inputs:
            n_sites = int(len(item["center_species"]))
            n_atom_pairs = int(len(item["rjs"]))
            output = np.empty(n_sites * self.feature_count, dtype=np.float64)
            int_data = np.ascontiguousarray(
                np.concatenate(
                    (
                        item["center_species"],
                        item["neighbor_species"],
                        item["n_neigh"],
                        self.alpha_max,
                    )
                ),
                dtype=np.int32,
            )
            real_data = np.ascontiguousarray(
                np.concatenate(
                    (
                        item["rjs"],
                        item["thetas"],
                        item["phis"],
                        self.rcut_hard,
                        self.rcut_soft,
                        self.nf,
                        self.global_scaling,
                        self.atom_sigma_r,
                        self.atom_sigma_r_scaling,
                        self.atom_sigma_t,
                        self.atom_sigma_t_scaling,
                        self.amplitude_scaling,
                        self.central_weight,
                    )
                ),
                dtype=np.float64,
            )
            self.function(
                n_sites,
                len(self.species),
                n_atom_pairs,
                int(self.config["l_max"]),
                self.feature_count,
                int_data.ctypes.data_as(I32_POINTER),
                real_data.ctypes.data_as(F64_POINTER),
                1 if self.config["basis"] == "poly3gauss" else 0,
                self._compression_id(self.config.get("compression")),
                output.ctypes.data_as(F64_POINTER),
            )
            rows.append(output.reshape(n_sites, self.feature_count))
        return np.concatenate(rows, axis=0)


def _timed(function: Any, warmup: int, repeat: int) -> tuple[np.ndarray, list[float]]:
    for _ in range(warmup):
        function()
    values: list[float] = []
    result: np.ndarray | None = None
    for _ in range(repeat):
        started = time.perf_counter()
        result = np.asarray(function(), dtype=np.float64)
        values.append(time.perf_counter() - started)
    assert result is not None
    return result, values


def _stats(raw: list[float], baseline: float | None = None) -> dict[str, Any]:
    median = float(statistics.median(raw))
    result: dict[str, Any] = {
        "raw_seconds": raw,
        "median_seconds": median,
        "p95_seconds": float(np.percentile(raw, 95)),
    }
    if baseline is not None:
        result["speedup_vs_1"] = baseline / median
        result["parallel_efficiency"] = baseline / (median * result["threads"])
    return result


def _config(species: list[int]) -> dict[str, Any]:
    return {
        "species": species,
        "alpha_max": [2] * len(species),
        "l_max": 2,
        "rcut_hard": 3.5,
        "rcut_soft": 3.0,
        "nf": 1.0,
        "radial_enhancement": 0,
        "basis": "poly3",
        "compression": None,
        "atom_sigma_r": 0.5,
        "atom_sigma_r_scaling": 0.0,
        "atom_sigma_t": 0.5,
        "atom_sigma_t_scaling": 0.0,
        "amplitude_scaling": 0.0,
        "central_weight": 1.0,
    }


def _case(
    name: str,
    source: Path,
    batch: StructureBatch,
    library: Path,
    function: Any,
    warmup: int,
    repeat: int,
    output_dir: Path,
) -> dict[str, Any]:
    species = sorted({int(value) for value in batch.numbers})
    config = _config(species)
    reference_inputs = _reference_inputs(batch, species, config["rcut_hard"])
    reference = OfficialSoapTurbo(function, species, config)
    project_single = SOAPTurbo(
        **config,
        execution=ExecutionOptions(device="cpu", num_threads=1),
    )
    reference_values, reference_raw = _timed(
        lambda: reference.compute(reference_inputs), warmup, repeat
    )
    project_values, project_raw = _timed(
        lambda: project_single.compute(batch).values, warmup, repeat
    )
    project_single.close()

    delta = project_values - reference_values
    abs_delta = np.abs(delta)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "upstream_output.npz", values=reference_values)
    np.savez_compressed(output_dir / "mdescriptor_output.npz", values=project_values)

    project_scaling: list[dict[str, Any]] = []
    for threads in THREADS:
        descriptor = SOAPTurbo(
            **config,
            execution=ExecutionOptions(device="cpu", num_threads=threads),
        )
        values, raw = _timed(
            lambda descriptor=descriptor: descriptor.compute(batch).values, warmup, repeat
        )
        descriptor.close()
        row: dict[str, Any] = {
            "threads": threads,
            **_stats(raw),
            "max_abs_vs_thread_1": float(np.max(np.abs(values - project_values))),
            "max_abs_vs_upstream": float(np.max(np.abs(values - reference_values))),
            "allclose_vs_upstream": bool(
                np.allclose(values, reference_values, rtol=RTOL, atol=ATOL)
            ),
        }
        project_scaling.append(row)
    project_baseline = project_scaling[0]["median_seconds"]
    for row in project_scaling:
        row["speedup_vs_1"] = project_baseline / row["median_seconds"]
        row["parallel_efficiency"] = row["speedup_vs_1"] / row["threads"]

    return {
        "name": name,
        "source": str(source.resolve()),
        "source_sha256": _sha256(source),
        "structures": batch.structures,
        "atoms": batch.atoms,
        "species": species,
        "configuration": config,
        "shape": list(project_values.shape),
        "reference_inputs": {
            "neighbor_pairs": int(sum(len(item["rjs"]) for item in reference_inputs)),
            "prepared_outside_timing": True,
        },
        "accuracy": {
            "max_abs_error": float(abs_delta.max(initial=0.0)),
            "mean_abs_error": float(abs_delta.mean()),
            "rmse": float(np.sqrt(np.mean(delta * delta))),
            "max_rel_error_reference_gt_1e-12": float(
                np.max(
                    np.divide(
                        abs_delta,
                        np.abs(reference_values),
                        out=np.zeros_like(abs_delta),
                        where=np.abs(reference_values) > 1e-12,
                    ),
                    initial=0.0,
                )
            ),
            "allclose": bool(np.allclose(project_values, reference_values, rtol=RTOL, atol=ATOL)),
            "rtol": RTOL,
            "atol": ATOL,
            "finite_upstream": bool(np.isfinite(reference_values).all()),
            "finite_project": bool(np.isfinite(project_values).all()),
        },
        "single_thread": {
            "upstream_serial": {
                "backend": "soap_turbo-master Fortran get_soap; serial source",
                **_stats(reference_raw),
            },
            "mdescriptor": {
                "backend": "MDescriptor C++17/OpenMP num_threads=1 public API",
                **_stats(project_raw),
            },
            "upstream_over_mdescriptor": float(
                statistics.median(reference_raw) / statistics.median(project_raw)
            ),
            "mdescriptor_over_upstream": float(
                statistics.median(project_raw) / statistics.median(reference_raw)
            ),
        },
        "scaling": {
            "upstream": {
                "parallel_support": False,
                "note": "The .deps source contains no OpenMP/thread worker option; serial baseline only.",
            },
            "mdescriptor": project_scaling,
        },
        "outputs": {
            "upstream": "upstream_output.npz",
            "mdescriptor": "mdescriptor_output.npz",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-library", type=Path, required=True)
    parser.add_argument("--official-source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=5)
    args = parser.parse_args()
    if args.warmup < 0 or args.repeat <= 0:
        parser.error("warmup must be non-negative and repeat must be positive")

    function = _official_function(args.official_library)
    cases = (
        (
            "two-structure-v1-2a727a880fef",
            TWO_STRUCTURE,
            _load_npz_batch(TWO_STRUCTURE),
        ),
        ("carbon_dataset_pbc", CARBON, _load_carbon_batch(CARBON)),
    )
    results = []
    for name, source, batch in cases:
        print(f"running {name}: structures={batch.structures} atoms={batch.atoms}", flush=True)
        results.append(
            _case(
                name,
                source,
                batch,
                args.official_library,
                function,
                args.warmup,
                args.repeat,
                args.output_dir / name,
            )
        )
    try:
        commit = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    payload = {
        "schema_version": 1,
        "date": time.strftime("%Y-%m-%d"),
        "descriptor": "SOAPTurbo",
        "project": "MDescriptor",
        "official": {
            "kind": "soap_turbo-master source archive",
            "archive": str((ROOT / ".deps/soap_turbo-master.zip").resolve()),
            "archive_sha256": _sha256(ROOT / ".deps/soap_turbo-master.zip"),
            "source_root": str(args.official_source.resolve()),
            "library": str(args.official_library.resolve()),
        },
        "git_commit": commit,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "ase": __import__("importlib.metadata").metadata.version("ase"),
            "dscribe": __import__("importlib.metadata").metadata.version("dscribe"),
            "mdescriptor": __import__("importlib.metadata").metadata.version("MDescriptor"),
            "platform": platform.platform(),
            "logical_cpus": os.cpu_count(),
        },
        "protocol": {
            "device": "cpu",
            "warmup_calls": args.warmup,
            "measured_calls": args.repeat,
            "threads": list(THREADS),
            "blas_threads": 1,
            "timed_scope": "descriptor compute only; imports, input parsing, batch conversion, descriptor construction and upstream neighbor-list preparation excluded",
            "accuracy_comparison": "MDescriptor values versus soap_turbo-master get_soap using identical native-built neighbor vectors and feature configuration",
        },
        "cases": results,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "results.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {output.resolve()}", flush=True)
    print(json.dumps(payload, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
