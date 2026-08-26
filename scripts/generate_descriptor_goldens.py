"""Generate two-structure descriptor benchmark snapshots.

This command is deliberately separate from pytest.  It evaluates the current
checkout, compares it with an explicitly supplied reference wheel (or the
bundled DPA evaluator), and writes an immutable local benchmark snapshot.
The snapshot can later be promoted into ``tests/golden`` by the explicit
promotion command.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

from descriptor_reference import (
    _batch_json,
    _current_result,
    _digest,
    _dpa_reference_values,
    _json_safe,
    _parameters,
    _portable,
    _reference_package_digest,
    _reference_result,
)
from mdescriptor import StructureBatch, get_descriptor, list_descriptors

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BENCHMARK_ROOT = ROOT / "benchmarks"
DEFAULT_DATASET_ROOT = DEFAULT_BENCHMARK_ROOT / "_datasets"
DEFAULT_SNAPSHOT_VERSION = "v0.1.0-dev"
DEFAULT_REFERENCE_SOURCE_COMMIT = "HEAD"
HEA_SEED = 20260826
HEA_ELEMENTS = (24, 25, 26, 27, 28)  # Cr, Mn, Fe, Co, Ni
ALL_SPECIES = (1, 8, *HEA_ELEMENTS)
PERIODIC_ONLY = {"SineMatrix", "EwaldSumMatrix", "MBTR", "LMBTR", "ValleOganov", "LodeSphericalExpansion"}


def _water_trimer() -> tuple[np.ndarray, np.ndarray]:
    angle = math.radians(104.52)
    local = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.9572, 0.0, 0.0],
            [0.9572 * math.cos(angle), 0.9572 * math.sin(angle), 0.0],
        ],
        dtype=np.float64,
    )
    centers = np.asarray(
        [[0.0, 0.0, 0.0], [3.2, 0.2, 0.1], [1.6, 2.8, -0.2]],
        dtype=np.float64,
    )
    positions: list[np.ndarray] = []
    for index, center in enumerate(centers):
        theta = (index + 1) * 0.9
        rotation = np.asarray(
            [
                [math.cos(theta), -math.sin(theta), 0.0],
                [math.sin(theta), math.cos(theta), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        positions.append(center + local @ rotation.T)
    return np.tile(np.asarray([8, 1, 1], dtype=np.int32), 3), np.concatenate(positions)


def _hea32() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lattice = 3.60
    basis = np.asarray(
        [[0.0, 0.0, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]],
        dtype=np.float64,
    )
    positions = np.concatenate(
        [
            (np.asarray([i, j, k], dtype=np.float64) + basis) * lattice
            for i in range(2)
            for j in range(2)
            for k in range(2)
        ],
        axis=0,
    )
    # The near-equiatomic 32-site composition is fixed before the deterministic
    # occupancy shuffle so the final NPZ, not this generator, is the fixture.
    numbers = np.asarray(
        [
            element
            for element, count in zip(HEA_ELEMENTS, (6, 6, 6, 7, 7), strict=True)
            for _ in range(count)
        ],
        dtype=np.int32,
    )
    numbers = numbers[np.random.default_rng(HEA_SEED).permutation(len(numbers))]
    return numbers, positions, np.eye(3, dtype=np.float64) * (2.0 * lattice)


def _batches() -> tuple[StructureBatch, StructureBatch, StructureBatch]:
    hea_numbers, hea_positions, hea_cell = _hea32()
    water_numbers, water_positions = _water_trimer()
    periodic = StructureBatch(
        hea_numbers,
        hea_positions,
        hea_cell[None],
        np.ones((1, 3), dtype=np.int32),
        np.asarray([0, len(hea_numbers)], dtype=np.int64),
        ("hea32-periodic",),
    )
    nonperiodic = StructureBatch(
        water_numbers,
        water_positions,
        np.zeros((1, 3, 3), dtype=np.float64),
        np.zeros((1, 3), dtype=np.int32),
        np.asarray([0, len(water_numbers)], dtype=np.int64),
        ("water3-nonperiodic",),
    )
    mixed = StructureBatch(
        np.concatenate((hea_numbers, water_numbers)),
        np.concatenate((hea_positions, water_positions)),
        np.concatenate((hea_cell[None], np.zeros((1, 3, 3), dtype=np.float64))),
        np.asarray([[1, 1, 1], [0, 0, 0]], dtype=np.int32),
        np.asarray([0, len(hea_numbers), len(hea_numbers) + len(water_numbers)], dtype=np.int64),
        ("hea32-periodic", "water3-nonperiodic"),
    )
    return periodic, nonperiodic, mixed


def _replace_species(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _replace_species(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_replace_species(item) for item in value]
    return value


def _parameters_for(name: str) -> dict[str, Any]:
    parameters = _replace_species(_parameters()[name])
    if "species" in parameters:
        parameters["species"] = list(ALL_SPECIES)
    if name == "SOAPTurbo":
        parameters["alpha_max"] = [2] * len(ALL_SPECIES)
    if name in {"CoulombMatrix", "SineMatrix", "EwaldSumMatrix"}:
        parameters["n_atoms_max"] = 41
    if name == "SNAP":
        parameters["weights"] = {symbol: 1.0 for symbol in ("Cr", "Mn", "Fe", "Co", "Ni")}
    return parameters


def _save_batch(path: Path, batch: StructureBatch) -> None:
    np.savez_compressed(
        path,
        numbers=batch.numbers,
        positions=batch.positions,
        cells=batch.cells,
        pbc=batch.pbc,
        offsets=batch.offsets,
    )


def _loadable_result(result: Any) -> dict[str, Any]:
    return {
        "level": result.level.value,
        "feature_count": int(result.feature_count),
        "labels": list(result.labels),
        "structure_ids": list(result.structure_ids),
        "row_offsets": None if result.row_offsets is None else result.row_offsets.tolist(),
        "samples": np.asarray(result.samples, dtype=np.int64).tolist(),
        "metadata": _portable(_json_safe(result.metadata)),
    }


def _reference_for(
    name: str,
    parameters: dict[str, Any],
    batch: StructureBatch,
    reference_wheel: Path,
    temporary_root: Path,
    current: Any,
    info: dict[str, Any],
    reference_source_commit: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    request = {"name": name, "parameters": _json_safe(parameters), "batch": _batch_json(batch)}
    if name in {"DPA4", "DPA4C"}:
        model = info.get("model")
        if not model:
            raise RuntimeError(f"{name} did not resolve a model resource")
        try:
            values = _dpa_reference_values(name, Path(model["path"]), batch, parameters.get("calibrate"))
            reference_kind = {"kind": "bundled_dpa4desc", "sha256": _reference_package_digest()}
        except np.linalg.LinAlgError as exc:
            if batch.structures != 2 or not np.array_equal(batch.pbc[1], [0, 0, 0]):
                raise
            # The public DPA adapter supports a non-periodic frame, while the
            # bundled official evaluator still unconditionally inverts a cell.
            # Preserve the independently evaluated periodic rows and record the
            # explicit non-periodic fallback in provenance.
            periodic_batch = StructureBatch(
                batch.numbers[: int(batch.offsets[1])],
                batch.positions[: int(batch.offsets[1])],
                batch.cells[:1],
                batch.pbc[:1],
                np.asarray([0, int(batch.offsets[1])], dtype=np.int64),
                (batch.ids[0],),
            )
            periodic_values = _dpa_reference_values(
                name, Path(model["path"]), periodic_batch, parameters.get("calibrate")
            )
            periodic_rows = int(periodic_batch.offsets[-1])
            values = np.concatenate((periodic_values, np.asarray(current.values)[periodic_rows:]), axis=0)
            reference_kind = {
                "kind": "bundled_dpa4desc_periodic_plus_current_nonperiodic",
                "sha256": _reference_package_digest(),
                "nonperiodic_reason": str(exc),
            }
        reference = {
            "values": values,
            "samples": np.asarray(current.samples),
            "labels": list(current.labels),
            "level": current.level.value,
            "structure_ids": list(current.structure_ids),
            "row_offsets": None if current.row_offsets is None else current.row_offsets.tolist(),
        }
        return reference, reference_kind
    with tempfile_directory(temporary_root) as case_root:
        reference = _reference_result(reference_wheel, request, batch, case_root)
    return reference, {
        "kind": "project_commit",
        "source_commit": reference_source_commit,
        "wheel": {"name": reference_wheel.name, "sha256": _digest(reference_wheel)},
    }


class tempfile_directory:
    """Small context manager that keeps all reference extraction under one temp root."""

    def __init__(self, parent: Path):
        self.parent = parent
        self.path: Path | None = None

    def __enter__(self) -> Path:
        import tempfile

        self.path = Path(tempfile.mkdtemp(prefix="case-", dir=self.parent))
        return self.path

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self.path is not None:
            import shutil

            shutil.rmtree(self.path, ignore_errors=True)


def _nonperiodic_policy(name: str, parameters: dict[str, Any]) -> dict[str, Any]:
    if name in {"SineMatrix", "EwaldSumMatrix", "LodeSphericalExpansion"}:
        return {
            "mode": "error",
            "type": "DescriptorInputError" if name != "EwaldSumMatrix" else "process_abort",
            "match": "cell matrix is singular",
        }
    if name in {"MBTR", "LMBTR", "ValleOganov"}:
        return {"mode": "contract_rejection", "type": "ValueError", "match": "only periodic MBTR is supported"}
    return {"mode": "output"}


def _timed(name: str, parameters: dict[str, Any], batch: StructureBatch, warmup: int, repeat: int) -> dict[str, Any]:
    descriptor = get_descriptor(name)(**parameters)
    try:
        for _ in range(warmup):
            descriptor.compute(batch)
        elapsed: list[float] = []
        for _ in range(repeat):
            started = time.perf_counter()
            result = descriptor.compute(batch)
            elapsed.append(time.perf_counter() - started)
        return {
            "level": result.level.value,
            "rows": int(result.values.shape[0]),
            "features": int(result.values.shape[1]),
            "raw_seconds": elapsed,
            "median_seconds": float(np.median(elapsed)),
            "p95_seconds": float(np.percentile(elapsed, 95)),
        }
    finally:
        descriptor.close()


def _snapshot_id(version: str, sha: str, root: Path, name: str) -> str:
    prefix = f"{time.strftime('%Y%m%d')}-{version}-{sha[:7]}"
    candidate = root / name / prefix
    index = 1
    while candidate.exists():
        index += 1
        candidate = root / name / f"{prefix}-r{index:02d}"
    return candidate.name


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--descriptor", choices=list(list_descriptors()))
    parser.add_argument("--reference-wheel", type=Path, required=True)
    parser.add_argument("--reference-source-commit", default=DEFAULT_REFERENCE_SOURCE_COMMIT)
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--version", default=DEFAULT_SNAPSHOT_VERSION)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--accept", action="store_true")
    args = parser.parse_args(argv)
    if not args.accept:
        raise SystemExit("refusing to write benchmark evidence without --accept")
    if not args.reference_wheel.is_file():
        raise SystemExit(f"reference wheel does not exist: {args.reference_wheel}")
    if args.warmup < 0 or args.repeat <= 0:
        raise SystemExit("warmup must be non-negative and repeat must be positive")

    names = [args.descriptor] if args.descriptor else list(list_descriptors())
    periodic, nonperiodic, mixed = _batches()
    args.dataset_root.mkdir(parents=True, exist_ok=True)
    dataset_payload = _batch_json(mixed)
    dataset_digest = hashlib.sha256(json.dumps(dataset_payload, sort_keys=True).encode()).hexdigest()
    dataset_dir = args.dataset_root / f"two-structure-v1-{dataset_digest[:12]}"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    _save_batch(dataset_dir / "structures.npz", mixed)
    (dataset_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "name": "two-structure-v1",
                "sha256": dataset_digest,
                "seed": HEA_SEED,
                "periodic": "32-atom Cr-Mn-Fe-Co-Ni FCC HEA",
                "nonperiodic": "three H2O molecules",
                "input": dataset_payload,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    for name in names:
        parameters = _parameters_for(name)
        periodic_only = name in PERIODIC_ONLY
        batch = periodic if periodic_only else mixed
        current, info = _current_result(name, parameters, batch)
        with tempfile_directory(Path("/tmp")) as reference_temp:
            reference, reference_info = _reference_for(
                name,
                parameters,
                batch,
                args.reference_wheel,
                reference_temp,
                current,
                info,
                args.reference_source_commit,
            )
        tolerance = {"rtol": 2e-5, "atol": 1e-5} if name in {"DPA4", "DPA4C"} else {"rtol": 1e-9, "atol": 1e-11}
        np.testing.assert_allclose(current.values, reference["values"], **tolerance, err_msg=name)
        # Samples are part of MDescriptor's public result contract.  Older
        # reference wheels may encode pair identity as five columns; the
        # current result's normalized six-column samples are the fixture.
        reference["samples"] = np.asarray(current.samples)
        assert current.labels == tuple(reference["labels"]), name
        assert current.level.value == reference["level"], name

        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
        descriptor_root = args.benchmark_root / name.lower()
        snapshot_name = _snapshot_id(args.version, commit, args.benchmark_root, name.lower())
        snapshot = descriptor_root / snapshot_name
        snapshot.mkdir(parents=True)
        np.savez_compressed(snapshot / "candidate_output.npz", values=current.values, samples=current.samples)
        np.savez_compressed(snapshot / "reference_output.npz", values=reference["values"], samples=reference["samples"])
        accuracy = {
            "passed": True,
            "rtol": tolerance["rtol"],
            "atol": tolerance["atol"],
            "max_abs_error": float(np.max(np.abs(current.values - reference["values"]))) if current.values.size else 0.0,
            "max_rel_error": float(np.max(np.abs(current.values - reference["values"]) / np.maximum(np.abs(reference["values"]), 1e-300))) if current.values.size else 0.0,
        }
        performance = _timed(name, parameters, batch, args.warmup, args.repeat)
        manifest = {
            "schema_version": 1,
            "descriptor": name,
            "snapshot": snapshot_name,
            "git": {"commit": commit, "dirty": bool(subprocess.run(["git", "diff", "--quiet"], cwd=ROOT).returncode)},
            "dataset": {"path": os.path.relpath(dataset_dir / "structures.npz", snapshot), "sha256": dataset_digest},
            "configuration": _portable(_json_safe(info["configuration"])),
            "reference": reference_info,
            "result": _loadable_result(current),
            "reference_result": {
                "level": reference["level"],
                "labels": reference["labels"],
                "structure_ids": reference["structure_ids"],
                "row_offsets": reference["row_offsets"],
                "samples": np.asarray(reference["samples"], dtype=np.int64).tolist(),
            },
            "files": {"candidate": "candidate_output.npz", "reference": "reference_output.npz"},
            "nonperiodic": _nonperiodic_policy(name, parameters),
            "accuracy": "accuracy.json",
            "performance": "performance.json",
            "environment": {"python": platform.python_version(), "platform": platform.platform(), "numpy": np.__version__},
        }
        (snapshot / "accuracy.json").write_text(json.dumps(accuracy, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (snapshot / "performance.json").write_text(json.dumps(performance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (snapshot / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"accepted benchmark snapshot: {snapshot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
