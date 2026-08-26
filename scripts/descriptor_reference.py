"""Shared reference-evaluation helpers for descriptor benchmark generation."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

import mdescriptor
from mdescriptor import StructureBatch, get_descriptor
from mdescriptor.descriptors.model_backed.dpa import (
    _ATOMIC_SYMBOLS,
    load_dpa_checkpoint,
    new_runtime,
)
from mdescriptor.models import ModelResource

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = Path(mdescriptor.__file__).resolve().parent

REFERENCE_PROGRAM = r"""
import json
import sys
from pathlib import Path

reference_site, venv_site, request_file, response_file = sys.argv[1:]
sys.path = [
    reference_site,
    venv_site,
    "/usr/lib/python3.12",
    "/usr/lib/python3.12/lib-dynload",
]
import numpy as np
import mdescriptor

request = json.loads(Path(request_file).read_text(encoding="utf-8"))
data = request["batch"]
batch = mdescriptor.StructureBatch(
    np.asarray(data["numbers"], dtype=np.int32),
    np.asarray(data["positions"], dtype=np.float64),
    np.asarray(data["cells"], dtype=np.float64),
    np.asarray(data["pbc"], dtype=np.int32),
    np.asarray(data["offsets"], dtype=np.int64),
    tuple(data["ids"]),
)
descriptor = mdescriptor.get_descriptor(request["name"])(**request["parameters"])
result = descriptor.compute(batch)
Path(response_file + ".json").write_text(
    json.dumps(
        {
            "level": result.level.value,
            "labels": list(result.labels),
            "structure_ids": list(result.structure_ids),
            "row_offsets": (
                None if result.row_offsets is None else result.row_offsets.tolist()
            ),
            "metadata": result.metadata,
        },
        sort_keys=True,
    ),
    encoding="utf-8",
)
np.savez_compressed(response_file, values=np.asarray(result.values))
"""


def _parameters() -> dict[str, dict[str, Any]]:
    species = [1, 8]
    return {
        "SOAP": {"species": species, "r_cut": 3.5, "n_max": 2, "l_max": 2},
        "SOAPTurbo": {
            "species": species,
            "alpha_max": [2, 2],
            "l_max": 2,
            "rcut_hard": 3.5,
            "rcut_soft": 3.0,
            "atom_sigma_r": 0.5,
            "atom_sigma_t": 0.5,
        },
        "ACSF": {"species": species, "r_cut": 3.5},
        "CoulombMatrix": {"n_atoms_max": 3},
        "SineMatrix": {"n_atoms_max": 3},
        "EwaldSumMatrix": {"n_atoms_max": 3},
        "MBTR": {
            "species": species,
            "geometry": {"function": "distance"},
            "grid": {"min": 0.0, "max": 4.0, "n": 20, "sigma": 0.1},
            "weighting": {"function": "exp", "scale": 0.3, "threshold": 1e-3},
        },
        "LMBTR": {
            "species": species,
            "geometry": {"function": "distance"},
            "grid": {"min": 0.0, "max": 4.0, "n": 20, "sigma": 0.1},
            "weighting": {"function": "exp", "scale": 0.3, "threshold": 1e-3},
        },
        "ValleOganov": {
            "species": species,
            "function": "distance",
            "n": 20,
            "sigma": 0.1,
            "r_cut": 3.5,
        },
        "AtomicComposition": {"species": species},
        "NeighborList": {"cutoff": 3.5},
        "SortedDistances": {"species": species, "cutoff": 3.5, "max_neighbors": 4},
        "SphericalExpansion": {
            "species": species,
            "cutoff": 3.5,
            "density_width": 0.6,
            "max_radial": 2,
            "max_angular": 2,
        },
        "SphericalExpansionByPair": {
            "species": species,
            "cutoff": 3.5,
            "density_width": 0.6,
            "max_radial": 2,
            "max_angular": 2,
        },
        "SoapRadialSpectrum": {
            "species": species,
            "cutoff": 3.5,
            "density_width": 0.6,
            "max_radial": 2,
            "max_angular": 2,
        },
        "SoapPowerSpectrum": {
            "species": species,
            "cutoff": 3.5,
            "density_width": 0.6,
            "max_radial": 2,
            "max_angular": 2,
        },
        "LodeSphericalExpansion": {
            "species": species,
            "cutoff": 3.5,
            "density_width": 0.6,
            "max_radial": 2,
            "max_angular": 2,
        },
        "EAD": {
            "parameters": {"L": 2, "eta": [0.05, 0.1], "Rs": [0.0, 0.5]},
            "Rc": 3.5,
        },
        "SO3": {"nmax": 2, "lmax": 2, "rcut": 3.5, "alpha": 2.0},
        "SO4": {"lmax": 2, "rcut": 3.5, "normalize_U": True},
        "SNAP": {"lmax": 2, "rcut": 3.5, "weights": {"O": 1.0}},
        "LBispectrum": {"twojmax": 3, "diagonal": 3, "rcut": 3.5},
        "MTP": {
            "species": species,
            "min_dist": 0.1,
            "max_dist": 3.5,
            "radial_basis_size": 2,
            "max_rank": 2,
        },
        "C00PSMLFF": {"species": species, "r_cut": 3.5, "n_radial": 2, "l_max": 2},
        "NEP": {},
        "DPA4": {},
        "DPA4C": {},
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, ModelResource):
        return value.to_dict()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _portable(value: Any) -> Any:
    """Replace checkout-specific absolute paths in the manifest."""

    if isinstance(value, str):
        package_root = str(PACKAGE_ROOT)
        if value == package_root:
            return "${PACKAGE_ROOT}"
        if value.startswith(package_root + "/"):
            return "${PACKAGE_ROOT}/" + value[len(package_root) + 1 :]
        root = str(ROOT)
        if value == root:
            return "${PROJECT_ROOT}"
        if value.startswith(root + "/"):
            return "${PROJECT_ROOT}/" + value[len(root) + 1 :]
        return value
    if isinstance(value, dict):
        return {key: _portable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_portable(item) for item in value]
    return value


def _batch_json(batch: StructureBatch) -> dict[str, Any]:
    return {
        "numbers": batch.numbers.tolist(),
        "positions": batch.positions.tolist(),
        "cells": batch.cells.tolist(),
        "pbc": batch.pbc.tolist(),
        "offsets": batch.offsets.tolist(),
        "ids": list(batch.ids),
    }


def _pair_samples(
    values: np.ndarray,
    metadata: dict[str, Any],
    row_offsets: list[int],
    batch: StructureBatch,
) -> np.ndarray:
    records = metadata.get("pair_records")
    if records is None or len(records) == 0:
        records = values[:, :5]
    records_array = np.asarray(records, dtype=np.int64)
    offsets = np.asarray(row_offsets, dtype=np.int64)
    structures = np.repeat(
        np.arange(len(offsets) - 1, dtype=np.int64),
        np.diff(offsets),
    )
    local_first = records_array[:, 0] - batch.offsets[structures]
    local_second = records_array[:, 1] - batch.offsets[structures]
    return np.column_stack(
        (structures, local_first, local_second, records_array[:, 2:5])
    ).astype(np.int64)


def _normalize_reference(
    raw: dict[str, Any],
    values: np.ndarray,
    batch: StructureBatch,
) -> dict[str, Any]:
    level = raw["level"]
    labels = list(raw["labels"])
    metadata = raw["metadata"]
    if level == "pair" and labels[:5] == [
        "first",
        "second",
        "cell_shift_a",
        "cell_shift_b",
        "cell_shift_c",
    ]:
        metadata = dict(metadata)
        metadata["pair_records"] = values[:, :5].tolist()
        values = values[:, 5:]
        labels = labels[5:]
    if level == "structure":
        samples = np.arange(values.shape[0], dtype=np.int64).reshape(-1, 1)
    elif level == "atom":
        offsets = np.asarray(raw["row_offsets"], dtype=np.int64)
        structures = np.repeat(
            np.arange(len(offsets) - 1, dtype=np.int64),
            np.diff(offsets),
        )
        local = np.concatenate(
            [np.arange(int(count), dtype=np.int64) for count in np.diff(offsets)]
        )
        samples = np.column_stack((structures, local))
    else:
        samples = _pair_samples(values, metadata, raw["row_offsets"], batch)
    return {
        "values": np.asarray(values),
        "level": level,
        "labels": labels,
        "structure_ids": raw["structure_ids"],
        "row_offsets": raw["row_offsets"],
        "samples": samples,
    }


def _reference_result(
    reference_wheel: Path,
    request: dict[str, Any],
    batch: StructureBatch,
    temp: Path,
) -> dict[str, Any]:
    site = temp / "site"
    site.mkdir()
    with zipfile.ZipFile(reference_wheel) as archive:
        archive.extractall(site)
    request_file = temp / "request.json"
    response_file = temp / "response.npz"
    request_file.write_text(json.dumps(_json_safe(request)), encoding="utf-8")
    venv_site = (
        Path(sys.prefix)
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    subprocess.run(
        [
            sys.executable,
            "-S",
            "-c",
            REFERENCE_PROGRAM,
            str(site),
            str(venv_site),
            str(request_file),
            str(response_file),
        ],
        check=True,
        cwd="/tmp",
    )
    raw = json.loads(
        (temp / "response.npz.json").read_text(encoding="utf-8")
    )
    with np.load(response_file) as arrays:
        return _normalize_reference(raw, arrays["values"], batch)


def _digest(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def _current_result(
    name: str,
    parameters: dict[str, Any],
    batch: StructureBatch,
) -> tuple[Any, dict[str, Any]]:
    descriptor = get_descriptor(name)(**parameters)
    try:
        result = descriptor.compute(batch)
        resolved = getattr(descriptor, "resolved_model", None)
        model = None
        if resolved is not None:
            model = {
                "digest": resolved.digest,
                "source": resolved.source,
                "path": str(resolved.path),
            }
        elif isinstance(parameters.get("model"), Path):
            model = {
                "digest": _digest(parameters["model"]),
                "source": "explicit",
                "path": str(parameters["model"]),
            }
        return result, {"configuration": descriptor.configuration.to_dict(), "model": model}
    finally:
        descriptor.close()


def _dpa_reference_values(
    name: str,
    model_path: Path,
    batch: StructureBatch,
    calibrate: Any,
) -> np.ndarray:
    """Evaluate DPA through the bundled reference evaluator, not its adapter."""

    _info, checkpoint = load_dpa_checkpoint(
        model_path,
        expected_descriptor="DPA4" if name == "DPA4" else "DPA4C",
    )
    evaluator = new_runtime(model_path, checkpoint)
    if name == "DPA4C":
        evaluator.descriptor._calibrate_output = True if calibrate is None else bool(calibrate)

    rows: list[np.ndarray] = []
    for frame in range(batch.structures):
        begin = int(batch.offsets[frame])
        end = int(batch.offsets[frame + 1])
        symbols = [_ATOMIC_SYMBOLS[int(number)] for number in batch.numbers[begin:end]]
        atype = evaluator.symbols_to_atype(symbols)
        values = evaluator.compute(
            batch.positions[begin:end],
            atype,
            batch.cells[frame],
        )
        rows.append(np.asarray(values)[0])
    if not rows:
        return np.empty((0, int(evaluator.dim_out)), dtype=np.float64)
    return np.concatenate(rows, axis=0).astype(np.float64, copy=False)


def _reference_package_digest() -> str:
    """Hash the direct NumPy evaluator used as the DPA wrapper reference."""

    package = ROOT / "src" / "mdescriptor" / "descriptors" / "model_backed" / "_vendor" / "dpa4desc"
    checksum = hashlib.sha256()
    for path in sorted(package.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts:
            checksum.update(str(path.relative_to(package)).encode("utf-8"))
            checksum.update(path.read_bytes())
    return checksum.hexdigest()
