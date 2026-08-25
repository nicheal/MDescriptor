"""Create immutable real-data descriptor benchmark snapshots.

The benchmark has two deliberately separate responsibilities:

* every package is evaluated on the same canonical ASE/extxyz input;
* only descriptor pairs with the same model or an explicitly declared
  equivalence relation receive an accuracy pass/fail result.

The reference packages are loaded lazily inside this file.  This lets the
parent process record warm timings while fresh subprocesses record cold
timings, and it means a broken optional package is recorded as ``unavailable``
instead of preventing the other packages from producing a snapshot.

Example::

    .venv/bin/python benchmarks/run_real_data_baseline.py \
        --dataset benchmarks/carbon_dataset_pbc.xyz

The default protocol is five fresh subprocesses, two warm-ups, and ten warm
measurements per case.  Use ``--cases`` to run a smaller smoke set while
developing an adapter; the protocol is stored in the resulting manifest.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib.metadata
import json
import mmap
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "benchmarks" / "carbon_dataset_pbc.xyz"
MODEL_FILES = {
    "nep89_20250409": ROOT / "src/mdescriptor/models/assets/nep89_20250409.txt",
    "DPA4-Air-OMat24-v20260704": ROOT / "src/mdescriptor/models/assets/DPA4-Air-OMat24-v20260704.pt",
    "DPA4C-Air-OMat24-v20260819": ROOT / "src/mdescriptor/models/assets/DPA4C-Air-OMat24-v20260819.pt",
}
VASPMLFF_ARCHIVE = ROOT / ".deps" / "vaspmlff.zip"
VASPMLFF_DEFAULT_LIBRARY = ROOT / ".deps" / "vaspmlff-build" / "libvaspmlff.so"

PROTOCOL = {
    "cold_subprocesses": 5,
    "warmup_calls": 2,
    "warm_measurements": 10,
    "threads": 1,
    "cold_phases": ["import", "input", "model", "kernel", "total"],
    "warm_phases": ["import", "input", "model", "kernel", "setup_total"],
    "raw_outputs": True,
    "normalized_outputs": True,
    "accuracy_metrics": [
        "max_abs",
        "rmse",
        "mae",
        "max_relative_abs",
        "cosine_similarity",
        "per_structure",
    ],
}

# These are intentionally modest, fixed configurations.  They provide a
# stable smoke/representative lane for each package while model-backed cases
# use their exact checkpoint-defined descriptor parameters.
CASES: tuple[dict[str, Any], ...] = (
    {
        "id": "mdescriptor_soap",
        "package": "MDescriptor",
        "adapter": "mdescriptor",
        "descriptor": "SOAP",
        "parameters": {
            "species": [6],
            "r_cut": 4.0,
            "n_max": 3,
            "l_max": 2,
            "sigma": 0.5,
            "average": "off",
        },
        "comparison_group": "soap_nominal",
    },
    {
        "id": "dscribe_soap",
        "package": "DScribe",
        "adapter": "dscribe",
        "descriptor": "SOAP",
        "parameters": {
            "species": ["C"],
            "r_cut": 4.0,
            "n_max": 3,
            "l_max": 2,
            "sigma": 0.5,
            "periodic": True,
            "average": "off",
        },
        "comparison_group": "soap_nominal",
    },
    {
        "id": "featomic_soap_power_spectrum",
        "package": "Featomic",
        "adapter": "featomic",
        "descriptor": "SoapPowerSpectrum",
        "parameters": {
            "cutoff": 4.0,
            "width": 0.5,
            "max_radial": 2,
            "max_angular": 2,
        },
        "comparison_group": "soap_power_nominal",
    },
    {
        "id": "mdescriptor_soap_power_spectrum",
        "package": "MDescriptor",
        "adapter": "mdescriptor",
        "descriptor": "SoapPowerSpectrum",
        "parameters": {
            "species": [6],
            "cutoff": 4.0,
            "density_width": 0.5,
            "max_radial": 2,
            "max_angular": 2,
        },
        "comparison_group": "soap_power_nominal",
    },
    {
        "id": "pyxtal_ff_acsf",
        "package": "PyXtal_FF",
        "adapter": "pyxtal_ff",
        "descriptor": "ACSF",
        "parameters": {
            "species": [6],
            "Rc": 6.0,
            "parameters": {"G2": {"eta": [0.2], "Rs": [0.0]}},
        },
        "comparison_group": "acsf_nominal",
    },
    {
        "id": "mdescriptor_acsf",
        "package": "MDescriptor",
        "adapter": "mdescriptor",
        "descriptor": "ACSF",
        "parameters": {
            "species": [6],
            "r_cut": 6.0,
            "g2_params": [[0.2, 0.0]],
        },
        "normalized_columns": [1],
        "comparison_group": "acsf_nominal",
    },
    {
        "id": "mdescriptor_c00psmlff",
        "package": "MDescriptor",
        "adapter": "mdescriptor",
        "descriptor": "C00PSMLFF",
        "parameters": {
            "species": [6],
            "r_cut": 5.0,
            "n_radial": 8,
            "l_max": 4,
            "cutoff_function": "bp",
            "radial_sigma": 0.5,
            "normalize_radial": True,
            "normalize_angular": True,
            "exclude_self_interaction": True,
        },
        "comparison_group": "c00ps_exact",
        "tolerance": {"rtol": 1e-8, "atol": 1e-8},
    },
    {
        "id": "vaspmlff_c00ps",
        "package": "VASPMLFF (.deps/vaspmlff.zip)",
        "adapter": "vaspmlff",
        "descriptor": "C00PS",
        "reference_package": ".deps/vaspmlff.zip",
        "reference_library": ".deps/vaspmlff-build/libvaspmlff.so",
        "parameters": {
            "rcut1": 5.0,
            "lmax1": 2,
            "mrb1": 8,
            "nr1": 10000,
            "ibroad1": 2,
            "lnorm1": True,
            "rcut2": 5.0,
            "lmax2": 4,
            "mrb2": 8,
            "nr2": 10000,
            "ibroad2": 2,
            "lnorm2": True,
            "ntyp": 1,
            "lsic": True,
            "sion1": 0.5,
            "sion2": 0.5,
            "source_alignment": "VASP 6.6.0 ML_SION=0.5, ML_LSIC=.TRUE.",
            "cell_layout": "ASE row vectors passed as Fortran column vectors",
            "cleanup": "isolated process exit; archive cleanup is unstable on Linux",
        },
        "comparison_group": "c00ps_exact",
        "tolerance": {"rtol": 1e-8, "atol": 1e-8},
        "warm_policy": "isolated_only",
    },
    {
        "id": "nep_adapters_nep",
        "package": "NEP-Adapters",
        "adapter": "nep_adapters",
        "descriptor": "NEP",
        "model": "nep89_20250409",
        "parameters": {"backend": "cpu", "mean_descriptor": False},
        "comparison_group": "nep_exact",
        "tolerance": {"rtol": 2e-5, "atol": 1e-6},
    },
    {
        "id": "mdescriptor_nep",
        "package": "MDescriptor",
        "adapter": "mdescriptor",
        "descriptor": "NEP",
        "model": "nep89_20250409",
        "parameters": {},
        "comparison_group": "nep_exact",
        "tolerance": {"rtol": 2e-5, "atol": 1e-6},
    },
    {
        "id": "mdescriptor_dpa4",
        "package": "MDescriptor",
        "adapter": "mdescriptor",
        "descriptor": "DPA4",
        "model": "DPA4-Air-OMat24-v20260704",
        "parameters": {},
        "comparison_group": "dpa4_exact",
        "tolerance": {"rtol": 2e-5, "atol": 2e-5},
    },
    {
        "id": "deepmd_kit_dpa4",
        "package": "deepmd-kit",
        "adapter": "deepmd_kit",
        "descriptor": "DPA4",
        "model": "DPA4-Air-OMat24-v20260704",
        "parameters": {"dtype": "native", "neighbor_backend": "auto"},
        "comparison_group": "dpa4_exact",
        "tolerance": {"rtol": 2e-5, "atol": 2e-5},
    },
    {
        "id": "mdescriptor_dpa4c",
        "package": "MDescriptor",
        "adapter": "mdescriptor",
        "descriptor": "DPA4C",
        "model": "DPA4C-Air-OMat24-v20260819",
        "parameters": {"calibrate": True},
        "comparison_group": "dpa4c_exact",
        "tolerance": {"rtol": 2e-5, "atol": 1e-5},
    },
    {
        "id": "deepmd_kit_dpa4c",
        "package": "deepmd-kit",
        "adapter": "deepmd_kit",
        "descriptor": "DPA4C",
        "model": "DPA4C-Air-OMat24-v20260819",
        "parameters": {
            "dtype": "native",
            "neighbor_graph_method": "ase",
            "route": "official_graph_descriptor",
        },
        "comparison_group": "dpa4c_exact",
        "tolerance": {"rtol": 2e-5, "atol": 1e-5},
    },
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _set_thread_environment(threads: int) -> None:
    value = str(threads)
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "RAYON_NUM_THREADS",
    ):
        os.environ[name] = value
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mdescriptor-mplconfig")


def _load_dataset(path: Path) -> tuple[list[Any], Any]:
    from ase.io import read

    from mdescriptor import StructureBatch

    structures = list(read(str(path), index=":", format="extxyz"))
    if not structures:
        raise ValueError(f"dataset contains no structures: {path}")
    ids = tuple(f"{path.name}#{index}" for index in range(len(structures)))
    return structures, StructureBatch.from_ase(structures, ids=ids)


def _dataset_metadata(path: Path, structures: list[Any], batch: Any) -> dict[str, Any]:
    determinants = np.linalg.det(batch.cells)
    energies = sum(
        "energy" in item.info or "energy" in getattr(getattr(item, "calc", None), "results", {})
        for item in structures
    )
    force_frames = sum("force" in item.arrays for item in structures)
    virial_frames = sum("virial" in item.info for item in structures)
    return {
        "path": str(path.resolve()),
        "filename": path.name,
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
        "format": "extxyz",
        "frames": batch.structures,
        "atoms": batch.atoms,
        "species": sorted({int(value) for value in batch.numbers.tolist()}),
        "pbc_values": sorted({tuple(map(int, row)) for row in batch.pbc.tolist()}),
        "cell_determinant_min": float(np.min(determinants)),
        "cell_determinant_max": float(np.max(determinants)),
        "energy_frames": energies,
        "force_frames": force_frames,
        "virial_frames": virial_frames,
    }


def _save_canonical_input(snapshot: Path, source: Path, structures: list[Any], batch: Any) -> None:
    input_dir = snapshot / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, input_dir / "source.extxyz")
    energies = np.full(batch.structures, np.nan, dtype=np.float64)
    forces = np.full((batch.atoms, 3), np.nan, dtype=np.float64)
    virials = np.full((batch.structures, 3, 3), np.nan, dtype=np.float64)
    for index, item in enumerate(structures):
        results = getattr(getattr(item, "calc", None), "results", {})
        if "energy" in item.info:
            energies[index] = float(item.info["energy"])
        elif "energy" in results:
            energies[index] = float(results["energy"])
        begin, end = int(batch.offsets[index]), int(batch.offsets[index + 1])
        if "force" in item.arrays:
            forces[begin:end] = np.asarray(item.arrays["force"], dtype=np.float64)
        if "virial" in item.info:
            value = np.asarray(item.info["virial"], dtype=np.float64)
            if value.size == 9:
                virials[index] = value.reshape(3, 3)
    np.savez_compressed(
        input_dir / "canonical.npz",
        numbers=batch.numbers,
        positions=batch.positions,
        cells=batch.cells,
        pbc=batch.pbc,
        offsets=batch.offsets,
        ids=np.asarray(batch.ids, dtype="U"),
        energies=energies,
        forces=forces,
        virials=virials,
    )


def _model_metadata(case: dict[str, Any]) -> dict[str, Any] | None:
    name = case.get("model")
    if name is None:
        return None
    path = MODEL_FILES[name]
    return {
        "name": name,
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
    }


def _reference_metadata(case: dict[str, Any]) -> dict[str, Any] | None:
    archive_name = case.get("reference_package")
    if archive_name is None:
        return None
    archive = ROOT / str(archive_name)
    metadata: dict[str, Any] = {
        "archive": str(archive.resolve()),
        "archive_exists": archive.exists(),
    }
    if archive.exists():
        metadata.update({"archive_sha256": _sha256(archive), "archive_bytes": archive.stat().st_size})
    library_name = case.get("reference_library")
    if library_name is not None:
        library = ROOT / str(library_name)
        metadata.update(
            {
                "library": str(library.resolve()),
                "library_exists": library.exists(),
            }
        )
        if library.exists():
            metadata.update({"library_sha256": _sha256(library), "library_bytes": library.stat().st_size})
        build_metadata = library.parent / "build.json"
        if build_metadata.exists():
            metadata["build_metadata"] = str(build_metadata.resolve())
            try:
                metadata["build"] = json.loads(build_metadata.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                metadata["build"] = {"status": "unreadable"}
    return metadata


def _array_payload(values: Any, *, level: str, row_offsets: np.ndarray, metadata: dict[str, Any]) -> dict[str, Any]:
    raw = np.asarray(values)
    return {
        "raw_kind": "array",
        "raw_values": raw,
        "values": np.asarray(raw, dtype=np.float64),
        "level": level,
        "row_offsets": np.asarray(row_offsets, dtype=np.int64),
        "labels": [f"feature_{index}" for index in range(raw.shape[1])] if raw.ndim == 2 else [],
        "metadata": {**metadata, "raw_dtype": str(raw.dtype)},
    }


def _descriptor_payload(result: Any, *, metadata: dict[str, Any]) -> dict[str, Any]:
    raw = np.asarray(result.values)
    labels = list(result.labels)
    return {
        "raw_kind": "array",
        "raw_values": raw,
        "values": np.asarray(raw, dtype=np.float64),
        "level": result.level.value,
        "row_offsets": (
            np.asarray(result.row_offsets, dtype=np.int64)
            if result.row_offsets is not None
            else np.asarray([], dtype=np.int64)
        ),
        "labels": labels,
        "metadata": {**dict(result.metadata), **metadata, "raw_dtype": str(raw.dtype)},
    }


def _tensor_map_payload(result: Any, batch: Any, *, metadata: dict[str, Any]) -> dict[str, Any]:
    blocks = []
    rows: dict[int, list[np.ndarray]] = {}
    for index in range(len(result)):
        block = result.block(index)
        values = np.asarray(block.values)
        samples = np.asarray(block.samples.values, dtype=np.int64)
        properties = np.asarray(block.properties.values, dtype=np.int64)
        blocks.append(
            {
                "keys": np.asarray(result.keys.values[index], dtype=np.int64),
                "values": values,
                "samples": samples,
                "properties": properties,
            }
        )
        if samples.ndim != 2 or samples.shape[1] < 2:
            raise ValueError("TensorMap block does not expose atom samples")
        for row, sample in zip(values, samples, strict=True):
            frame, local_atom = map(int, sample[:2])
            global_atom = int(batch.offsets[frame]) + local_atom
            rows.setdefault(global_atom, []).append(np.asarray(row))
    if not rows:
        normalized = np.empty((0, 0), dtype=np.float64)
    else:
        normalized_rows = [np.concatenate(rows[index]) for index in sorted(rows)]
        normalized = np.asarray(normalized_rows, dtype=np.float64)
    return {
        "raw_kind": "tensormap",
        "raw_blocks": blocks,
        "values": normalized,
        "level": "atom",
        "row_offsets": np.asarray(batch.offsets, dtype=np.int64),
        "labels": [f"feature_{index}" for index in range(normalized.shape[1])],
        "metadata": {**metadata, "block_count": len(blocks), "raw_dtype": str(blocks[0]["values"].dtype) if blocks else "float64"},
    }


def _write_payload(case_dir: Path, payload: dict[str, Any]) -> None:
    case_dir.mkdir(parents=True, exist_ok=True)
    raw_path = case_dir / "raw.npz"
    if payload["raw_kind"] == "array":
        np.savez_compressed(raw_path, values=payload["raw_values"])
    else:
        arrays: dict[str, Any] = {"keys": np.asarray([block["keys"] for block in payload["raw_blocks"]], dtype=np.int64)}
        for index, block in enumerate(payload["raw_blocks"]):
            arrays[f"block_{index}_values"] = block["values"]
            arrays[f"block_{index}_samples"] = block["samples"]
            arrays[f"block_{index}_properties"] = block["properties"]
        np.savez_compressed(raw_path, **arrays)
    np.savez_compressed(
        case_dir / "normalized.npz",
        values=np.asarray(payload["values"], dtype=np.float64),
        row_offsets=np.asarray(payload["row_offsets"], dtype=np.int64),
    )
    metadata = {
        "raw_kind": payload["raw_kind"],
        "level": payload["level"],
        "shape": list(np.asarray(payload["values"]).shape),
        "labels": payload["labels"],
        "metadata": _json_safe(payload["metadata"]),
    }
    (case_dir / "output.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _close_runtime(runtime: Any) -> None:
    close = getattr(runtime, "close", None)
    if callable(close):
        close()


def _deepmd_dpa4c_graph_descriptor(
    backend: Any,
    structures: list[Any],
    type_index: dict[str, int],
) -> np.ndarray:
    """Evaluate DPA4C through deepmd-kit's graph descriptor ABI.

    deepmd-kit 3.2.0 exposes DPA4C checkpoints as graph models but its public
    ``eval_descriptor`` helper still routes them through the dense ABI and the
    checkpoint's sentinel ``sel=999999``.  That path attempts a multi-terabyte
    allocation even for a small cell.  The graph builder and descriptor call
    below are deepmd-kit objects from the loaded official checkpoint; using
    them keeps the numerical route official while avoiding that adapter bug.
    """

    import torch
    from deepmd.dpmodel.utils.neighbor_graph import NeighborGraph

    coords = np.stack([np.asarray(item.positions, dtype=np.float64) for item in structures])
    atom_types = np.asarray(
        [
            [type_index[symbol] for symbol in item.get_chemical_symbols()]
            for item in structures
        ],
        dtype=np.int32,
    )
    cells = (
        np.stack([np.asarray(item.cell.array, dtype=np.float64) for item in structures])
        if all(bool(item.pbc.all()) for item in structures)
        else None
    )
    graph = backend._build_eval_graph(coords, atom_types, cells, torch.device("cpu"))
    graph_fields = {}
    for name in (
        "n_node",
        "edge_index",
        "edge_vec",
        "edge_mask",
        "n_local",
        "destination_order",
        "destination_row_ptr",
        "source_order",
        "source_row_ptr",
    ):
        value = getattr(graph, name, None)
        if value is not None:
            graph_fields[name] = torch.as_tensor(value)
    torch_graph = NeighborGraph(
        **graph_fields,
        destination_sorted=bool(getattr(graph, "destination_sorted", False)),
    )
    flat_types = torch.as_tensor(atom_types.reshape(-1), dtype=torch.int64)
    descriptor = backend._dpmodel.get_dp_atomic_model().descriptor
    with torch.no_grad():
        values = descriptor.call_graph(torch_graph, flat_types)[0]
    return values.detach().cpu().numpy().reshape(len(structures), len(structures[0]), -1)


class _VaspmlffC00PSReference:
    """Small ctypes binding for the C API shipped in ``vaspmlff.zip``.

    The archive's Python module is Windows-oriented and hard-codes the DLL
    name.  The C API itself is platform-neutral, so the benchmark binds that
    API directly to the Linux library built from the archive's Fortran source.
    ``close`` deliberately only drops the Python handle: the extracted
    Linux cleanup routine is not safe at interpreter shutdown with this
    legacy Fortran build, and each reference measurement is isolated in a
    worker process.
    """

    def __init__(self, library: Path, parameters: dict[str, Any]) -> None:
        self._library = ctypes.CDLL(str(library))
        setup = self._library.vaspmlff_setup
        setup.restype = ctypes.c_int
        setup.argtypes = [
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        compute = self._library.vaspmlff_compute
        compute.restype = ctypes.c_int
        compute.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int32,
        ]
        nfeatures = self._library.vaspmlff_nfeatures
        nfeatures.restype = ctypes.c_int
        nfeatures.argtypes = [ctypes.c_void_p]

        def integer(name: str, default: int) -> int:
            return int(parameters.get(name, default))

        def real(name: str, default: float) -> float:
            return float(parameters.get(name, default))

        def logical(name: str, default: bool) -> int:
            return int(bool(parameters.get(name, default)))

        iparams = (ctypes.c_int32 * 23)(
            integer("lmax1", 2),
            integer("mrb1", 8),
            integer("nr1", 10000),
            integer("ibroad1", 0),
            integer("icut1", 1),
            integer("iwindow1", 0),
            logical("lnorm1", True),
            logical("lvartran1", False),
            logical("lwindow1", False),
            logical("lmetric1", False),
            integer("lmax2", 4),
            integer("mrb2", 8),
            integer("nr2", 10000),
            integer("ibroad2", 0),
            integer("icut2", 1),
            integer("iwindow2", 0),
            logical("lnorm2", True),
            logical("lvartran2", False),
            logical("lwindow2", False),
            logical("lmetric2", False),
            integer("desc_type", 0),
            integer("iafilt2", 0),
            logical("lafilt2", False),
        )
        rparams = (ctypes.c_double * 7)(
            real("rcut1", 5.0),
            real("rcut2", 5.0),
            real("w1", 1.0),
            real("w2", 1.0),
            real("afilt2", 0.0),
            real("rmetric1", 1.0),
            real("rmetric2", 1.0),
        )
        self._ctx = ctypes.c_void_p()
        return_code = setup(
            iparams,
            rparams,
            ctypes.c_int32(integer("ntyp", 1)),
            ctypes.byref(self._ctx),
        )
        if return_code != 0:
            raise RuntimeError(f"vaspmlff_setup failed (rc={return_code})")
        self.n_features = int(nfeatures(self._ctx))
        if self.n_features <= 0:
            raise RuntimeError(f"vaspmlff_nfeatures returned {self.n_features}")
        self._compute = compute

    def _compute_into(self, cell: Any, positions: Any, types: Any, values: np.ndarray) -> None:
        cell_array = np.ascontiguousarray(cell, dtype=np.float64)
        position_array = np.ascontiguousarray(positions, dtype=np.float64)
        type_array = np.ascontiguousarray(types, dtype=np.int32)
        if cell_array.shape != (3, 3):
            raise ValueError(f"C00PS reference cell must be (3,3), got {cell_array.shape}")
        if position_array.ndim != 2 or position_array.shape[1] != 3:
            raise ValueError("C00PS reference positions must be (n_atoms,3)")
        if type_array.shape != (position_array.shape[0],):
            raise ValueError("C00PS reference types must be (n_atoms,)")
        if values.shape != (position_array.shape[0], self.n_features):
            raise ValueError("C00PS reference output has an unexpected shape")
        return_code = self._compute(
            self._ctx,
            cell_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            position_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            type_array.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            ctypes.c_int32(position_array.shape[0]),
            values.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_int32(self.n_features),
        )
        if return_code != 0:
            raise RuntimeError(f"vaspmlff_compute failed (rc={return_code})")

    def compute(self, cell: Any, positions: Any, types: Any) -> np.ndarray:
        position_array = np.asarray(positions)
        values = np.zeros((position_array.shape[0], self.n_features), dtype=np.float64)
        self._compute_into(cell, positions, types, values)
        return values

    def compute_isolated(self, cell: Any, positions: Any, types: Any) -> np.ndarray:
        """Compute one structure in a forked child and return its output.

        The extracted legacy core corrupts allocator state after one call in a
        long-lived process.  Forking after setup preserves the expensive radial
        basis and ASA tables while ensuring each child performs only one
        compute call and exits with ``os._exit`` before Fortran finalizers run.
        """

        if not hasattr(os, "fork"):
            raise RuntimeError("VASPMLFF reference isolation requires os.fork on this platform")
        position_array = np.asarray(positions)
        shape = (position_array.shape[0], self.n_features)
        size = int(np.prod(shape, dtype=np.int64) * np.dtype(np.float64).itemsize)
        shared = mmap.mmap(-1, size, access=mmap.ACCESS_WRITE)
        values = np.ndarray(shape, dtype=np.float64, buffer=shared)
        values.fill(0.0)
        child = os.fork()
        if child == 0:
            try:
                self._compute_into(cell, positions, types, values)
            except BaseException:
                traceback.print_exc()
                os._exit(1)
            os._exit(0)
        _, status = os.waitpid(child, 0)
        try:
            if not os.WIFEXITED(status) or os.WEXITSTATUS(status) != 0:
                raise RuntimeError(f"isolated VASPMLFF child failed (status={status})")
            return values.copy()
        finally:
            shared.close()

    def close(self) -> None:
        # Do not invoke vaspmlff_cleanup here; see the class docstring.
        self._ctx = ctypes.c_void_p()


def _load_factory(case: dict[str, Any], threads: int) -> Callable[[list[Any], Any], tuple[Callable[[], dict[str, Any]], Any, Callable[[int], dict[str, Any]]]]:
    """Import one backend and return a function that builds a timed runtime."""

    adapter = case["adapter"]
    parameters = case["parameters"]
    descriptor = case["descriptor"]
    model_path = MODEL_FILES.get(case.get("model", ""))

    if adapter == "mdescriptor":
        import mdescriptor
        from mdescriptor import ExecutionOptions

        def build(structures: list[Any], batch: Any) -> tuple[Callable[[], dict[str, Any]], Any]:
            kwargs = dict(parameters)
            if model_path is not None:
                kwargs["model"] = model_path
            if descriptor == "DPA4C":
                kwargs.setdefault("calibrate", True)
            if descriptor not in {"DPA4", "DPA4C"}:
                kwargs["execution"] = ExecutionOptions(device="cpu", num_threads=threads)
            instance = mdescriptor.get_descriptor(descriptor)(**kwargs)

            def compute() -> dict[str, Any]:
                payload = _descriptor_payload(instance.compute(batch), metadata={"runner": "mdescriptor"})
                columns = case.get("normalized_columns")
                if columns is not None:
                    columns_array = np.asarray(columns, dtype=np.int64)
                    payload["values"] = np.asarray(payload["values"])[:, columns_array]
                    payload["labels"] = [payload["labels"][int(index)] for index in columns_array]
                    payload["metadata"]["normalized_columns"] = columns_array.tolist()
                return payload

            def compute_single(index: int) -> dict[str, Any]:
                single_batch = mdescriptor.StructureBatch.from_ase(
                    [structures[index]], ids=[batch.ids[index]]
                )
                payload = _descriptor_payload(
                    instance.compute(single_batch), metadata={"runner": "mdescriptor"}
                )
                columns = case.get("normalized_columns")
                if columns is not None:
                    columns_array = np.asarray(columns, dtype=np.int64)
                    payload["values"] = np.asarray(payload["values"])[:, columns_array]
                    payload["labels"] = [payload["labels"][int(item)] for item in columns_array]
                return payload

            return compute, instance, compute_single

        return build

    if adapter == "vaspmlff":
        library_name = os.environ.get("MDESCRIPTOR_VASPMLFF_LIBRARY")
        library = Path(library_name).expanduser().resolve() if library_name else VASPMLFF_DEFAULT_LIBRARY
        if not library.exists():
            raise FileNotFoundError(
                f"VASPMLFF Linux library not found: {library}; run "
                "`.venv/bin/python benchmarks/build_vaspmlff_reference.py`"
            )
        reference_metadata = _reference_metadata(case) or {}
        reference_metadata["library_used"] = str(library)
        reference_metadata["library_sha256_used"] = _sha256(library)

        def build(structures: list[Any], batch: Any) -> tuple[Callable[[], dict[str, Any]], Any, Callable[[int], dict[str, Any]]]:
            instance = _VaspmlffC00PSReference(library, parameters)
            expected_ntyp = int(parameters.get("ntyp", 1))
            if expected_ntyp != 1:
                raise ValueError("the benchmark VASPMLFF adapter currently supports ntyp=1 only")

            def compute_structure(item: Any) -> np.ndarray:
                symbols = item.get_chemical_symbols()
                if any(symbol != "C" for symbol in symbols):
                    raise ValueError("the VASPMLFF benchmark case is configured for carbon only")
                types = np.ones(len(item), dtype=np.int32)
                # ASE stores lattice vectors as rows.  Because the Fortran
                # BIND(C) array is column-major, passing this C-contiguous
                # buffer makes those rows the Fortran columns expected by
                # the reference code.
                return instance.compute_isolated(item.cell.array, item.positions, types)

            metadata = {
                "runner": "vaspmlff-c-api",
                "descriptor": "C00PS",
                "n_features": instance.n_features,
                "reference": reference_metadata,
                "cell_layout": "ASE cell rows passed directly; Fortran sees lattice vectors as columns",
                "cleanup": "isolated process exit; vaspmlff_cleanup not called",
            }

            def compute() -> dict[str, Any]:
                values = np.concatenate([compute_structure(item) for item in structures], axis=0)
                return _array_payload(
                    values,
                    level="atom",
                    row_offsets=np.asarray(batch.offsets, dtype=np.int64),
                    metadata=metadata,
                )

            def compute_single(index: int) -> dict[str, Any]:
                values = compute_structure(structures[index])
                return _array_payload(
                    values,
                    level="atom",
                    row_offsets=np.asarray([0, len(structures[index])], dtype=np.int64),
                    metadata=metadata,
                )

            return compute, instance, compute_single

        return build

    if adapter == "dscribe":
        from dscribe.descriptors import SOAP

        def build(structures: list[Any], batch: Any) -> tuple[Callable[[], dict[str, Any]], Any]:
            del batch
            instance = SOAP(**parameters)

            def compute() -> dict[str, Any]:
                values = instance.create(structures, n_jobs=1)
                if isinstance(values, list):
                    values = np.concatenate([np.asarray(item) for item in values], axis=0)
                return _array_payload(values, level="atom", row_offsets=np.asarray([0] + list(np.cumsum([len(item) for item in structures])), dtype=np.int64), metadata={"runner": "dscribe", "descriptor": "SOAP"})

            def compute_single(index: int) -> dict[str, Any]:
                values = instance.create([structures[index]], n_jobs=1)
                if isinstance(values, list):
                    values = values[0]
                return _array_payload(
                    values,
                    level="atom",
                    row_offsets=np.asarray([0, len(structures[index])], dtype=np.int64),
                    metadata={"runner": "dscribe", "descriptor": "SOAP"},
                )

            return compute, instance, compute_single

        return build

    if adapter == "featomic":
        import featomic
        from featomic import basis, density

        cutoff = float(parameters["cutoff"])

        def build(structures: list[Any], batch: Any) -> tuple[Callable[[], dict[str, Any]], Any]:
            radial = basis.Gto(max_radial=int(parameters["max_radial"]), radius=cutoff)
            tensor_basis = basis.TensorProduct(
                max_angular=int(parameters["max_angular"]), radial=radial, spline_accuracy=None
            )
            calculator = featomic.SoapPowerSpectrum(
                cutoff={"radius": cutoff, "smoothing": {"type": "ShiftedCosine", "width": min(0.5, cutoff)}},
                density=density.Gaussian(width=float(parameters["width"])),
                basis=tensor_basis,
            )

            def compute() -> dict[str, Any]:
                result = calculator.compute(structures, use_native_system=True)
                return _tensor_map_payload(result, batch, metadata={"runner": "featomic", "descriptor": descriptor})

            def compute_single(index: int) -> dict[str, Any]:
                from mdescriptor import StructureBatch

                single_structures = [structures[index]]
                single_batch = StructureBatch.from_ase(single_structures, ids=[batch.ids[index]])
                result = calculator.compute(single_structures, use_native_system=True)
                return _tensor_map_payload(result, single_batch, metadata={"runner": "featomic", "descriptor": descriptor})

            return compute, calculator, compute_single

        return build

    if adapter == "nep_adapters":
        from nep_adapters import NEPCalculator

        def build(structures: list[Any], batch: Any) -> tuple[Callable[[], dict[str, Any]], Any]:
            del batch
            instance = NEPCalculator(model_path, backend=str(parameters["backend"]))

            def compute() -> dict[str, Any]:
                values = instance.get_structures_descriptor(
                    structures, mean_descriptor=bool(parameters["mean_descriptor"])
                )
                offsets = np.asarray([0] + list(np.cumsum([len(item) for item in structures])), dtype=np.int64)
                return _array_payload(values, level="atom", row_offsets=offsets, metadata={"runner": "nep_adapters", "descriptor": "NEP"})

            def compute_single(index: int) -> dict[str, Any]:
                values = instance.get_structures_descriptor(
                    [structures[index]], mean_descriptor=bool(parameters["mean_descriptor"])
                )
                return _array_payload(
                    values,
                    level="atom",
                    row_offsets=np.asarray([0, len(structures[index])], dtype=np.int64),
                    metadata={"runner": "nep_adapters", "descriptor": "NEP"},
                )

            return compute, instance, compute_single

        return build

    if adapter == "pyxtal_ff":
        os.environ.setdefault("NUMBA_DISABLE_JIT", "0")
        from pyxtal_ff.descriptors.ACSF import ACSF

        def build(structures: list[Any], batch: Any) -> tuple[Callable[[], dict[str, Any]], Any]:
            del batch
            instance = ACSF(parameters["parameters"], Rc=float(parameters["Rc"]), derivative=False)

            def compute() -> dict[str, Any]:
                values = np.concatenate(
                    [np.asarray(instance.calculate(item, system=parameters["species"])["x"]) for item in structures],
                    axis=0,
                )
                offsets = np.asarray([0] + list(np.cumsum([len(item) for item in structures])), dtype=np.int64)
                return _array_payload(values, level="atom", row_offsets=offsets, metadata={"runner": "pyxtal_ff", "descriptor": "ACSF"})

            def compute_single(index: int) -> dict[str, Any]:
                values = np.asarray(
                    instance.calculate(structures[index], system=parameters["species"])["x"]
                )
                return _array_payload(
                    values,
                    level="atom",
                    row_offsets=np.asarray([0, len(structures[index])], dtype=np.int64),
                    metadata={"runner": "pyxtal_ff", "descriptor": "ACSF"},
                )

            return compute, instance, compute_single

        return build

    if adapter == "deepmd_kit":
        from deepmd.infer import DeepEval

        def build(structures: list[Any], batch: Any) -> tuple[Callable[[], dict[str, Any]], Any]:
            del batch
            instance = DeepEval(
                str(model_path),
                **(
                    {"neighbor_graph_method": str(parameters["neighbor_graph_method"])}
                    if descriptor == "DPA4C" and "neighbor_graph_method" in parameters
                    else {}
                ),
            )
            type_map = list(instance.get_type_map())
            type_index = {symbol: index for index, symbol in enumerate(type_map)}
            backend = instance.deep_eval
            route = str(parameters.get("route", "public_eval_descriptor"))

            def compute_indices(indices: list[int]) -> np.ndarray:
                groups: dict[tuple[int, bool], list[int]] = {}
                for index in indices:
                    item = structures[index]
                    groups.setdefault((len(item), bool(item.pbc.all())), []).append(index)
                rows: list[np.ndarray | None] = [None] * len(indices)
                for group_indices in groups.values():
                    items = [structures[index] for index in group_indices]
                    if descriptor == "DPA4C":
                        group_values = _deepmd_dpa4c_graph_descriptor(
                            backend, items, type_index
                        )
                    else:
                        coords = np.stack(
                            [np.asarray(item.positions, dtype=np.float64) for item in items]
                        )
                        atom_type_rows = np.asarray(
                            [
                                [type_index[symbol] for symbol in item.get_chemical_symbols()]
                                for item in items
                            ],
                            dtype=np.int32,
                        )
                        if not np.all(atom_type_rows == atom_type_rows[0]):
                            raise ValueError(
                                "deepmd-kit public eval_descriptor requires one atom-type "
                                "vector per grouped batch; split frames with different types"
                            )
                        atom_types = atom_type_rows[0]
                        cells = (
                            np.stack(
                                [
                                    np.asarray(item.cell.array, dtype=np.float64).reshape(9)
                                    for item in items
                                ]
                            )
                            if all(bool(item.pbc.all()) for item in items)
                            else None
                        )
                        group_values = np.asarray(
                            instance.eval_descriptor(
                                coords,
                                cells,
                                atom_types,
                                dtype=str(parameters.get("dtype", "native")),
                            )
                        )
                    for local, index in enumerate(group_indices):
                        rows[indices.index(index)] = np.asarray(group_values[local])
                return np.concatenate([row for row in rows if row is not None], axis=0)

            metadata = {
                "runner": "deepmd-kit",
                "descriptor": descriptor,
                "type_map": type_map,
                "route": route,
            }

            def compute() -> dict[str, Any]:
                values = compute_indices(list(range(len(structures))))
                offsets = np.asarray([0] + list(np.cumsum([len(item) for item in structures])), dtype=np.int64)
                return _array_payload(values, level="atom", row_offsets=offsets, metadata=metadata)

            def compute_single(index: int) -> dict[str, Any]:
                item = structures[index]
                values = compute_indices([index])
                return _array_payload(
                    values,
                    level="atom",
                    row_offsets=np.asarray([0, len(item)], dtype=np.int64),
                    metadata=metadata,
                )

            return compute, instance, compute_single

        return build

    raise ValueError(f"unknown benchmark adapter {adapter!r}")


def _run_worker(case: dict[str, Any], dataset: Path, output_dir: Path | None, threads: int) -> dict[str, Any]:
    _set_thread_environment(threads)
    total_start = time.perf_counter()
    phase: dict[str, float] = {}
    try:
        import_start = time.perf_counter()
        factory = _load_factory(case, threads)
        phase["import"] = time.perf_counter() - import_start
        input_start = time.perf_counter()
        structures, batch = _load_dataset(dataset)
        phase["input"] = time.perf_counter() - input_start
        model_start = time.perf_counter()
        compute, runtime, _compute_single = factory(structures, batch)
        phase["model"] = time.perf_counter() - model_start
        kernel_start = time.perf_counter()
        payload = compute()
        phase["kernel"] = time.perf_counter() - kernel_start
        phase["total"] = time.perf_counter() - total_start
        if output_dir is not None:
            _write_payload(output_dir, payload)
        _close_runtime(runtime)
        return {
            "status": "ok",
            "case_id": case["id"],
            "phase_seconds": phase,
            "shape": list(np.asarray(payload["values"]).shape),
            "dtype": str(np.asarray(payload["raw_values"] if payload["raw_kind"] == "array" else payload["values"]).dtype),
        }
    except Exception as exc:
        phase["total"] = time.perf_counter() - total_start
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "failure.json").write_text(
                json.dumps(
                    {
                        "status": "unavailable",
                        "case_id": case["id"],
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                        "phase_seconds": phase,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
        return {
            "status": "unavailable",
            "case_id": case["id"],
            "phase_seconds": phase,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def _worker_main(args: argparse.Namespace) -> int:
    case = next((item for item in CASES if item["id"] == args.case_id), None)
    if case is None:
        raise SystemExit(f"unknown case: {args.case_id}")
    result = _run_worker(
        case,
        Path(args.dataset).resolve(),
        None if args.output_dir is None else Path(args.output_dir).resolve(),
        args.threads,
    )
    print(json.dumps(result, sort_keys=True))
    if case.get("adapter") == "vaspmlff":
        # The legacy Fortran library may corrupt its allocator state during
        # interpreter shutdown even after the numeric output is complete.
        # The worker has already written the payload and JSON protocol result,
        # so terminate the isolated process without invoking Python finalizers.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
    return 0


def _parse_worker_output(stdout: str, returncode: int) -> dict[str, Any]:
    lines = [line for line in stdout.splitlines() if line.strip()]
    if lines:
        try:
            value = json.loads(lines[-1])
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            pass
    return {
        "status": "unavailable",
        "error_type": "WorkerProtocolError",
        "error": f"worker returned no JSON result (exit={returncode})",
    }


def _run_cold(case: dict[str, Any], dataset: Path, case_dir: Path, repeats: int, threads: int) -> list[dict[str, Any]]:
    logs = case_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    results = []
    for index in range(repeats):
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--case-id",
            case["id"],
            "--dataset",
            str(dataset),
            "--threads",
            str(threads),
        ]
        if index == 0:
            command.extend(["--output-dir", str(case_dir)])
        start = time.perf_counter()
        completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        wall = time.perf_counter() - start
        result = _parse_worker_output(completed.stdout, completed.returncode)
        result["subprocess_wall_seconds"] = wall
        results.append(result)
        (logs / f"cold_{index:02d}.stdout.log").write_text(completed.stdout, encoding="utf-8")
        (logs / f"cold_{index:02d}.stderr.log").write_text(completed.stderr, encoding="utf-8")
    return results


def _run_warm(
    case: dict[str, Any],
    dataset: Path,
    case_dir: Path,
    warmups: int,
    repeats: int,
    threads: int,
    *,
    measure_per_structure: bool,
) -> dict[str, Any]:
    _set_thread_environment(threads)
    total_start = time.perf_counter()
    try:
        import_start = time.perf_counter()
        factory = _load_factory(case, threads)
        import_seconds = time.perf_counter() - import_start
        input_start = time.perf_counter()
        structures, batch = _load_dataset(dataset)
        input_seconds = time.perf_counter() - input_start
        model_start = time.perf_counter()
        compute, runtime, compute_single = factory(structures, batch)
        model_seconds = time.perf_counter() - model_start
        payload = None
        for _ in range(warmups):
            payload = compute()
        samples = []
        for _ in range(repeats):
            start = time.perf_counter()
            payload = compute()
            samples.append(time.perf_counter() - start)
        per_structure = []
        if measure_per_structure:
            for index in range(len(structures)):
                start = time.perf_counter()
                single_payload = compute_single(index)
                per_structure.append(
                    {
                        "structure": index,
                        "atoms": len(structures[index]),
                        "seconds": time.perf_counter() - start,
                        "features": int(np.asarray(single_payload["values"]).shape[1]),
                    }
                )
        if payload is not None and not (case_dir / "normalized.npz").exists():
            _write_payload(case_dir, payload)
        _close_runtime(runtime)
        return {
            "status": "ok",
            "phase_seconds": {
                "import": import_seconds,
                "input": input_seconds,
                "model": model_seconds,
                "kernel_samples": samples,
                "kernel_median": float(np.median(samples)),
                "per_structure": per_structure,
                "per_structure_status": "measured" if measure_per_structure else "skipped",
                "setup_total": time.perf_counter() - total_start,
            },
            "shape": list(np.asarray(payload["values"]).shape) if payload is not None else [],
        }
    except Exception as exc:
        return {
            "status": "unavailable",
            "phase_seconds": {"setup_total": time.perf_counter() - total_start},
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }


def _median_phase(results: list[dict[str, Any]], name: str) -> float | None:
    values = [item.get("phase_seconds", {}).get(name) for item in results if item.get("status") == "ok"]
    values = [float(value) for value in values if value is not None]
    return float(np.median(values)) if values else None


def _load_normalized(case_dir: Path) -> np.ndarray | None:
    path = case_dir / "normalized.npz"
    if not path.exists():
        return None
    with np.load(path) as archive:
        return np.asarray(archive["values"], dtype=np.float64)


def _accuracy(left: np.ndarray, right: np.ndarray, offsets: np.ndarray, *, rtol: float, atol: float) -> dict[str, Any]:
    if left.shape != right.shape:
        return {"status": "incompatible", "left_shape": list(left.shape), "right_shape": list(right.shape)}
    delta = left - right
    absolute = np.abs(delta)
    denominator = np.maximum(np.abs(right), 1e-12)
    flat_left = left.reshape(-1)
    flat_right = right.reshape(-1)
    left_norm = float(np.linalg.norm(flat_left))
    right_norm = float(np.linalg.norm(flat_right))
    cosine = float(np.dot(flat_left, flat_right) / (left_norm * right_norm)) if left_norm and right_norm else None
    per_structure = []
    for index, (begin, end) in enumerate(zip(offsets[:-1], offsets[1:], strict=True)):
        block = delta[int(begin):int(end)]
        ref = right[int(begin):int(end)]
        per_structure.append(
            {
                "structure": index,
                "max_abs": float(np.max(np.abs(block), initial=0.0)),
                "rmse": float(np.sqrt(np.mean(block * block))) if block.size else 0.0,
                "mae": float(np.mean(np.abs(block))) if block.size else 0.0,
                "max_relative_abs": float(np.max(np.abs(block) / np.maximum(np.abs(ref), 1e-12), initial=0.0)),
            }
        )
    return {
        "status": "ok",
        "max_abs": float(np.max(absolute, initial=0.0)),
        "rmse": float(np.sqrt(np.mean(delta * delta))) if delta.size else 0.0,
        "mae": float(np.mean(absolute)) if delta.size else 0.0,
        "max_relative_abs": float(np.max(absolute / denominator, initial=0.0)),
        "cosine_similarity": cosine,
        "allclose": bool(np.allclose(left, right, rtol=rtol, atol=atol)),
        "rtol": rtol,
        "atol": atol,
        "per_structure": per_structure,
    }


def _case_record(case: dict[str, Any], case_dir: Path, cold: list[dict[str, Any]], warm: dict[str, Any]) -> dict[str, Any]:
    status = "ok" if any(item.get("status") == "ok" for item in cold) or warm.get("status") == "ok" else "unavailable"
    errors = [
        {key: item[key] for key in ("error_type", "error") if key in item}
        for item in cold + [warm]
        if item.get("status") == "unavailable"
    ]
    return {
        "id": case["id"],
        "package": case["package"],
        "descriptor": case["descriptor"],
        "adapter": case["adapter"],
        "model": _model_metadata(case),
        "reference": _reference_metadata(case),
        "parameters": case["parameters"],
        "comparison_group": case.get("comparison_group"),
        "tolerance": case.get("tolerance"),
        "status": status,
        "output": {
            "raw": str((case_dir / "raw.npz").relative_to(case_dir.parent.parent)) if (case_dir / "raw.npz").exists() else None,
            "normalized": str((case_dir / "normalized.npz").relative_to(case_dir.parent.parent)) if (case_dir / "normalized.npz").exists() else None,
            "metadata": str((case_dir / "output.json").relative_to(case_dir.parent.parent)) if (case_dir / "output.json").exists() else None,
        },
        "cold": {
            "repeats": len(cold),
            "successful_repeats": sum(item.get("status") == "ok" for item in cold),
            "median_phase_seconds": {name: _median_phase(cold, name) for name in ("import", "input", "model", "kernel", "total", "subprocess_wall_seconds")},
            "samples": cold,
        },
        "warm": warm,
        "errors": errors,
    }


def _select_cases(value: str | None) -> list[dict[str, Any]]:
    if not value:
        return [dict(item) for item in CASES]
    selected = {item.strip() for item in value.split(",") if item.strip()}
    unknown = selected - {item["id"] for item in CASES}
    if unknown:
        raise ValueError(f"unknown case(s): {', '.join(sorted(unknown))}")
    return [dict(item) for item in CASES if item["id"] in selected]


def _make_snapshot_dir(root: Path, dataset: Path, requested: str | None) -> tuple[Path, str]:
    dataset_id = dataset.stem
    base = root / dataset_id
    base.mkdir(parents=True, exist_ok=True)
    snapshot_id = requested or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    candidate = base / snapshot_id
    suffix = 1
    while candidate.exists():
        candidate = base / f"{snapshot_id}-{suffix:02d}"
        suffix += 1
    candidate.mkdir()
    return candidate, dataset_id


def _update_index(base: Path, snapshot_id: str, manifest: dict[str, Any]) -> None:
    index_path = base / "manifest.json"
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
    else:
        index = {"schema_version": 1, "dataset_id": base.name, "snapshots": []}
    index["snapshots"].append(
        {
            "snapshot": snapshot_id,
            "manifest": f"{snapshot_id}/manifest.json",
            "source_sha256": manifest["dataset"]["sha256"],
            "created_at": manifest["created_at"],
        }
    )
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _run_parent(args: argparse.Namespace) -> int:
    _set_thread_environment(args.threads)
    dataset = Path(args.dataset).resolve()
    if not dataset.exists():
        raise SystemExit(f"dataset does not exist: {dataset}")
    structures, batch = _load_dataset(dataset)
    snapshot, dataset_id = _make_snapshot_dir(Path(args.output_root).resolve(), dataset, args.snapshot_id)
    _save_canonical_input(snapshot, dataset, structures, batch)
    cases = _select_cases(args.cases)
    started = datetime.now(timezone.utc).isoformat()
    records = []
    for case in cases:
        print(f"[{case['id']}] cold {args.cold_repeats} subprocesses", flush=True)
        case_dir = snapshot / "cases" / case["id"]
        cold = _run_cold(case, dataset, case_dir, args.cold_repeats, args.threads)
        print(f"[{case['id']}] warm {args.warmup} + {args.repeat}", flush=True)
        if args.skip_warm or case.get("warm_policy") == "isolated_only":
            warm = {
                "status": "skipped",
                "phase_seconds": {},
                "reason": (
                    "explicitly skipped for a materialization-only run"
                    if args.skip_warm
                    else "reference backend is isolated because its legacy cleanup is not stable"
                ),
            }
        else:
            warm = _run_warm(
                case,
                dataset,
                case_dir,
                args.warmup,
                args.repeat,
                args.threads,
                measure_per_structure=not args.skip_per_structure,
            )
        records.append(_case_record(case, case_dir, cold, warm))

    by_group: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        group = record.get("comparison_group")
        if group:
            by_group.setdefault(group, []).append(record)
    comparisons = []
    offsets = np.asarray(batch.offsets, dtype=np.int64)
    for group, group_records in sorted(by_group.items()):
        if len(group_records) != 2:
            continue
        first, second = group_records
        left = _load_normalized(snapshot / "cases" / first["id"])
        right = _load_normalized(snapshot / "cases" / second["id"])
        comparison = {
            "group": group,
            "left": first["id"],
            "right": second["id"],
            "kind": "exact" if group.endswith("_exact") else "nominal_only",
        }
        if left is None or right is None:
            comparison["status"] = "unavailable"
        else:
            tolerance = first.get("tolerance") or second.get("tolerance") or {"rtol": 1e-8, "atol": 1e-8}
            comparison["metrics"] = _accuracy(left, right, offsets, rtol=float(tolerance["rtol"]), atol=float(tolerance["atol"]))
            if comparison["kind"] == "nominal_only":
                comparison["metrics"]["pass_fail_applicable"] = False
            else:
                comparison["metrics"]["pass_fail_applicable"] = True
                comparison["status"] = "pass" if comparison["metrics"].get("allclose") else "fail"
        comparisons.append(comparison)

    manifest = {
        "schema_version": 1,
        "created_at": started,
        "snapshot": snapshot.name,
        "dataset_id": dataset_id,
        "dataset": _dataset_metadata(dataset, structures, batch),
        "protocol": {
            **PROTOCOL,
            "cold_subprocesses": args.cold_repeats,
            "warmup_calls": args.warmup,
            "warm_measurements": args.repeat,
            "threads": args.threads,
            "per_structure_latencies": not args.skip_per_structure,
            "warm_skipped": args.skip_warm,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "numpy": np.__version__,
            "ase": _version("ase"),
            "featomic": _version("featomic"),
            "dscribe": _version("dscribe"),
            "nep_adapters": _version("nep-adapters"),
            "pyxtal_ff": _version("PyXtal_FF"),
            "deepmd_kit": _version("deepmd-kit"),
            "mdescriptor": _version("MDescriptor"),
            "torch": _version("torch"),
            "e3nn": _version("e3nn"),
            "vesin": _version("vesin"),
            "vesin_torch": _version("vesin-torch"),
            "mpich": _version("mpich"),
        },
        "cases": records,
        "comparisons": comparisons,
        "notes": [
            "Raw source and canonical input are copied into input/.",
            "Raw package outputs are kept separately from normalized float64 outputs.",
            "Exact pass/fail is emitted only for NEP/DPA model pairs; nominal descriptor families are metrics-only.",
            "C00PS uses the VASP 6.6.0-aligned reference build; other descriptor families remain separate comparisons.",
        ],
    }
    (snapshot / "manifest.json").write_text(json.dumps(_json_safe(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _update_index(snapshot.parent, snapshot.name, manifest)
    (snapshot / "summary.json").write_text(json.dumps({"snapshot": snapshot.name, "cases": records, "comparisons": comparisons}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"saved={snapshot / 'manifest.json'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-root", type=Path, default=ROOT / "benchmarks" / "baselines")
    parser.add_argument("--snapshot-id")
    parser.add_argument("--cases", help="comma-separated case IDs; defaults to all cases")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--cold-repeats", type=int, default=PROTOCOL["cold_subprocesses"])
    parser.add_argument("--warmup", type=int, default=PROTOCOL["warmup_calls"])
    parser.add_argument("--repeat", type=int, default=PROTOCOL["warm_measurements"])
    parser.add_argument(
        "--skip-per-structure",
        action="store_true",
        help="skip the 450 individual-structure timings; records this in the protocol",
    )
    parser.add_argument(
        "--skip-warm",
        action="store_true",
        help="only materialize cold output; records warm timing as explicitly skipped",
    )
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--case-id")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    if args.threads <= 0 or args.cold_repeats <= 0 or args.warmup < 0 or args.repeat <= 0:
        parser.error("threads/cold-repeats/repeat must be positive and warmup must be non-negative")
    if args.worker:
        if not args.case_id:
            parser.error("--worker requires --case-id")
        return _worker_main(args)
    return _run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
