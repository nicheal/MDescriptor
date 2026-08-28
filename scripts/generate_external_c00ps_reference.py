#!/usr/bin/env python3
"""Generate the C00PSMLFF golden from a local external reference library.

The licensed reference archive and derived library are user-supplied,
local-only inputs
under ``.deps`` and are never distributed by this repository.  This program
calls the local library through its C ABI and then performs only a column
permutation: the reference stores ordered element-pair blocks while the public
descriptor exposes one upper triangle of flattened channels.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import mmap
import os
import re
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from external_reference import align_external_c00ps, sha256

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXTURE = ROOT / "tests" / "golden" / "c00psmlff"
DEFAULT_ARCHIVE = ROOT / ".deps" / "external-mlff-reference.tgz"
DEFAULT_LIBRARY = ROOT / ".deps" / "external-mlff-reference.so"
MAPPER = ROOT / "scripts" / "external_reference.py"


class ExternalC00PS:
    """Direct binding to the local serial C API reference library."""

    def __init__(self, library: Path, parameters: dict[str, Any], api_prefix: str) -> None:
        self._library = ctypes.CDLL(str(library))
        setup = getattr(self._library, f"{api_prefix}_setup")
        setup.restype = ctypes.c_int
        setup.argtypes = [
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        compute = getattr(self._library, f"{api_prefix}_compute")
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
        nfeatures = getattr(self._library, f"{api_prefix}_nfeatures")
        nfeatures.restype = ctypes.c_int
        nfeatures.argtypes = [ctypes.c_void_p]

        if parameters["cutoff_function"] != "bp":
            raise ValueError("external reference currently supports cutoff_function='bp'")
        if float(parameters["radial_sigma"]) != 0.5:
            raise ValueError("external reference requires radial_sigma=0.5")
        if not parameters["exclude_self_interaction"]:
            raise ValueError("external reference requires self-interaction correction")
        if parameters["super_vector"]:
            raise ValueError("external raw C00PS reference does not expose super_vector")
        if float(parameters["radial_weight"]) != 1.0 or float(parameters["angular_weight"]) != 1.0:
            raise ValueError("external reference generator currently requires unit weights")

        iparams = (ctypes.c_int32 * 23)(
            0,
            int(parameters["n_radial"]),
            10000,
            2,
            1,
            0,
            int(bool(parameters["normalize_radial"])),
            0,
            0,
            0,
            int(parameters["l_max"]),
            int(parameters["n_radial"]),
            10000,
            2,
            1,
            0,
            int(bool(parameters["normalize_angular"])),
            0,
            0,
            0,
            0,
            0,
            0,
        )
        rparams = (ctypes.c_double * 7)(
            float(parameters["r_cut"]),
            float(parameters["r_cut"]),
            1.0,
            1.0,
            0.0,
            1.0,
            1.0,
        )
        self._context = ctypes.c_void_p()
        code = setup(
            iparams,
            rparams,
            ctypes.c_int32(len(parameters["species"])),
            ctypes.byref(self._context),
        )
        if code != 0:
            raise RuntimeError(f"external_reference_setup failed (rc={code})")
        self.n_features = int(nfeatures(self._context))
        if self.n_features <= 0:
            raise RuntimeError(f"external_reference_nfeatures returned {self.n_features}")
        self._compute = compute

    def _compute_into(
        self,
        cell: np.ndarray,
        positions: np.ndarray,
        types: np.ndarray,
        output: np.ndarray,
    ) -> None:
        cell = np.ascontiguousarray(cell, dtype=np.float64)
        positions = np.ascontiguousarray(positions, dtype=np.float64)
        types = np.ascontiguousarray(types, dtype=np.int32)
        code = self._compute(
            self._context,
            cell.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            positions.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            types.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            ctypes.c_int32(positions.shape[0]),
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_int32(self.n_features),
        )
        if code != 0:
            raise RuntimeError(f"external_reference_compute failed (rc={code})")

    def compute_isolated(
        self, cell: np.ndarray, positions: np.ndarray, types: np.ndarray
    ) -> np.ndarray:
        """Run one structure in a child to isolate legacy Fortran cleanup."""

        if not hasattr(os, "fork"):
            raise RuntimeError("the external reference requires os.fork")
        shape = (len(positions), self.n_features)
        byte_count = int(np.prod(shape) * np.dtype(np.float64).itemsize)
        shared = mmap.mmap(-1, byte_count, access=mmap.ACCESS_WRITE)
        output = np.ndarray(shape, dtype=np.float64, buffer=shared)
        output.fill(0.0)
        child = os.fork()
        if child == 0:
            try:
                self._compute_into(cell, positions, types, output)
            except BaseException:
                traceback.print_exc()
                os._exit(1)
            os._exit(0)
        _, status = os.waitpid(child, 0)
        try:
            if not os.WIFEXITED(status) or os.WEXITSTATUS(status) != 0:
                raise RuntimeError(f"isolated external-reference child failed (status={status})")
            return output.copy()
        finally:
            shared.close()


def _radial_counts(labels: list[str], l_max: int) -> tuple[int, ...]:
    counts = [0] * (l_max + 1)
    pattern = re.compile(r"n1=(\d+).*n2=(\d+),l=(\d+)$")
    for label in labels:
        match = pattern.search(label)
        if match is None:
            continue
        first, second, degree = (int(item) for item in match.groups())
        counts[degree] = max(counts[degree], first + 1, second + 1)
    if any(count == 0 for count in counts):
        raise ValueError(f"could not infer all external radial counts from labels: {counts}")
    return tuple(counts)


def _evaluate_library(
    library: Path,
    parameters: dict[str, Any],
    arrays: dict[str, np.ndarray],
    api_prefix: str,
) -> np.ndarray:
    reference = ExternalC00PS(library, parameters, api_prefix)
    species_to_type = {int(number): index + 1 for index, number in enumerate(parameters["species"])}
    blocks: list[np.ndarray] = []
    for structure, (first, last) in enumerate(
        zip(arrays["offsets"][:-1], arrays["offsets"][1:], strict=True)
    ):
        first = int(first)
        last = int(last)
        numbers = arrays["numbers"][first:last]
        try:
            types = np.asarray([species_to_type[int(number)] for number in numbers], dtype=np.int32)
        except KeyError as error:
            raise ValueError(f"fixture contains unconfigured species {error.args[0]}") from error
        cell = arrays["cells"][structure]
        if not bool(np.all(arrays["pbc"][structure])):
            # The source routine is periodic-only.  A 20-Angstrom box has no
            # image inside this fixture's cutoff, so it exactly represents
            # the requested isolated structure.
            cell = np.eye(3, dtype=np.float64) * 20.0
        blocks.append(reference.compute_isolated(cell, arrays["positions"][first:last], types))
    return np.concatenate(blocks, axis=0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--library", type=Path, default=DEFAULT_LIBRARY)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument(
        "--raw-input",
        type=Path,
        help="reuse a captured raw external output; intended for audited recovery only",
    )
    parser.add_argument("--accept", action="store_true", help="replace the committed golden")
    parser.add_argument("--api-prefix", required=True, help="prefix of the local C ABI symbols")
    args = parser.parse_args()

    fixture = args.fixture.resolve()
    archive = args.source_archive.resolve()
    library = args.library.resolve()
    manifest_path = fixture / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    parameters = manifest["configuration"]["parameters"]
    if not parameters["include_angular"]:
        raise ValueError("this generator currently requires include_angular=true")

    with np.load(fixture / manifest["input"]) as source:
        arrays = {name: np.asarray(source[name]) for name in source.files}
    if args.raw_input is None:
        raw = _evaluate_library(library, parameters, arrays, args.api_prefix)
        raw_provenance: dict[str, Any] = {"mode": "direct_library_evaluation"}
    else:
        raw_path = args.raw_input.resolve()
        with np.load(raw_path) as source:
            raw = np.asarray(source["values"], dtype=np.float64)
        raw_provenance = {
            "mode": "audited_raw_capture",
            "sha256": sha256(raw_path),
        }

    radial_counts = _radial_counts(manifest["result"]["labels"], int(parameters["l_max"]))
    aligned = align_external_c00ps(
        raw,
        species_count=len(parameters["species"]),
        radial_counts=radial_counts,
        include_radial=bool(parameters["include_radial"]),
    )

    from mdescriptor import DescriptorConfiguration, StructureBatch, create_descriptor

    batch = StructureBatch(
        np.asarray(arrays["numbers"], dtype=np.int32),
        np.asarray(arrays["positions"], dtype=np.float64),
        np.asarray(arrays["cells"], dtype=np.float64),
        np.asarray(arrays["pbc"], dtype=np.int32),
        np.asarray(arrays["offsets"], dtype=np.int64),
        tuple(manifest["input_ids"]),
    )
    descriptor = create_descriptor(DescriptorConfiguration.from_dict(manifest["configuration"]))
    try:
        current = descriptor.compute(batch)
    finally:
        descriptor.close()
    delta = np.abs(current.values - aligned)
    np.testing.assert_allclose(current.values, aligned, rtol=1e-8, atol=1e-8)

    manifest["reference"] = {
        "kind": "licensed_external_mlff_source",
        "source": "licensed external MLFF C00/PS reference routines",
        "distribution_boundary": (
            "source archive and derived reference library are local-only, "
            "user-supplied inputs and are not distributed"
        ),
        "source_archive": "local-only input (path intentionally not recorded)",
        "source_archive_sha256": sha256(archive),
        "library": "local-only input (path intentionally not recorded)",
        "library_sha256": sha256(library),
        "evaluator": str(Path(__file__).resolve().relative_to(ROOT)),
        "evaluator_sha256": sha256(Path(__file__).resolve()),
        "column_mapper": str(MAPPER.relative_to(ROOT)),
        "column_mapper_sha256": sha256(MAPPER),
        "raw_output": raw_provenance,
        "column_order": "ordered species pair, angular degree, radial upper triangle",
        "radial_counts": list(radial_counts),
        "nonperiodic_substitution": "20 Angstrom cubic cell; no image within r_cut",
        "verification": {
            "rtol": 1e-8,
            "atol": 1e-8,
            "max_abs": float(delta.max(initial=0.0)),
            "mae": float(delta.mean()) if delta.size else 0.0,
        },
    }
    manifest["tolerance"] = {"rtol": 1e-8, "atol": 1e-8}
    if args.accept:
        np.savez_compressed(
            fixture / manifest["expected_output"],
            values=aligned,
            samples=np.asarray(current.samples, dtype=np.int64),
        )
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(
            f"accepted external reference {aligned.shape}; "
            f"max_abs={delta.max(initial=0.0):.3e}, mae={delta.mean():.3e}"
        )
    else:
        print(
            f"verified external reference {aligned.shape}; "
            f"max_abs={delta.max(initial=0.0):.3e}, mae={delta.mean():.3e}; "
            "pass --accept to replace the golden"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
