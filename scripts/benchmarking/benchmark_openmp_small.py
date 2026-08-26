"""Compare one and four OpenMP threads on deterministic small batches.

This is an observation benchmark, not a performance gate: machine load and the
number of available cores make a fixed speed threshold unsuitable for CI.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any

for _thread_env in (
    "OMP_DYNAMIC",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_env] = "FALSE" if _thread_env == "OMP_DYNAMIC" else "1"

import numpy as np  # noqa: E402
from ase import Atoms  # noqa: E402

from mdescriptor import ExecutionOptions, StructureBatch  # noqa: E402
from mdescriptor.descriptors import (  # noqa: E402
    EAD,
    LMBTR,
    MBTR,
    SNAP,
    SO3,
    SO4,
    AtomicComposition,
    CoulombMatrix,
    EwaldSumMatrix,
    LBispectrum,
    NeighborList,
    SineMatrix,
    ValleOganov,
)


def _batch(structures: int = 8, atoms_per_structure: int = 16) -> StructureBatch:
    rng = np.random.default_rng(20260826)
    cell = np.diag([18.0, 18.0, 18.0])
    numbers = np.resize(np.asarray([1, 6, 8, 14], dtype=np.int32), atoms_per_structure)
    systems = []
    for _index in range(structures):
        positions = rng.random((atoms_per_structure, 3)) * np.diag(cell)
        systems.append(
            Atoms(
                numbers=numbers,
                positions=positions,
                cell=cell,
                pbc=True,
            )
        )
    return StructureBatch.from_ase(systems)


def _matrix_batch(atoms: int = 64) -> StructureBatch:
    rng = np.random.default_rng(20260827)
    cell = np.diag([24.0, 24.0, 24.0])
    return StructureBatch.from_ase(
        [
            Atoms(
                numbers=np.resize(np.asarray([1, 6, 8, 14], dtype=np.int32), atoms),
                positions=rng.random((atoms, 3)) * np.diag(cell),
                cell=cell,
                pbc=True,
            )
        ]
    )


def _descriptor(name: str, threads: int) -> Any:
    execution = ExecutionOptions(num_threads=threads)
    species = [1, 6, 8, 14]
    if name == "CoulombMatrix":
        return CoulombMatrix(n_atoms_max=64, permutation="none", execution=execution)
    if name == "SineMatrix":
        return SineMatrix(n_atoms_max=64, permutation="none", execution=execution)
    if name == "EwaldSumMatrix":
        return EwaldSumMatrix(
            n_atoms_max=64,
            permutation="none",
            accuracy=1e-5,
            w=1.0,
            r_cut=6.0,
            g_cut=3.0,
            a=0.3,
            execution=execution,
        )
    if name in {"MBTR", "LMBTR"}:
        descriptor_type = MBTR if name == "MBTR" else LMBTR
        return descriptor_type(
            species=species,
            geometry={"function": "angle"},
            grid={"min": 0.0, "max": 180.0, "n": 24, "sigma": 0.2},
            weighting={"function": "smooth_cutoff", "r_cut": 4.0, "sharpness": 2.0},
            execution=execution,
        )
    if name == "ValleOganov":
        return ValleOganov(
            species=species,
            function="angle",
            n=24,
            sigma=0.2,
            r_cut=4.0,
            execution=execution,
        )
    if name == "AtomicComposition":
        return AtomicComposition(species=species, execution=execution)
    if name == "NeighborList":
        return NeighborList(cutoff=4.0, execution=execution)
    if name == "EAD":
        return EAD(
            parameters={"L": 2, "eta": [0.05, 0.1], "Rs": [0.0, 0.5]},
            Rc=4.0,
            execution=execution,
        )
    if name == "SO3":
        return SO3(nmax=2, lmax=2, rcut=4.0, execution=execution)
    if name == "SO4":
        return SO4(lmax=2, rcut=4.0, execution=execution)
    if name == "SNAP":
        return SNAP(lmax=2, rcut=4.0, execution=execution)
    if name == "LBispectrum":
        return LBispectrum(twojmax=4, diagonal=2, rcut=4.0, execution=execution)
    raise ValueError(f"unknown descriptor: {name}")


TARGETS = (
    "CoulombMatrix",
    "SineMatrix",
    "EwaldSumMatrix",
    "MBTR",
    "LMBTR",
    "ValleOganov",
    "AtomicComposition",
    "NeighborList",
    "EAD",
    "SO3",
    "SO4",
    "SNAP",
    "LBispectrum",
)


def _measure(
    descriptor: Any,
    batch: StructureBatch,
    warmup: int,
    repeat: int,
) -> tuple[float, np.ndarray]:
    for _ in range(warmup):
        descriptor.compute(batch)
    elapsed = []
    result = None
    for _ in range(repeat):
        started = time.perf_counter()
        result = descriptor.compute(batch)
        elapsed.append(time.perf_counter() - started)
    assert result is not None
    return float(np.median(elapsed)), np.asarray(result.values, dtype=np.float64)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    if args.warmup < 0 or args.repeat <= 0 or args.threads <= 1:
        raise SystemExit("warmup must be non-negative, repeat positive, and threads greater than one")

    batches = {name: _matrix_batch() if name.endswith("Matrix") else _batch() for name in TARGETS}
    measurements: list[dict[str, Any]] = []
    for name in TARGETS:
        serial = _descriptor(name, 1)
        threaded = _descriptor(name, args.threads)
        try:
            serial_seconds, serial_values = _measure(serial, batches[name], args.warmup, args.repeat)
            threaded_seconds, threaded_values = _measure(threaded, batches[name], args.warmup, args.repeat)
        finally:
            serial.close()
            threaded.close()
        measurements.append(
            {
                "descriptor": name,
                "serial_seconds": serial_seconds,
                "threaded_seconds": threaded_seconds,
                "speedup": serial_seconds / threaded_seconds,
                "max_abs_error": float(np.max(np.abs(serial_values - threaded_values))),
                "rows": int(serial_values.shape[0]),
                "features": int(serial_values.shape[1]),
            }
        )

    print(json.dumps({"threads": args.threads, "warmup": args.warmup, "repeat": args.repeat, "results": measurements}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
