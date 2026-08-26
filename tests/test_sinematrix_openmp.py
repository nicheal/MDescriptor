"""Numerical stability checks for the native SineMatrix OpenMP path."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parents[1]


def _compute_in_subprocess(threads: int) -> np.ndarray:
    script = r"""
import json
import numpy as np

from mdescriptor import StructureBatch
from mdescriptor.descriptors import SineMatrix

rng = np.random.default_rng(1729)
count = 48
cell_lengths = np.asarray([10.0, 11.0, 12.0])
batch = StructureBatch(
    np.resize(np.asarray([6, 8, 14, 26], dtype=np.int32), count),
    rng.random((count, 3)) * cell_lengths,
    np.diag(cell_lengths)[None, :, :],
    np.ones((1, 3), dtype=np.int32),
    np.asarray([0, count], dtype=np.int64),
    ("sinematrix-openmp",),
)
descriptor = SineMatrix(n_atoms_max=count, permutation="none")
try:
    values = descriptor.compute(batch).values
finally:
    descriptor.close()
print(json.dumps(values.tolist(), separators=(",", ":")))
"""
    environment = os.environ.copy()
    environment.update({
        "OMP_DYNAMIC": "FALSE",
        "OMP_NUM_THREADS": str(threads),
    })
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    return np.asarray(json.loads(completed.stdout), dtype=np.float64)


def test_sinematrix_openmp_matches_single_thread_bitwise():
    single_thread = _compute_in_subprocess(1)
    parallel = _compute_in_subprocess(4)

    np.testing.assert_array_equal(parallel, single_thread)
