"""Symmetry and determinism checks for the CUDA NEP descriptor."""

from __future__ import annotations

import importlib
import os
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.io import read

import mdescriptor
from mdescriptor import ExecutionOptions, StructureBatch
from mdescriptor.descriptors import NEP
from mdescriptor.models import NEP_MODEL


def _load_cuda_plugin() -> None:
    try:
        importlib.import_module("mdescriptor._cuda")
        return
    except (ImportError, OSError):
        pass

    configured = os.environ.get("MDESCRIPTOR_CUDA_PLUGIN_DIR")
    candidates = [Path(configured)] if configured else []
    candidates.append(Path(__file__).parents[2] / "build-cuda")
    for candidate in candidates:
        if not any(candidate.glob("_cuda*.so")):
            continue
        candidate_text = str(candidate)
        if candidate_text not in mdescriptor.__path__:
            mdescriptor.__path__.insert(0, candidate_text)
        try:
            importlib.import_module("mdescriptor._cuda")
        except (ImportError, OSError):
            continue
        return
    pytest.skip("CUDA plugin is not installed in this test environment")


def _water(*, periodic: bool) -> Atoms:
    positions = np.asarray(
        [
            [8.0, 8.0, 8.0],
            [8.0 + 0.9572 * np.sin(np.deg2rad(104.52 / 2.0)), 8.0 + 0.9572 * np.cos(np.deg2rad(104.52 / 2.0)), 8.0],
            [8.0 - 0.9572 * np.sin(np.deg2rad(104.52 / 2.0)), 8.0 + 0.9572 * np.cos(np.deg2rad(104.52 / 2.0)), 8.0],
        ]
    )
    return Atoms(
        "OHH",
        positions=positions,
        cell=np.diag([16.0, 16.0, 16.0]) if periodic else np.zeros((3, 3)),
        pbc=periodic,
    )


def _rotation() -> np.ndarray:
    axis = np.asarray([1.0, 2.0, 3.0])
    axis /= np.linalg.norm(axis)
    angle = np.deg2rad(37.0)
    cross = np.asarray(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]]
    )
    return (
        np.eye(3) * np.cos(angle)
        + (1.0 - np.cos(angle)) * np.outer(axis, axis)
        + np.sin(angle) * cross
    )


def _transformations(reference: Atoms) -> tuple[list[Atoms], np.ndarray]:
    rotation = _rotation()
    center = np.mean(reference.positions, axis=0)
    rotated = reference.copy()
    rotated.positions = (reference.positions - center) @ rotation.T + center
    if reference.pbc.any():
        rotated.cell = reference.cell @ rotation.T

    translated = reference.copy()
    translated.positions = reference.positions + np.asarray([0.271, -0.419, 0.337])

    permutation = np.arange(len(reference), dtype=np.int64)[::-1]
    permuted = reference[permutation]
    return [reference, rotated, translated, permuted], permutation


def _rows(result: object, structure: int, permutation: np.ndarray | None = None) -> np.ndarray:
    descriptor_result = result
    values = np.asarray(descriptor_result.values)
    start = int(descriptor_result.row_offsets[structure])
    stop = int(descriptor_result.row_offsets[structure + 1])
    rows = values[start:stop].copy()
    if permutation is not None:
        rows = rows[np.argsort(permutation)]
    return rows


def _difference(left: np.ndarray, right: np.ndarray) -> tuple[float, float, float, bool]:
    delta = np.abs(left - right)
    maximum = float(np.max(delta, initial=0.0))
    scale = max(float(np.max(np.abs(left), initial=0.0)), float(np.max(np.abs(right), initial=0.0)), 1.0e-12)
    relative = maximum / scale
    # CUDA symmetry is checked at float32-scale noise.  The stricter model
    # parity tolerance is still printed below as a diagnostic for regressions.
    tolerance = 1.0e-6 + 1.0e-5 * np.abs(right)
    strict_tolerance = 1.0e-7 + 1.0e-6 * np.abs(right)
    strict_ratio = float(np.max(delta / strict_tolerance, initial=0.0))
    return maximum, relative, strict_ratio, bool(np.all(delta <= tolerance))


@pytest.mark.gpu
@pytest.mark.model
def test_cuda_nep_rotation_translation_permutation_and_determinism(capsys: pytest.CaptureFixture[str]) -> None:
    """CUDA NEP preserves invariance for periodic and isolated structures."""

    _load_cuda_plugin()
    carbon = read(
        Path(__file__).parents[2] / "benchmarks/_datasets/legacy/carbon_dataset_pbc.xyz",
        index=34,
    )
    cases = {
        "water-periodic": _water(periodic=True),
        "water-isolated": _water(periodic=False),
        "carbon-periodic": carbon,
    }
    descriptor = NEP(model=NEP_MODEL, execution=ExecutionOptions(device="cuda"))
    report: list[tuple[str, str, float, float, float, bool]] = []
    try:
        for name, reference in cases.items():
            systems, permutation = _transformations(reference)
            batch = StructureBatch.from_ase(systems)
            result = descriptor.compute(batch)
            repeated = descriptor.compute(batch)
            np.testing.assert_array_equal(
                result.values,
                repeated.values,
                err_msg=f"CUDA NEP is nondeterministic for {name}",
            )
            reference_rows = _rows(result, 0)
            for transform_name, structure_index, restore_permutation in (
                ("rotation", 1, None),
                ("translation", 2, None),
                ("permutation", 3, permutation),
            ):
                maximum, relative, ratio, passed = _difference(
                    reference_rows,
                    _rows(result, structure_index, restore_permutation),
                )
                report.append((name, transform_name, maximum, relative, ratio, passed))
    finally:
        descriptor.close()

    print("\nCUDA NEP symmetry report")
    print("| Case | Transform | Max abs | Relative | Strict model ratio | Pass* |")
    print("|---|---|---:|---:|---:|:---:|")
    for name, transform, maximum, relative, ratio, passed in report:
        print(f"| {name} | {transform} | {maximum:.3e} | {relative:.3e} | {ratio:.3f} | {'yes' if passed else 'no'} |")
    print("* Pass uses CUDA symmetry tolerance atol=1e-6, rtol=1e-5; strict model ratio uses atol=1e-7, rtol=1e-6.")
    captured = capsys.readouterr()
    print(captured.out, end="")
    assert all(item[-1] for item in report)
