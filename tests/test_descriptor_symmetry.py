"""Symmetry checks for every descriptor in the public catalog.

The water molecule is deliberately placed in a large periodic cell so that
the transformations below probe the descriptor geometry rather than a cell
boundary.  Atom- and pair-level results are compared after restoring the
input ordering; this tests permutation equivariance of local outputs while
still reporting a single permutation-symmetry result for each descriptor.
"""

import numpy as np
from ase import Atoms

from tests._public import StructureBatch, builtin_registry

_SYMMETRY_RTOL = 1e-5
_SYMMETRY_ATOL = 1e-7


def _water_systems() -> tuple[list[Atoms], np.ndarray]:
    """Return reference, rotated, translated and H-swapped water systems."""
    cell = np.diag([16.0, 16.0, 16.0])
    center = np.asarray([8.0, 8.0, 8.0])
    bond_length = 0.9572
    half_angle = np.deg2rad(104.52 / 2.0)
    relative_positions = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [bond_length * np.sin(half_angle), bond_length * np.cos(half_angle), 0.0],
            [-bond_length * np.sin(half_angle), bond_length * np.cos(half_angle), 0.0],
        ]
    )
    positions = center + relative_positions

    # Rodrigues rotation around a deliberately non-coordinate axis.
    axis = np.asarray([1.0, 2.0, 3.0])
    axis /= np.linalg.norm(axis)
    angle = np.deg2rad(37.0)
    cross = np.asarray(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]]
    )
    rotation = (
        np.eye(3) * np.cos(angle)
        + (1.0 - np.cos(angle)) * np.outer(axis, axis)
        + np.sin(angle) * cross
    )

    rotated = center + relative_positions @ rotation.T
    rotated_cell = cell @ rotation.T
    translated = positions + np.asarray([0.271, -0.419, 0.337])
    swapped = positions[[0, 2, 1]]
    systems = [
        Atoms("OHH", positions=part, cell=part_cell, pbc=True)
        for part, part_cell in (
            (positions, cell),
            (rotated, rotated_cell),
            (translated, cell),
            (swapped, cell),
        )
    ]
    # ``permutation[new_index]`` gives the corresponding reference index.
    return systems, np.asarray([0, 2, 1])


def _calculator(name: str, calculator_class: type):
    """Use small, fixed configurations so the catalog test stays practical."""
    if name == "SOAP":
        return calculator_class(
            species=[1, 8], r_cut=3.0, n_max=2, l_max=2, sigma=0.4, average="off"
        )
    if name == "SOAPTurbo":
        return calculator_class(
            species=[1, 8], alpha_max=[2, 2], l_max=2,
            rcut_hard=3.0, rcut_soft=2.5, atom_sigma_r=0.4, atom_sigma_t=0.4,
        )
    if name == "ACSF":
        return calculator_class(
            species=[1, 8], r_cut=3.0,
            g2_params=[[1.0, 0.5]], g3_params=[1.0], g4_params=[[0.1, 1.0, 1.0]],
        )
    if name == "ACE":
        return calculator_class(species=[1, 8], N=2, maxdeg=4, rcut=3.0)
    if name in {"CoulombMatrix", "SineMatrix", "EwaldSumMatrix"}:
        return calculator_class(n_atoms_max=3)
    if name in {"MBTR", "LMBTR"}:
        return calculator_class(
            species=[1, 8],
            geometry={"function": "distance"},
            grid={"min": 0.0, "max": 3.5, "n": 20, "sigma": 0.05},
            weighting={"function": "exp", "scale": 0.5, "threshold": 1e-3},
        )
    if name == "ValleOganov":
        return calculator_class(species=[1, 8], function="distance", n=20, sigma=0.05, r_cut=3.5)
    if name == "AtomicComposition":
        return calculator_class(species=[1, 8])
    if name == "NeighborList":
        return calculator_class(cutoff=3.0)
    if name == "SortedDistances":
        return calculator_class(species=[1, 8], cutoff=3.0, max_neighbors=6)
    if name in {
        "SphericalExpansion", "SphericalExpansionByPair", "SoapRadialSpectrum",
        "SoapPowerSpectrum", "LodeSphericalExpansion",
    }:
        return calculator_class(
            species=[1, 8], cutoff=3.0, density_width=0.4,
            max_radial=2, max_angular=2,
        )
    if name == "EAD":
        return calculator_class(parameters={"L": 2, "eta": [0.05, 0.1], "Rs": [0.0, 0.5]}, Rc=3.0)
    if name == "SO3":
        return calculator_class(nmax=2, lmax=2, rcut=3.0)
    if name == "SO4":
        return calculator_class(lmax=2, rcut=3.0, normalize_U=True)
    if name == "SNAP":
        return calculator_class(weights={1: 1.0, 8: 1.0}, lmax=2, rcut=3.0)
    if name == "LBispectrum":
        return calculator_class(twojmax=3, diagonal=1, rcut=3.0)
    if name == "MTP":
        return calculator_class(
            species=[1, 8], min_dist=0.1, max_dist=3.0,
            radial_basis_size=2, max_rank=2,
        )
    if name == "C00PSMLFF":
        return calculator_class(species=[1, 8], r_cut=3.0, n_radial=2, l_max=2)
    if name in {"NEP", "DPA4", "DPA4C"}:
        return calculator_class()
    raise AssertionError(f"no water configuration for catalog descriptor {name!r}")


def _structure_rows(result, structure: int, permutation: np.ndarray | None = None) -> np.ndarray:
    """Return rows in a common order for atom, structure and pair outputs."""
    values = np.asarray(result.values)
    if result.level == "structure":
        return values[structure : structure + 1]

    start = int(result.row_offsets[structure])
    stop = int(result.row_offsets[structure + 1])
    rows = values[start:stop].copy()
    if result.level == "atom":
        if permutation is not None:
            # The transformed result is in new atom order; restore reference order.
            rows = rows[np.argsort(permutation)]
        return rows

    # Pair outputs can be emitted in a different traversal order after a
    # translation or atom permutation.  Sort on the canonical sample identity.
    keys = np.asarray(result.samples)[start:stop, 1:].copy()
    if permutation is not None:
        keys[:, :2] = permutation[keys[:, :2].astype(np.int64)]
    order = np.lexsort(tuple(keys[:, column] for column in reversed(range(keys.shape[1]))))
    return rows[order]


def _difference(
    left: np.ndarray,
    right: np.ndarray,
    *,
    rtol: float = _SYMMETRY_RTOL,
    atol: float = _SYMMETRY_ATOL,
) -> tuple[float, float, bool]:
    if left.shape != right.shape:
        return float("inf"), float("inf"), False
    delta = np.abs(left - right)
    maximum = float(np.max(delta)) if delta.size else 0.0
    scale = max(float(np.max(np.abs(left))) if left.size else 0.0,
                float(np.max(np.abs(right))) if right.size else 0.0, 1e-12)
    relative = maximum / scale
    equivalent = bool(np.allclose(left, right, rtol=rtol, atol=atol))
    return maximum, relative, equivalent


def _format_difference(value: tuple[float, float, bool]) -> str:
    maximum, relative, _ = value
    return f"{maximum:.3e} ({relative:.3e})"


def _print_report(report: list[dict[str, object]]) -> None:
    print("\nWater-molecule descriptor symmetry report")
    print("(entries are max absolute difference; relative difference is in parentheses)")
    print(
        "| Descriptor | Level | Rotation Δ | Rotation | Translation Δ | Translation "
        "| H-coordinate swap Δ | H swap |"
    )
    print("|---|---|---:|:---:|---:|:---:|---:|:---:|")
    for row in report:
        print(
            f"| {row['name']} | {row['level']} | {row['rotation_delta']} | "
            f"{'yes' if row['rotation_ok'] else 'no'} | {row['translation_delta']} | "
            f"{'yes' if row['translation_ok'] else 'no'} | {row['permutation_delta']} | "
            f"{'yes' if row['permutation_ok'] else 'no'} |"
        )


def test_all_descriptors_on_single_water_symmetry_report():
    systems, permutation = _water_systems()
    batch = StructureBatch.from_ase(systems)
    report = []
    skipped = []

    expected_non_rotational = {
        "NeighborList", "SphericalExpansion", "SphericalExpansionByPair", "LodeSphericalExpansion"
    }
    descriptors = tuple((spec.name, spec.load_class()) for spec in builtin_registry)
    for name, calculator_class in descriptors:
        calculator = _calculator(name, calculator_class)
        try:
            result = calculator.compute(batch)
            assert np.isfinite(np.asarray(result.values)).all(), name
            reference = _structure_rows(result, 0)
            tolerance = {"rtol": 1e-5, "atol": 3e-5} if name in {"DPA4", "DPA4C"} else {}
            rotation = _difference(reference, _structure_rows(result, 1), **tolerance)
            translation = _difference(reference, _structure_rows(result, 2), **tolerance)
            permutation_result = _difference(
                reference, _structure_rows(result, 3, permutation=permutation), **tolerance
            )
            report.append(
                {
                    "name": name,
                    "level": result.level,
                    "rotation_delta": _format_difference(rotation),
                    "rotation_ok": rotation[2],
                    "translation_delta": _format_difference(translation),
                    "translation_ok": translation[2],
                    "permutation_delta": _format_difference(permutation_result),
                    "permutation_ok": permutation_result[2],
                }
            )
        finally:
            calculator.close()

    _print_report(report)
    assert {row["name"] for row in report} | {item.split(":", 1)[0] for item in skipped} == {
        name for name, _ in descriptors
    }
    assert all(row["translation_ok"] for row in report)
    assert all(row["permutation_ok"] for row in report)
    assert {
        row["name"] for row in report if not row["rotation_ok"]
    } == expected_non_rotational
