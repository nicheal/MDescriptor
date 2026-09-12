"""Small, descriptor-independent helpers for external numerical references.

Imported both as ``scripts.external_reference`` (from the test suite, with the
repository root on ``sys.path``) and as top-level ``external_reference`` (by
sibling scripts run directly from ``scripts/``).  Everything here must stay
importable with only NumPy and the standard library; the project package is
imported lazily inside the helpers that need it.
"""

from __future__ import annotations

import hashlib
import importlib
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from mdescriptor import StructureBatch

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Shared Featomic block-flattening layer for the pinned featomic==0.6.6 water
# fixture.  Used by both the reference tests and the static golden generator.
_SPECIES = (1, 8)
_MAX_RADIAL = 2
_MAX_ANGULAR = 2
_RADIAL_COUNT = _MAX_RADIAL + 1


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of *path* without loading it all at once."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _block_data(block: Any) -> tuple[np.ndarray, np.ndarray]:
    samples = np.asarray(block.samples.values, dtype=np.int64)
    raw_values = np.asarray(block.values, dtype=np.float64)
    width = int(np.prod(raw_values.shape[1:], dtype=np.int64))
    values = raw_values.reshape(samples.shape[0], width)
    return samples, values


def _keys_and_blocks(tensor_map: Any) -> dict[tuple[int, ...], tuple[np.ndarray, np.ndarray]]:
    return {
        tuple(int(value) for value in key): _block_data(tensor_map[key])
        for key in tensor_map.keys
    }


def _values_for_atom(data: tuple[np.ndarray, np.ndarray], atom: int, width: int) -> np.ndarray:
    samples, values = data
    matches = np.flatnonzero((samples[:, 0] == 0) & (samples[:, 1] == atom))
    if len(matches) == 0:
        return np.zeros(width, dtype=np.float64)
    if len(matches) != 1:
        raise AssertionError(f"expected one reference row for atom {atom}, got {len(matches)}")
    return values[matches[0]]


def _flatten_atomic_composition(tensor_map: Any, atom_count: int, *, per_system: bool) -> np.ndarray:
    blocks = _keys_and_blocks(tensor_map)
    if per_system:
        result = np.zeros((1, len(_SPECIES)), dtype=np.float64)
        for column, species in enumerate(_SPECIES):
            samples, values = blocks[(species,)]
            if samples.shape != (1, 1):
                raise AssertionError(f"unexpected per-system samples for {species}: {samples.shape}")
            result[0, column] = values[0, 0]
        return result

    result = np.zeros((atom_count, len(_SPECIES)), dtype=np.float64)
    for column, species in enumerate(_SPECIES):
        samples, values = blocks[(species,)]
        for sample, value in zip(samples, values, strict=True):
            result[int(sample[1]), column] = value[0]
    return result


def _flatten_sorted_distances(tensor_map: Any, center_species: np.ndarray) -> np.ndarray:
    blocks = _keys_and_blocks(tensor_map)
    result = np.zeros((len(center_species), len(_SPECIES) * 4), dtype=np.float64)
    for atom, center in enumerate(center_species):
        offset = 0
        for neighbor in _SPECIES:
            data = blocks.get(
                (int(center), neighbor),
                (np.empty((0, 2)), np.empty((0, 4))),
            )
            result[atom, offset : offset + 4] = _values_for_atom(data, atom, 4)
            offset += 4
    return result


def _flatten_spherical_expansion(tensor_map: Any, atom_count: int) -> np.ndarray:
    blocks = _keys_and_blocks(tensor_map)
    group_width = sum((2 * angular + 1) * _RADIAL_COUNT for angular in range(_MAX_ANGULAR + 1))
    result = np.zeros((atom_count, len(_SPECIES) * len(_SPECIES) * group_width), dtype=np.float64)
    offset = 0
    for center in _SPECIES:
        for neighbor in _SPECIES:
            for angular in range(_MAX_ANGULAR + 1):
                candidates = [
                    (key, data)
                    for key, data in blocks.items()
                    if key[0] == angular and key[2:] == (center, neighbor)
                ]
                if len(candidates) > 1:
                    raise AssertionError(f"multiple Featomic blocks for {(angular, center, neighbor)}")
                width = (2 * angular + 1) * _RADIAL_COUNT
                data = candidates[0][1] if candidates else (np.empty((0, 2)), np.empty((0, width)))
                if data[1].shape[1] != width:
                    raise AssertionError(f"unexpected spherical block width: {data[1].shape[1]} != {width}")
                for atom in range(atom_count):
                    start = offset + (2 * angular + 1) * _RADIAL_COUNT
                    result[atom, start - width : start] = _values_for_atom(data, atom, width)
                offset += width
    return result


def _flatten_power_spectrum(tensor_map: Any, atom_count: int) -> np.ndarray:
    blocks = _keys_and_blocks(tensor_map)
    group_width = (_MAX_ANGULAR + 1) * _RADIAL_COUNT * _RADIAL_COUNT
    group_count = len(_SPECIES) * (len(_SPECIES) + 1) // 2
    result = np.zeros((atom_count, len(_SPECIES) * group_count * group_width), dtype=np.float64)
    offset = 0
    for center in _SPECIES:
        for first_index, first in enumerate(_SPECIES):
            for second in _SPECIES[first_index:]:
                data = blocks.get(
                    (center, first, second),
                    (np.empty((0, 2)), np.empty((0, group_width))),
                )
                if data[1].shape[1] != group_width:
                    raise AssertionError(f"unexpected power-spectrum block width: {data[1].shape[1]}")
                for atom in range(atom_count):
                    result[atom, offset : offset + group_width] = _values_for_atom(
                        data, atom, group_width
                    )
                offset += group_width
    return result


def _flatten_radial_spectrum(tensor_map: Any, atom_count: int) -> np.ndarray:
    blocks = _keys_and_blocks(tensor_map)
    result = np.zeros((atom_count, len(_SPECIES) * len(_SPECIES) * _RADIAL_COUNT), dtype=np.float64)
    offset = 0
    for center in _SPECIES:
        for neighbor in _SPECIES:
            data = blocks.get(
                (center, neighbor),
                (np.empty((0, 2)), np.empty((0, _RADIAL_COUNT))),
            )
            if data[1].shape[1] != _RADIAL_COUNT:
                raise AssertionError(f"unexpected radial-spectrum block width: {data[1].shape[1]}")
            for atom in range(atom_count):
                result[atom, offset : offset + _RADIAL_COUNT] = _values_for_atom(
                    data, atom, _RADIAL_COUNT
                )
            offset += _RADIAL_COUNT
    return result


def _flatten_spherical_expansion_by_pair(tensor_map: Any, atomic_numbers: np.ndarray) -> np.ndarray:
    blocks = _keys_and_blocks(tensor_map)
    group_width = sum((2 * angular + 1) * _RADIAL_COUNT for angular in range(_MAX_ANGULAR + 1))
    result = np.zeros((len(atomic_numbers), len(atomic_numbers), group_width), dtype=np.float64)
    for first_atom, first_type in enumerate(atomic_numbers):
        for second_atom, second_type in enumerate(atomic_numbers):
            offset = 0
            for angular in range(_MAX_ANGULAR + 1):
                width = (2 * angular + 1) * _RADIAL_COUNT
                data = blocks.get(
                    (angular, 1, int(first_type), int(second_type)),
                    (np.empty((0, 6)), np.empty((0, width))),
                )
                samples, values = data
                matches = np.flatnonzero(
                    (samples[:, 0] == 0)
                    & (samples[:, 1] == first_atom)
                    & (samples[:, 2] == second_atom)
                )
                if len(matches) == 0 and first_atom == second_atom and (angular > 0):
                    offset += width
                    continue
                if len(matches) != 1:
                    raise AssertionError(
                        f"expected one Featomic pair row for "
                        f"({first_atom}, {second_atom}, l={angular}), got {len(matches)}"
                    )
                result[first_atom, second_atom, offset : offset + width] = values[matches[0]]
                offset += width
    return result.reshape(len(atomic_numbers) * len(atomic_numbers), group_width)


def _portable(value: Any) -> Any:
    """Replace checkout-specific absolute paths in the generated manifest."""

    package_root = Path(importlib.import_module("mdescriptor").__file__).resolve().parent
    if isinstance(value, Path):
        value = str(value)
    if isinstance(value, str):
        package = str(package_root)
        root = str(PROJECT_ROOT)
        if value == package:
            return "${PACKAGE_ROOT}"
        if value.startswith(package + "/"):
            return "${PACKAGE_ROOT}/" + value[len(package) + 1 :]
        if value == root:
            return "${PROJECT_ROOT}"
        if value.startswith(root + "/"):
            return "${PROJECT_ROOT}/" + value[len(root) + 1 :]
        return value
    if isinstance(value, dict):
        return {str(key): _portable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_portable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _restore_paths(value: Any, package_root: Path | None = None) -> Any:
    """Expand ``${PACKAGE_ROOT}``/``${PROJECT_ROOT}`` manifest placeholders.

    ``package_root`` defaults to the installed ``mdescriptor`` package,
    resolved lazily so wheel-verification can point it at an install tree.
    """

    if isinstance(value, str):
        if package_root is None:
            import mdescriptor

            package_root = Path(mdescriptor.__file__).resolve().parent
        if value.startswith("${PACKAGE_ROOT}/"):
            return str(Path(package_root) / value.removeprefix("${PACKAGE_ROOT}/"))
        if value.startswith("${PROJECT_ROOT}/"):
            return str(PROJECT_ROOT / value.removeprefix("${PROJECT_ROOT}/"))
    if isinstance(value, dict):
        return {key: _restore_paths(item, package_root) for key, item in value.items()}
    if isinstance(value, list):
        return [_restore_paths(item, package_root) for item in value]
    return value


def _batch_from_npz(path: Path, ids: tuple[str, ...]) -> StructureBatch:
    from mdescriptor import StructureBatch

    with np.load(path) as arrays:
        return StructureBatch(
            np.asarray(arrays["numbers"], dtype=np.int32),
            np.asarray(arrays["positions"], dtype=np.float64),
            np.asarray(arrays["cells"], dtype=np.float64),
            np.asarray(arrays["pbc"], dtype=np.int32),
            np.asarray(arrays["offsets"], dtype=np.int64),
            ids,
        )


def _single_structure(batch: StructureBatch, index: int) -> StructureBatch:
    from mdescriptor import StructureBatch

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


def assert_result_matches(
    result: Any,
    expected: dict[str, Any],
    arrays: Any,
    tolerance: dict[str, float],
    *,
    context: str = "",
) -> None:
    """Check one computed result against the golden manifest and output NPZ.

    This is the single check list (values/samples/level/feature_count/labels/
    structure_ids/row_offsets) shared by the pytest suite (``tests/_golden.py``)
    and the pytest-free wheel verifier (``scripts/verify_wheel.py``); it must
    therefore stay importable with only NumPy.  ``context`` names the failing
    fixture in error messages.
    """

    prefix = f"{context}: " if context else ""
    np.testing.assert_allclose(
        np.asarray(result.values),
        arrays["values"],
        rtol=tolerance["rtol"],
        atol=tolerance["atol"],
        err_msg=prefix or None,
    )
    np.testing.assert_array_equal(result.samples, arrays["samples"], err_msg=prefix or None)
    if result.level.value != expected["level"]:
        raise AssertionError(f"{prefix}level changed")
    if result.feature_count != expected["feature_count"]:
        raise AssertionError(f"{prefix}feature count changed")
    if result.labels != tuple(expected["labels"]):
        raise AssertionError(f"{prefix}labels changed")
    if result.structure_ids != tuple(expected["structure_ids"]):
        raise AssertionError(f"{prefix}structure ids changed")
    expected_offsets = expected["row_offsets"]
    if expected_offsets is None:
        if result.row_offsets is not None:
            raise AssertionError(f"{prefix}row offsets changed")
    else:
        np.testing.assert_array_equal(result.row_offsets, expected_offsets, err_msg=prefix or None)


def external_c00ps_project_columns(
    *,
    species_count: int,
    radial_counts: tuple[int, ...],
    include_radial: bool,
) -> np.ndarray:
    """Map project C00PS columns to an external reference column order.

    The external reference constructs ordered species pairs outside the
    angular and radial loops::

        JNTYP0, JJNTYP0, L, IRB, JRB=IRB:NRB2(L)

    The public project representation instead flattens species/radial
    channels for each ``L`` and keeps the upper triangle.  Mixed-radial
    channels whose radial indices are reversed therefore live in the
    reversed ordered-species block in the external reference.
    """

    if species_count <= 0:
        raise ValueError("species_count must be positive")
    if not radial_counts or any(count <= 0 for count in radial_counts):
        raise ValueError("radial_counts must contain positive values")

    radial_features = species_count * radial_counts[0]
    per_species_pair = sum(count * (count + 1) // 2 for count in radial_counts)
    indices: list[int] = []
    if include_radial:
        indices.extend(range(radial_features))

    degree_offsets: list[int] = []
    running = 0
    for count in radial_counts:
        degree_offsets.append(running)
        running += count * (count + 1) // 2

    for degree, count in enumerate(radial_counts):
        channels = species_count * count
        for first in range(channels):
            first_species, first_radial = divmod(first, count)
            for second in range(first, channels):
                second_species, second_radial = divmod(second, count)
                if first_radial <= second_radial:
                    outer_species = first_species
                    inner_species = second_species
                    left_radial = first_radial
                    right_radial = second_radial
                else:
                    outer_species = second_species
                    inner_species = first_species
                    left_radial = second_radial
                    right_radial = first_radial

                triangular = (
                    left_radial * count
                    - left_radial * (left_radial - 1) // 2
                    + right_radial
                    - left_radial
                )
                ordered_pair = outer_species * species_count + inner_species
                indices.append(
                    radial_features
                    + ordered_pair * per_species_pair
                    + degree_offsets[degree]
                    + triangular
                )
    return np.asarray(indices, dtype=np.int64)


def align_external_c00ps(
    raw_values: np.ndarray,
    *,
    species_count: int,
    radial_counts: tuple[int, ...],
    include_radial: bool,
) -> np.ndarray:
    """Select and order raw external columns as MDescriptor public columns."""

    raw = np.asarray(raw_values, dtype=np.float64)
    if raw.ndim != 2:
        raise ValueError("raw external values must be a two-dimensional array")
    columns = external_c00ps_project_columns(
        species_count=species_count,
        radial_counts=radial_counts,
        include_radial=include_radial,
    )
    if columns.size and int(columns.max()) >= raw.shape[1]:
        raise ValueError(
            f"raw external output has {raw.shape[1]} columns, mapping needs column {int(columns.max())}"
        )
    return raw[:, columns]
