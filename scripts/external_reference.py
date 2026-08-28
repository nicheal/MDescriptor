"""Small, descriptor-independent helpers for external numerical references."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of *path* without loading it all at once."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
