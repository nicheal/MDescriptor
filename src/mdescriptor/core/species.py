"""Canonical species declarations shared by descriptor families."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

import numpy as np

from .errors import DescriptorConfigError, DescriptorInputError

if TYPE_CHECKING:  # pragma: no cover
    from .input import StructureBatch


def normalize_species(species: Iterable[int] | None) -> tuple[int, ...] | None:
    """Validate and freeze an optional atomic-number declaration."""

    if species is None:
        return None
    try:
        raw_values = tuple(species)
    except TypeError as exc:
        raise DescriptorConfigError("species must be a sequence of atomic numbers") from exc
    if any(isinstance(value, (bool, np.bool_)) for value in raw_values):
        raise DescriptorConfigError("species must contain integer atomic numbers")
    try:
        values = tuple(int(value) for value in raw_values)
    except (TypeError, ValueError, OverflowError) as exc:
        raise DescriptorConfigError("species must contain integer atomic numbers") from exc
    for raw, normalized in zip(raw_values, values, strict=True):
        if isinstance(raw, (float, np.floating)):
            if not np.isfinite(raw):
                raise DescriptorConfigError("species must contain finite integer atomic numbers")
            if float(raw) != normalized:
                raise DescriptorConfigError("species must contain integer atomic numbers")
    if not values or len(set(values)) != len(values):
        raise DescriptorConfigError("species must be a non-empty sequence of unique atomic numbers")
    if any(value <= 0 for value in values):
        raise DescriptorConfigError("species must contain positive atomic numbers")
    return values


def require_species(species: Iterable[int] | None, *, descriptor: str) -> tuple[int, ...]:
    """Require a fixed species map at construction time."""

    normalized = normalize_species(species)
    if normalized is None:
        raise DescriptorConfigError(
            f"{descriptor} requires an explicit species declaration at construction"
        )
    return normalized


def species_from_batch(batch: StructureBatch) -> tuple[int, ...]:
    """Return the deterministic atomic-number set present in a batch."""

    values = tuple(int(value) for value in np.unique(batch.numbers))
    if not values:
        raise DescriptorInputError("the input batch contains no atoms")
    return values


def validate_batch_species(
    batch: StructureBatch, species: Iterable[int], *, descriptor: str
) -> tuple[int, ...]:
    """Validate that a batch does not contain an undeclared element."""

    normalized = require_species(species, descriptor=descriptor)
    missing = set(np.unique(batch.numbers)) - set(normalized)
    if missing:
        raise DescriptorInputError(
            f"{descriptor} input contains species outside its declaration: {sorted(missing)}"
        )
    return normalized


__all__ = [
    "normalize_species",
    "require_species",
    "species_from_batch",
    "validate_batch_species",
]
