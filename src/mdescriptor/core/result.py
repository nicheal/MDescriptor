"""Common result container for atom, structure, and pair descriptors."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from collections.abc import Mapping
from typing import Any

import numpy as np


class DescriptorLevel(str, Enum):
    ATOM = "atom"
    STRUCTURE = "structure"
    PAIR = "pair"


@dataclass(frozen=True, slots=True)
class DescriptorResult:
    """Values plus the row and feature metadata needed to interpret them."""

    values: Any
    level: DescriptorLevel | str
    structure_ids: tuple[str, ...]
    row_offsets: np.ndarray | None
    labels: tuple[str, ...]
    metadata: dict[str, Any] = field(default_factory=dict)
    samples: Any = None
    feature_count: int | None = None

    def __post_init__(self) -> None:
        level = DescriptorLevel(self.level)
        object.__setattr__(self, "level", level)
        object.__setattr__(self, "structure_ids", tuple(str(item) for item in self.structure_ids))
        object.__setattr__(self, "labels", tuple(str(item) for item in self.labels))
        object.__setattr__(self, "metadata", _json_safe(self.metadata))
        if self.row_offsets is not None:
            object.__setattr__(
                self,
                "row_offsets",
                np.ascontiguousarray(self.row_offsets, dtype=np.int64),
            )
        feature_count = self.feature_count
        shape = tuple(getattr(self.values, "shape", ()))
        if len(shape) != 2:
            raise ValueError("descriptor values must be a two-dimensional array")
        if self.labels and len(self.labels) != int(shape[1]):
            raise ValueError(
                "labels must contain exactly one entry per feature column"
            )
        if feature_count is None:
            feature_count = int(shape[1]) if len(shape) >= 2 else None
        elif len(shape) >= 2 and int(feature_count) != int(shape[1]):
            raise ValueError("feature_count does not match the result feature dimension")
        elif feature_count < 0:
            raise ValueError("feature_count must be non-negative or None")
        rows = int(shape[0])
        offsets = self.row_offsets
        if level is DescriptorLevel.STRUCTURE:
            if offsets is not None:
                raise ValueError("structure-level results must not define row offsets")
            if len(self.structure_ids) != rows:
                raise ValueError(
                    "structure_ids must contain exactly one identifier per structure row"
                )
        else:
            if offsets is None:
                raise ValueError(f"{level.value}-level results require row offsets")
            if offsets.ndim != 1:
                raise ValueError("row offsets must be a one-dimensional integer array")
            if len(offsets) != len(self.structure_ids) + 1:
                raise ValueError(
                    "row offsets must contain one boundary per structure plus the end"
                )
            if len(offsets) == 0 or int(offsets[0]) != 0:
                raise ValueError("row offsets must start at zero")
            if np.any(np.diff(offsets) < 0):
                raise ValueError("row offsets must be monotonically non-decreasing")
            if int(offsets[-1]) != rows:
                raise ValueError(
                    "the final row offset must equal the number of result rows"
                )
        object.__setattr__(self, "feature_count", feature_count)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.values.shape)

    def __array__(self, dtype: Any = None) -> np.ndarray:
        values = self.values.todense() if hasattr(self.values, "todense") else self.values
        return np.asarray(values, dtype=dtype)


def _json_safe(value: Any) -> Any:
    """Convert metadata containers to values accepted by ``json.dumps``."""

    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    # Metadata is diagnostic rather than an object-serialization channel.  A
    # stable textual representation keeps it JSON-safe without retaining live
    # tensors, paths, or backend objects.
    return str(value)
