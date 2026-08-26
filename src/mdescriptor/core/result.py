"""Common result and sample-index contracts.

The numeric kernels return values, while this module owns the stable meaning of
their rows.  In particular, sample identities never live in ad-hoc metadata:
they are always a contiguous two-dimensional ``int64`` array.
"""

from __future__ import annotations

import importlib
import math
from collections.abc import Iterator, Mapping
from copy import copy, deepcopy
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
from os import PathLike, fspath
from typing import Any

import numpy as np

from .errors import DescriptorConfigError

RESULT_SCHEMA_VERSION = 1


def format_values(values: Any, *, dtype: str = "float64", sparse: bool = False) -> Any:
    """Apply the single dense/CSR output representation contract."""

    target_dtype = np.dtype(dtype)
    try:
        scipy_sparse = importlib.import_module("scipy.sparse")
    except ImportError:
        scipy_sparse = None
    if sparse:
        if scipy_sparse is None:  # pragma: no cover - optional extra
            raise DescriptorConfigError(
                "sparse output requires the optional 'sparse' extra"
            )
        if scipy_sparse.issparse(values):
            return values.tocsr().astype(target_dtype, copy=False)
        return scipy_sparse.csr_matrix(np.asarray(values, dtype=target_dtype))
    if scipy_sparse is not None and scipy_sparse.issparse(values):
        return np.asarray(values.toarray(), dtype=target_dtype)
    if hasattr(values, "todense"):
        return np.asarray(values.todense(), dtype=target_dtype)
    return np.asarray(values, dtype=target_dtype)


class DescriptorLevel(str, Enum):
    ATOM = "atom"
    STRUCTURE = "structure"
    PAIR = "pair"


class _NormalizedMetadata(Mapping[str, Any]):
    """Internal immutable-by-convention metadata template.

    Kernels may reuse a template whose values have already passed the JSON-safe
    metadata normalization.  Results still receive a deep copy so mutating one
    public result cannot affect a later result from the same kernel.
    """

    __slots__ = ("_value",)

    def __init__(self, value: Mapping[str, Any]) -> None:
        self._value = dict(value)

    def __getitem__(self, key: str) -> Any:
        return self._value[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._value)

    def __len__(self) -> int:
        return len(self._value)

    def copy_for_result(self) -> dict[str, Any]:
        return deepcopy(self._value)


def structure_samples(rows: int) -> np.ndarray:
    """Build ``[structure]`` samples for structure-level values."""

    return np.arange(rows, dtype=np.int64).reshape(rows, 1)


def atom_samples(row_offsets: np.ndarray) -> np.ndarray:
    """Build ``[structure, local_atom]`` samples from atom row offsets."""

    offsets = np.ascontiguousarray(row_offsets, dtype=np.int64)
    if offsets.ndim != 1 or len(offsets) == 0:
        raise ValueError("atom row offsets must be a non-empty one-dimensional array")
    counts = np.diff(offsets)
    structure = np.repeat(np.arange(len(counts), dtype=np.int64), counts)
    local_parts = [np.arange(int(count), dtype=np.int64) for count in counts]
    local = (
        np.concatenate(local_parts).astype(np.int64, copy=False)
        if local_parts
        else np.empty(0, dtype=np.int64)
    )
    return np.column_stack((structure, local)).astype(np.int64, copy=False)


def pair_samples(
    records: Any,
    pair_row_offsets: np.ndarray,
    atom_row_offsets: np.ndarray,
) -> np.ndarray:
    """Convert kernel pair records to ``[structure, local_i, local_j, sx, sy, sz]``.

    C++ pair kernels expose global atom indices followed by the three integer
    cell shifts.  The conversion is kept here so every pair descriptor uses the
    same local-index convention.
    """

    raw = np.asarray(records)
    if raw.ndim != 2 or raw.shape[1] < 5:
        raise ValueError("pair records must have at least five columns")
    pair_offsets = np.ascontiguousarray(pair_row_offsets, dtype=np.int64)
    atom_offsets = np.ascontiguousarray(atom_row_offsets, dtype=np.int64)
    if pair_offsets.ndim != 1 or atom_offsets.ndim != 1:
        raise ValueError("pair and atom offsets must be one-dimensional")
    if len(pair_offsets) != len(atom_offsets):
        raise ValueError("pair and atom offsets describe different structures")
    if len(pair_offsets) == 0 or pair_offsets[0] != 0:
        raise ValueError("pair row offsets must start at zero")
    if int(pair_offsets[-1]) != len(raw):
        raise ValueError("pair row offsets must cover every pair record")
    structure = np.repeat(
        np.arange(len(pair_offsets) - 1, dtype=np.int64),
        np.diff(pair_offsets),
    )
    global_first = np.asarray(raw[:, 0], dtype=np.int64)
    global_second = np.asarray(raw[:, 1], dtype=np.int64)
    local_first = global_first - atom_offsets[structure]
    local_second = global_second - atom_offsets[structure]
    atom_counts = np.diff(atom_offsets)
    if len(raw) and (
        np.any(local_first < 0)
        or np.any(local_second < 0)
        or np.any(local_first >= atom_counts[structure])
        or np.any(local_second >= atom_counts[structure])
    ):
        raise ValueError("pair records contain an invalid local atom index")
    shifts = np.asarray(raw[:, 2:5], dtype=np.int64)
    return np.ascontiguousarray(
        np.column_stack((structure, local_first, local_second, shifts)),
        dtype=np.int64,
    )


@dataclass(frozen=True, slots=True)
class DescriptorResult:
    """Values plus the row and feature metadata needed to interpret them."""

    values: Any
    level: DescriptorLevel | str
    structure_ids: tuple[str, ...]
    row_offsets: np.ndarray | None
    labels: tuple[str, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    samples: Any = None
    feature_count: int | None = None
    # Pair rows need the atom offsets separately from their own row offsets in
    # order to validate local atom identities. This is construction context,
    # not part of the serialized/public result payload.
    _atom_row_offsets: np.ndarray | None = field(
        default=None,
        repr=False,
        compare=False,
        kw_only=True,
    )

    def __post_init__(self) -> None:
        level = DescriptorLevel(self.level)
        object.__setattr__(self, "level", level)
        object.__setattr__(self, "structure_ids", tuple(str(item) for item in self.structure_ids))
        labels = tuple(str(item) for item in self.labels)
        object.__setattr__(self, "labels", labels)

        shape = tuple(getattr(self.values, "shape", ()))
        if len(shape) != 2:
            raise ValueError("descriptor values must be a two-dimensional array")
        rows, width = int(shape[0]), int(shape[1])
        if len(labels) != width:
            raise ValueError("labels must contain exactly one entry per feature column")

        feature_count = self.feature_count
        if feature_count is None:
            feature_count = width
        elif isinstance(feature_count, bool) or feature_count < 0 or int(feature_count) != width:
            raise ValueError("feature_count does not match the result feature dimension")
        feature_count = int(feature_count)
        object.__setattr__(self, "feature_count", feature_count)

        offsets = self.row_offsets
        if offsets is not None:
            raw_offsets = np.asarray(offsets)
            if raw_offsets.ndim != 1 or not np.issubdtype(raw_offsets.dtype, np.integer):
                raise ValueError("row offsets must be a one-dimensional integer array")
            offsets = np.ascontiguousarray(raw_offsets, dtype=np.int64)
            object.__setattr__(self, "row_offsets", offsets)

        atom_offsets = self._atom_row_offsets
        if atom_offsets is not None:
            raw_atom_offsets = np.asarray(atom_offsets)
            if (
                raw_atom_offsets.ndim != 1
                or not np.issubdtype(raw_atom_offsets.dtype, np.integer)
            ):
                raise ValueError("atom row offsets must be a one-dimensional integer array")
            atom_offsets = np.ascontiguousarray(raw_atom_offsets, dtype=np.int64)
            if len(atom_offsets) != len(self.structure_ids) + 1:
                raise ValueError("atom row offsets must contain one entry per structure boundary")
            if len(atom_offsets) == 0 or atom_offsets[0] != 0:
                raise ValueError("atom row offsets must start at zero")
            if np.any(np.diff(atom_offsets) < 0):
                raise ValueError("atom row offsets must be non-decreasing")
            object.__setattr__(self, "_atom_row_offsets", atom_offsets)
        self._validate_layout(level, rows, offsets)

        samples = self.samples
        if samples is None:
            if level is DescriptorLevel.STRUCTURE:
                samples = structure_samples(rows)
            elif level is DescriptorLevel.ATOM:
                if offsets is None:
                    raise ValueError("atom-level results require row offsets")
                samples = atom_samples(offsets)
            else:
                raise ValueError("pair-level results require explicit pair samples")
        samples = np.asarray(samples)
        if samples.ndim != 2 or int(samples.shape[0]) != rows:
            raise ValueError("samples must be a two-dimensional array with one row per sample")
        expected_columns = {
            DescriptorLevel.STRUCTURE: 1,
            DescriptorLevel.ATOM: 2,
            DescriptorLevel.PAIR: 6,
        }[level]
        if int(samples.shape[1]) != expected_columns:
            raise ValueError(
                f"{level.value}-level samples must have {expected_columns} columns"
            )
        if not np.issubdtype(samples.dtype, np.integer):
            raise ValueError("samples must contain integer indices")
        samples = np.ascontiguousarray(samples, dtype=np.int64)
        index_columns = 3 if level is DescriptorLevel.PAIR else samples.shape[1]
        if np.any(samples[:, :index_columns] < 0):
            raise ValueError("sample indices cannot be negative")
        if level is DescriptorLevel.STRUCTURE:
            expected = np.arange(rows, dtype=np.int64)
            if not np.array_equal(samples[:, 0], expected):
                raise ValueError("structure samples must identify rows in structure order")
        elif level is DescriptorLevel.ATOM:
            assert offsets is not None  # validated above
            counts = np.diff(offsets)
            if np.any(samples[:, 0] >= len(counts)):
                raise ValueError("atom samples contain an invalid structure index")
            if rows and np.any(samples[:, 1] >= counts[samples[:, 0]]):
                raise ValueError("atom samples contain an invalid local atom index")
        else:
            assert offsets is not None  # validated above
            if np.any(samples[:, 0] >= len(offsets) - 1):
                raise ValueError("pair samples contain an invalid structure index")
            if self._atom_row_offsets is None:
                raise ValueError("pair-level results require atom row offsets for sample validation")
            atom_counts = np.diff(self._atom_row_offsets)
            if rows and (
                np.any(samples[:, 1] >= atom_counts[samples[:, 0]])
                or np.any(samples[:, 2] >= atom_counts[samples[:, 0]])
            ):
                raise ValueError("pair samples contain an invalid local atom index")
        object.__setattr__(self, "samples", samples)

        if isinstance(self.metadata, _NormalizedMetadata):
            metadata = self.metadata.copy_for_result()
        else:
            metadata = _metadata_v1(self.metadata, level, feature_count)
        object.__setattr__(self, "metadata", metadata)

    def _validate_layout(
        self,
        level: DescriptorLevel,
        rows: int,
        offsets: np.ndarray | None,
    ) -> None:
        if level is DescriptorLevel.STRUCTURE:
            if offsets is not None:
                raise ValueError("structure-level results must not define row offsets")
            if len(self.structure_ids) != rows:
                raise ValueError(
                    "structure_ids must contain exactly one identifier per structure row"
                )
            return
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
            raise ValueError("the final row offset must equal the number of result rows")

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.values.shape)

    def _replace_output(
        self,
        values: Any,
        output: Mapping[str, Any],
    ) -> DescriptorResult:
        """Replace output representation without rerunning result validation.

        The shared adapter calls this only after this result has completed the
        full constructor validation.  Formatting preserves the two-dimensional
        shape, so rebuilding the dataclass would only repeat validation and
        metadata normalization on the hot path.
        """

        if tuple(getattr(values, "shape", ())) != self.shape:
            raise ValueError("formatted descriptor values changed the result shape")
        updated = copy(self)
        metadata = dict(self.metadata)
        metadata["output"] = {
            "dtype": output["dtype"],
            "sparse": output["sparse"],
        }
        object.__setattr__(updated, "values", values)
        object.__setattr__(updated, "metadata", metadata)
        return updated

    def __array__(self, dtype: Any = None) -> np.ndarray:
        values = self.values.todense() if hasattr(self.values, "todense") else self.values
        return np.asarray(values, dtype=dtype)


def _metadata_v1(
    metadata: Mapping[str, Any],
    level: DescriptorLevel,
    feature_count: int | None,
) -> dict[str, Any]:
    """Normalize legacy kernel details into the fixed metadata envelope."""

    if not isinstance(metadata, Mapping):
        raise TypeError("descriptor metadata must be a mapping")
    raw = _json_safe(dict(metadata))
    if not isinstance(raw, dict):  # pragma: no cover - defensive
        raise TypeError("descriptor metadata must be a JSON object")
    descriptor = raw.pop("descriptor", "unknown")
    backend = raw.pop("backend", "unknown")
    raw.pop("schema_version", None)
    raw.pop("level", None)
    raw.pop("feature_count", None)
    # Pair identity belongs in ``samples`` only.
    raw.pop("pair_records", None)
    existing_details = raw.pop("details", None)

    output_value = raw.pop("output", None)
    if output_value is None:
        output_value = {
            "dtype": raw.pop("dtype", "float64"),
            "sparse": raw.pop("sparse", False),
        }
    output = _metadata_options(output_value, "output", {"dtype", "sparse"})
    execution_value = raw.pop("execution", None)
    if execution_value is None:
        execution_value = {
            "device": raw.pop("device", "cpu"),
            "num_threads": raw.pop("num_threads", None),
        }
    execution = _metadata_options(
        execution_value,
        "execution",
        {"device", "num_threads"},
    )
    if not isinstance(descriptor, str) or not descriptor.strip():
        raise TypeError("metadata descriptor must be a non-empty string")
    if not isinstance(backend, str) or not backend.strip():
        raise TypeError("metadata backend must be a non-empty string")

    normalized: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "descriptor": descriptor,
        "backend": backend,
        "level": level.value,
        "feature_count": feature_count,
        "output": output,
        "execution": execution,
    }
    model = raw.pop("model", None)
    if model is not None:
        normalized["model"] = model
    if existing_details is not None and not isinstance(existing_details, Mapping):
        raise TypeError("metadata details must be a JSON object")
    details = dict(existing_details or {})
    details.update(raw)
    if details:
        normalized["details"] = details
    return _json_safe(normalized)


def normalize_metadata(
    metadata: Mapping[str, Any],
    level: DescriptorLevel | str,
    feature_count: int | None,
) -> Mapping[str, Any]:
    """Create a reusable, already validated metadata template for kernels."""

    return _NormalizedMetadata(
        _metadata_v1(metadata, DescriptorLevel(level), feature_count)
    )


def _metadata_options(
    value: Any,
    name: str,
    fields_allowed: set[str],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"metadata {name} must be a JSON object")
    unknown = set(value) - fields_allowed
    if unknown:
        raise TypeError(
            f"metadata {name} has unsupported field(s): {', '.join(sorted(map(str, unknown)))}"
        )
    result = {key: value[key] for key in fields_allowed if key in value}
    if name == "output":
        dtype = result.get("dtype", "float64")
        sparse = result.get("sparse", False)
        if dtype not in {"float32", "float64"}:
            raise TypeError("metadata output dtype must be 'float32' or 'float64'")
        if not isinstance(sparse, bool):
            raise TypeError("metadata output sparse must be a boolean")
        return {"dtype": dtype, "sparse": sparse}
    device = result.get("device", "cpu")
    num_threads = result.get("num_threads")
    if not isinstance(device, str) or not device.strip():
        raise TypeError("metadata execution device must be a non-empty string")
    if num_threads == 0:
        num_threads = None
    if isinstance(num_threads, bool) or (
        num_threads is not None
        and (not isinstance(num_threads, int) or num_threads <= 0)
    ):
        raise TypeError("metadata execution num_threads must be positive or None")
    return {"device": device, "num_threads": num_threads}


def _json_safe(value: Any) -> Any:
    """Convert a value to strict JSON-safe primitives.

    Unknown live objects are rejected.  Metadata is a persistence contract, so
    silently replacing them with ``str(value)`` would make snapshots unstable.
    """

    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, Enum):
        return _json_safe(value.value)
    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: _json_safe(getattr(value, item.name))
            for item in fields(value)
        }
    if isinstance(value, PathLike):
        raw = fspath(value)
        return raw.decode() if isinstance(raw, bytes) else str(raw)
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if isinstance(key, str):
                json_key = key
            elif isinstance(key, (bool, int, float)) and not isinstance(key, complex):
                json_key = str(key)
            else:
                raise TypeError("JSON object keys must be strings or JSON scalar keys")
            result[json_key] = _json_safe(item)
        return result
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise TypeError("JSON metadata cannot contain non-finite numbers")
        return value
    raise TypeError(f"metadata value of type {type(value).__name__} is not JSON-safe")


__all__ = [
    "DescriptorLevel",
    "DescriptorResult",
    "RESULT_SCHEMA_VERSION",
    "atom_samples",
    "format_values",
    "pair_samples",
    "structure_samples",
]
