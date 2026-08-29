"""Torch-free DPA4/DPA4C checkpoint and runtime seam.

The numerical implementation is vendored from ``dpa4-descriptor``.  This
module is the small MDescriptor-owned boundary around it: it validates an
official checkpoint once, creates an independent NumPy runtime per descriptor
instance, and converts the flattened :class:`StructureBatch` contract into
the frame-local graph contract used by the vendored code.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from ...core.errors import CancelledError, ModelLoadError
from ...core.input import StructureBatch
from ._vendor.dpa4desc.api import (
    _DPA4C_REQUIRED_TENSORS,
    DescriptorEvaluator,
)
from ._vendor.dpa4desc.dpmodel.utils.neighbor_graph import (
    graph_from_dense_quartet,
)
from ._vendor.dpa4desc.weights import load_torch_checkpoint
from .graph import _ATOMIC_SYMBOLS


@dataclass(frozen=True, slots=True)
class DpaCheckpointInfo:
    """Validated immutable identity for one official DPA checkpoint."""

    descriptor: Literal["DPA4", "DPA4C"]
    type_map: tuple[str, ...]
    feature_count: int
    cutoff: float
    precision: str
    supports_spin: bool
    supports_charge_spin: bool


def _checkpoint_parameters(checkpoint: Mapping[str, Any]) -> Mapping[str, Any]:
    model = checkpoint.get("model")
    if not isinstance(model, Mapping):
        raise ValueError("official DPA checkpoint is missing its model mapping")
    extra = model.get("_extra_state")
    if not isinstance(extra, Mapping):
        raise ValueError("official DPA checkpoint is missing model._extra_state")
    parameters = extra.get("model_params")
    if not isinstance(parameters, Mapping):
        raise ValueError("official DPA checkpoint is missing model parameters")
    return parameters


def _validate_checkpoint_mapping(
    checkpoint: Mapping[str, Any],
    *,
    path: str,
    expected_descriptor: Literal["DPA4", "DPA4C"],
) -> tuple[DpaCheckpointInfo, Mapping[str, Any]]:
    if not isinstance(checkpoint, Mapping):
        raise ValueError("checkpoint root must be a mapping")
    parameters = _checkpoint_parameters(checkpoint)
    descriptor = parameters.get("descriptor")
    actual_type = descriptor.get("type") if isinstance(descriptor, Mapping) else None
    expected_type = expected_descriptor.lower()
    if actual_type != expected_type:
        raise ValueError(
            f"checkpoint descriptor type is {actual_type!r}, expected {expected_type!r}"
        )
    raw_type_map = parameters.get("type_map")
    if not isinstance(raw_type_map, (list, tuple)) or not raw_type_map:
        raise ValueError("checkpoint type_map must be a non-empty list")
    type_map = tuple(str(value).strip() for value in raw_type_map)
    if any(not value for value in type_map) or len(set(type_map)) != len(type_map):
        raise ValueError("checkpoint type_map must contain unique non-empty symbols")

    model = checkpoint.get("model")
    if not isinstance(model, Mapping):
        raise ValueError("checkpoint is missing its model mapping")
    descriptor_entries = {
        key: value
        for key, value in model.items()
        if ".descriptor." in key
    }
    if not descriptor_entries:
        raise ValueError("checkpoint has no descriptor tensors")
    for key, value in descriptor_entries.items():
        if not isinstance(value, np.ndarray):
            raise TypeError(f"descriptor tensor {key!r} is not a NumPy array")
        if value.dtype.kind not in "biufc":
            raise TypeError(f"descriptor tensor {key!r} has unsupported dtype {value.dtype}")
        if value.dtype.kind in "fc" and not np.isfinite(value).all():
            raise ValueError(f"descriptor tensor {key!r} contains non-finite values")

    if expected_descriptor == "DPA4C":
        marker = ".descriptor."
        prefix = next(
            (key[: key.index(marker) + len(marker)] for key in descriptor_entries),
            None,
        )
        if prefix is None:
            raise ValueError("checkpoint has no DPA4C descriptor prefix")
        names = {key[len(prefix) :] for key in descriptor_entries}
        missing = sorted(_DPA4C_REQUIRED_TENSORS - names)
        if missing:
            raise ValueError(f"checkpoint is missing DPA4C tensor(s): {missing}")

    evaluator = DescriptorEvaluator.from_checkpoint(
        path,
        strict=True,
        checkpoint=checkpoint,
    )
    model_descriptor = evaluator.descriptor
    actual_type_map = tuple(str(value).strip() for value in evaluator.type_map)
    if actual_type_map != type_map:
        raise ValueError("checkpoint type_map was not preserved by the descriptor loader")
    info = DpaCheckpointInfo(
        descriptor=expected_descriptor,
        type_map=type_map,
        feature_count=int(evaluator.dim_out),
        cutoff=float(evaluator.rcut),
        precision=str(getattr(model_descriptor, "precision", "float64")),
        supports_spin=bool(model_descriptor.supports_native_spin()),
        supports_charge_spin=bool(model_descriptor.supports_charge_spin()),
    )
    return info, checkpoint


def load_dpa_checkpoint(
    path: Path,
    *,
    expected_descriptor: Literal["DPA4", "DPA4C"],
) -> tuple[DpaCheckpointInfo, Mapping[str, Any]]:
    """Read and strictly validate an official ``.pt`` without importing Torch."""

    try:
        checkpoint = load_torch_checkpoint(str(path))
        return _validate_checkpoint_mapping(
            checkpoint,
            path=str(path),
            expected_descriptor=expected_descriptor,
        )
    except ModelLoadError:
        raise
    except Exception as exc:
        raise ModelLoadError(f"invalid {expected_descriptor} checkpoint: {path}") from exc


def validate_dpa_checkpoint_mapping(
    checkpoint: Mapping[str, Any],
    *,
    expected_descriptor: Literal["DPA4", "DPA4C"],
) -> DpaCheckpointInfo:
    """Validate an already parsed checkpoint without touching the filesystem."""

    try:
        info, _ = _validate_checkpoint_mapping(
            checkpoint,
            path="<in-memory checkpoint>",
            expected_descriptor=expected_descriptor,
        )
        return info
    except Exception as exc:
        raise ModelLoadError(
            f"invalid {expected_descriptor} checkpoint mapping"
        ) from exc


def new_runtime(
    path: Path,
    checkpoint: Mapping[str, Any],
) -> DescriptorEvaluator:
    """Create one independent NumPy evaluator from a shared CPU checkpoint."""

    try:
        return DescriptorEvaluator.from_checkpoint(
            str(path),
            strict=True,
            checkpoint=checkpoint,
        )
    except Exception as exc:
        raise ModelLoadError(f"failed to construct DPA runtime from {path}") from exc


def _frame_inputs(
    evaluator: DescriptorEvaluator,
    coordinates: np.ndarray,
    atype: np.ndarray,
    cell: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build the archive's dense graph inputs, growing neighbor capacity safely."""

    selection = int(evaluator.sel)
    frame_coordinates = np.asarray(coordinates, dtype=np.float64)[None, ...]
    frame_atype = np.asarray(atype, dtype=np.int64)[None, ...]
    frame_cell = None if cell is None else np.asarray(cell, dtype=np.float64)[None, ...]
    while True:
        coord_ext, atype_ext, mapping, nlist = evaluator._build_nlist(
            frame_coordinates,
            frame_atype,
            frame_cell,
            selection,
        )
        full_rows = bool(np.asarray((nlist >= 0).all(axis=-1).any()))
        if not full_rows or selection >= 4096:
            return coord_ext, atype_ext, mapping, nlist
        selection *= 2


def _frame_descriptor(
    evaluator: DescriptorEvaluator,
    coordinates: np.ndarray,
    atype: np.ndarray,
    cell: np.ndarray | None,
    *,
    spin: np.ndarray | None,
    charge_spin: np.ndarray | None,
) -> np.ndarray:
    """Evaluate one frame through the archive's DPA4/DPA4C graph ABI."""

    coord_ext, atype_ext, mapping, nlist = _frame_inputs(
        evaluator,
        coordinates,
        atype,
        cell,
    )
    descriptor = evaluator.descriptor
    if evaluator.type_map and descriptor.__class__.__name__ == "DescrptDPA4":
        output = descriptor.call(
            coord_ext,
            atype_ext,
            nlist,
            mapping,
            spin=spin,
            charge_spin=charge_spin,
        )[0]
        return np.asarray(output, dtype=np.float64).reshape(len(coordinates), -1)

    # DPA4C exposes a dense ``call`` adapter as well.  Keep this path on the
    # adapter instead of calling ``evaluate_graph`` directly: ``call`` is
    # decorated with ``cast_precision`` and therefore performs the same
    # float32 (or checkpoint-selected) geometry conversion as the native
    # descriptor.  Calling ``evaluate_graph`` here with the float64 arrays
    # produced by ``_frame_inputs`` silently promoted the whole reference
    # calculation and made it disagree with the C++/DeepMD execution path.
    # A spin-conditioned DPA4C has no spin argument on the dense ABI, so it
    # keeps the graph route below where the per-atom spin tensor is threaded.
    if (
        descriptor.__class__.__name__ == "DescrptDPA4C"
        and getattr(descriptor, "spin", None) is None
    ):
        output = descriptor.call(
            coord_ext,
            atype_ext,
            nlist,
            mapping,
            charge_spin=charge_spin,
        )[0]
        return np.asarray(output, dtype=np.float64).reshape(len(coordinates), -1)

    # ``evaluate_graph`` expects geometry in descriptor precision.  This
    # branch is currently needed only for spin-conditioned DPA4C, whose dense
    # ABI cannot carry the per-atom spin tensor.  Match the graph-native
    # precision contract before constructing the graph rather than letting
    # float64 coordinates leak into all downstream reductions.
    descriptor_precision = getattr(descriptor, "precision", None)
    graph_coordinates = coord_ext
    if descriptor_precision is not None:
        try:
            graph_coordinates = np.asarray(coord_ext, dtype=np.dtype(descriptor_precision))
        except TypeError:
            # ``default`` is a valid dpmodel precision alias but not a NumPy
            # dtype name; the descriptor's own graph lower will resolve it.
            pass

    graph, atype_local = graph_from_dense_quartet(
        graph_coordinates,
        atype_ext,
        nlist,
        mapping,
    )
    frame_count = int(graph.n_node.shape[0])
    if descriptor.spin is not None:
        spin = descriptor.require_spin(spin)
    charge_spin = descriptor.require_charge_spin(
        charge_spin,
        frame_count,
        graph.edge_vec,
    )
    output, _envelope = descriptor.evaluate_graph(
        graph,
        atype_local,
        descriptor.type_embedding.call(),
        spin,
        charge_spin,
    )
    return np.asarray(output, dtype=np.float64).reshape(len(coordinates), -1)


def compute_batch(
    evaluator: DescriptorEvaluator,
    batch: StructureBatch,
    *,
    control: Any = None,
) -> np.ndarray:
    """Compute all frames and restore the project's flattened atom order."""

    if control is not None:
        control.reset(batch.structures)
    rows: list[np.ndarray] = []
    for frame in range(batch.structures):
        if control is not None and bool(control.cancelled()):
            raise CancelledError("descriptor computation was cancelled")
        begin = int(batch.offsets[frame])
        end = int(batch.offsets[frame + 1])
        symbols: list[str] = []
        for number in batch.numbers[begin:end].tolist():
            try:
                symbols.append(_ATOMIC_SYMBOLS[int(number)])
            except KeyError as exc:
                raise ValueError(
                    f"atomic number {number} is absent from the checkpoint type_map"
                ) from exc
        try:
            atype = evaluator.symbols_to_atype(symbols)
        except KeyError as exc:
            raise ValueError(
                f"element {exc.args[0]!r} is absent from the checkpoint type_map"
            ) from exc
        spin = None if batch.spins is None else batch.spins[begin:end]
        charge_spin = None if batch.charge_spin is None else batch.charge_spin[frame : frame + 1]
        rows.append(
            _frame_descriptor(
                evaluator,
                batch.positions[begin:end],
                atype,
                batch.cells[frame] if np.all(batch.pbc[frame] == 1) else None,
                spin=spin,
                charge_spin=charge_spin,
            )
        )
        if control is not None:
            control.mark_completed()
    if not rows:
        return np.empty((0, int(evaluator.dim_out)), dtype=np.float64)
    return np.concatenate(rows, axis=0)


__all__ = [
    "DpaCheckpointInfo",
    "compute_batch",
    "load_dpa_checkpoint",
    "new_runtime",
    "validate_dpa_checkpoint_mapping",
]
