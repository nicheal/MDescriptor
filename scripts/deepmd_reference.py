"""DeepMD-kit 3.2.0 reference evaluation for DPA4 and DPA4C.

The project runtime is intentionally Torch-free.  This module is only used by
the optional external-reference suite and by the golden generator when the
``reference-deepmd`` extra is installed.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, Literal

import numpy as np

from mdescriptor import StructureBatch

DescriptorName = Literal["DPA4", "DPA4C"]


def _torch_graph(graph: Any, torch: Any) -> Any:
    """Convert DeepMD-kit's NumPy graph to the torch graph ABI."""

    values: dict[str, Any] = {}
    for field in dataclasses.fields(graph):
        value = getattr(graph, field.name)
        if value is None or field.name == "destination_sorted":
            values[field.name] = value
        elif field.name == "edge_vec":
            values[field.name] = torch.as_tensor(value, dtype=torch.float64)
        elif field.name == "edge_mask":
            values[field.name] = torch.as_tensor(value, dtype=torch.bool)
        else:
            values[field.name] = torch.as_tensor(value, dtype=torch.int64)
    return dataclasses.replace(graph, **values)


def _external_stack() -> tuple[Any, Any, Any]:
    """Load and validate the pinned DeepMD/Torch reference stack."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised by optional jobs
        raise RuntimeError(
            "DPA4 external goldens require the reference-deepmd extra "
            "(deepmd-kit[torch]==3.2.0)"
        ) from exc
    try:
        import deepmd
    except ImportError as exc:  # pragma: no cover - exercised by optional jobs
        raise RuntimeError(
            "DPA4 external goldens require deepmd-kit==3.2.0"
        ) from exc
    version = getattr(deepmd, "__version__", None)
    if version != "3.2.0":
        raise RuntimeError(
            f"DPA4 external goldens require deepmd-kit==3.2.0, found {version!r}"
        )
    from deepmd.pt_expt.utils.env import DEVICE

    return torch, deepmd, DEVICE


def _frame_reference(
    name: DescriptorName,
    deep_pot: Any,
    torch: Any,
    device: Any,
    numbers: np.ndarray,
    positions: np.ndarray,
    cell: np.ndarray | None,
) -> np.ndarray:
    from ase.data import atomic_numbers

    type_indices = {
        int(atomic_numbers[symbol]): index
        for index, symbol in enumerate(deep_pot.get_type_map())
    }
    try:
        atom_types = np.asarray(
            [[type_indices[int(number)] for number in numbers]],
            dtype=np.int32,
        )
    except KeyError as exc:
        raise ValueError(
            f"atomic number {exc.args[0]} is absent from the DeepMD type_map"
        ) from exc
    coordinates = np.asarray(positions, dtype=np.float64)[None, :, :]
    box = None if cell is None else np.asarray(cell, dtype=np.float64).reshape(1, 9)
    deep_eval = deep_pot.deep_eval

    if name == "DPA4":
        value = deep_eval.eval_descriptor(coordinates, box, atom_types)
        return np.asarray(value, dtype=np.float64).reshape(len(numbers), -1)

    deep_eval._dpmodel.eval()
    graph_descriptor = deep_eval._dpmodel.get_dp_atomic_model().descriptor
    type_embedding = graph_descriptor.type_embedding.call()
    graph = deep_eval._build_eval_graph(coordinates, atom_types, box, device)
    graph = _torch_graph(graph, torch)
    with torch.no_grad():
        output, _ = graph_descriptor.call_graph(
            graph,
            torch.as_tensor(atom_types.reshape(-1), dtype=torch.int64),
            type_embedding=type_embedding,
        )
    return output.detach().cpu().numpy().astype(np.float64).reshape(len(numbers), -1)


def evaluate_batch(
    name: DescriptorName,
    model_path: str | Path,
    batch: StructureBatch,
) -> np.ndarray:
    """Evaluate every frame through DeepMD-kit 3.2.0 and restore row order.

    DeepMD accepts ``cells=None`` for a non-periodic frame.  This is the
    external counterpart to MDescriptor's all-zero cell and all-zero-PBC
    representation; a singular zero cell is never handed to DeepMD as a
    periodic box.
    """

    if name not in ("DPA4", "DPA4C"):
        raise ValueError(f"unsupported DeepMD descriptor {name!r}")
    torch, _deepmd, device = _external_stack()
    kwargs = {"neighbor_graph_method": "ase"} if name == "DPA4C" else {}
    from deepmd.infer import DeepPot

    deep_pot = DeepPot(str(model_path), **kwargs)
    rows: list[np.ndarray] = []
    try:
        for frame in range(batch.structures):
            begin = int(batch.offsets[frame])
            end = int(batch.offsets[frame + 1])
            cell = batch.cells[frame] if bool(np.all(batch.pbc[frame] == 1)) else None
            rows.append(
                _frame_reference(
                    name,
                    deep_pot,
                    torch,
                    device,
                    batch.numbers[begin:end],
                    batch.positions[begin:end],
                    cell,
                )
            )
    finally:
        close = getattr(deep_pot, "close", None)
        if close is not None:
            close()
    if not rows:
        return np.empty((0, 0), dtype=np.float64)
    return np.concatenate(rows, axis=0)


__all__ = ["evaluate_batch"]
