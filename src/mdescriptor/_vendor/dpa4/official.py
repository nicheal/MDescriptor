"""Project-local loader and graph adapter for official DPA4 ``.pt`` files.

This module intentionally contains no import from the upstream training
package.  The network, graph contract, and checkpoint deserializer live under
``mdescriptor._vendor.dpa4._official_core`` and are used directly here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from .model import Dpa4Config
from ._official_core.dpa4 import DescrptDPA4
from ._official_core.dpmodel.utils.neighbor_graph import NeighborGraph


def _torch_load(path: str | Path) -> Any:
    # The model extra requires a Torch version with the safe weights-only
    # loader.  Never fall back to executing arbitrary pickle payloads.
    return torch.load(str(path), map_location="cpu", weights_only=True)


def _as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _official_descriptor_config(
    checkpoint: Mapping[str, Any],
) -> tuple[dict[str, Any], Mapping[str, Any]]:
    model = checkpoint.get("model")
    if not isinstance(model, Mapping):
        raise ValueError("official DPA4 .pt file is missing its model mapping")
    extra_state = model.get("_extra_state")
    if not isinstance(extra_state, Mapping):
        raise ValueError("official DPA4 .pt file is missing model._extra_state")
    model_params = extra_state.get("model_params")
    if not isinstance(model_params, Mapping):
        raise ValueError("official DPA4 .pt file is missing model parameters")
    descriptor = model_params.get("descriptor")
    if not isinstance(descriptor, Mapping) or descriptor.get("type") != "dpa4":
        actual = descriptor.get("type") if isinstance(descriptor, Mapping) else None
        raise ValueError(f"official .pt descriptor type is not DPA4: {actual!r}")
    type_map = model_params.get("type_map")
    if not isinstance(type_map, (list, tuple)) or not type_map:
        raise ValueError("official DPA4 .pt file has no non-empty model type_map")

    config = dict(descriptor)
    config.pop("type", None)
    config["ntypes"] = len(type_map)
    config["type_map"] = [str(value) for value in type_map]
    # The public project config keeps this compact summary field. The official
    # core itself receives the complete radial_so2_rank setting below.
    config["radial_modes"] = int(config.get("radial_so2_rank", 1))
    return config, model


class OfficialDpa4Model:
    """Loaded official DPA4 descriptor with a project graph-facing call API."""

    def __init__(self, descriptor: DescrptDPA4, config: Dpa4Config) -> None:
        self.descriptor = descriptor
        self.config = config
        self.compute_dtype = {
            "float16": torch.float16,
            "float32": torch.float32,
            "float64": torch.float64,
            "bfloat16": torch.bfloat16,
        }.get(descriptor.compute_precision, torch.float32)
        self.feature_count = int(descriptor.get_dim_out())

    def __call__(
        self,
        edge_vectors: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        atype: torch.Tensor,
        n_node: int | torch.Tensor,
        *,
        spin: torch.Tensor | None = None,
        charge_spin: torch.Tensor | None = None,
    ) -> torch.Tensor:
        edge_vectors = edge_vectors.to(dtype=self.compute_dtype)
        src = src.to(dtype=torch.long, device=edge_vectors.device)
        dst = dst.to(dtype=torch.long, device=edge_vectors.device)
        atype = atype.to(dtype=torch.long, device=edge_vectors.device)
        if isinstance(n_node, int):
            n_node = torch.tensor([n_node], dtype=torch.long, device=edge_vectors.device)
        else:
            n_node = n_node.to(dtype=torch.long, device=edge_vectors.device).reshape(-1)
        if int(atype.numel()) == 0:
            raise ValueError("DPA4 requires at least one atom")
        if edge_vectors.ndim != 2 or edge_vectors.shape[-1] != 3:
            raise ValueError("DPA4 edge_vectors must have shape (n_edges, 3)")
        if src.shape != dst.shape or src.shape != (edge_vectors.shape[0],):
            raise ValueError("DPA4 graph arrays have inconsistent edge counts")

        # Keep two masked guard edges in the graph. This is the official graph
        # contract for stable empty-neighbour and export paths, while preserving
        # the original edge order for all valid edges.
        guard_index = torch.zeros((2, 2), dtype=torch.long, device=edge_vectors.device)
        guard_vectors = torch.zeros((2, 3), dtype=edge_vectors.dtype, device=edge_vectors.device)
        graph = NeighborGraph(
            n_node=n_node,
            edge_index=torch.cat((torch.stack((src, dst), dim=0), guard_index), dim=1),
            edge_vec=torch.cat((edge_vectors, guard_vectors), dim=0),
            edge_mask=torch.cat(
                (
                    torch.ones(edge_vectors.shape[0], dtype=torch.bool, device=edge_vectors.device),
                    torch.zeros(2, dtype=torch.bool, device=edge_vectors.device),
                ),
                dim=0,
            ),
        )
        values, _ = self.descriptor.call_graph(
            graph,
            atype,
            spin=spin,
            charge_spin=charge_spin,
        )
        return values


def load_official_dpa4_checkpoint(
    path: str | Path,
    *,
    device: str = "cpu",
) -> tuple[OfficialDpa4Model, Dpa4Config]:
    """Load an official DPA4 ``.pt`` checkpoint without the upstream package."""

    checkpoint = _torch_load(path)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("official DPA4 checkpoint must be a mapping")
    config_mapping, model_state = _official_descriptor_config(checkpoint)
    config = Dpa4Config.from_mapping(config_mapping)

    descriptor_kwargs = dict(config_mapping)
    descriptor_kwargs.pop("radial_modes", None)
    descriptor = DescrptDPA4(**descriptor_kwargs)
    serialized = descriptor.serialize()
    variables = serialized.get("@variables")
    if not isinstance(variables, dict):
        raise ValueError("project-local DPA4 core did not produce a variable archive")

    prefixes = (
        "model.Default.atomic_model.descriptor.",
        "Default.atomic_model.descriptor.",
        "descriptor.",
    )
    prefix = next(
        (candidate for candidate in prefixes if all(candidate + key in model_state for key in variables)),
        None,
    )
    if prefix is None:
        raise ValueError("official DPA4 .pt variables do not match the project-local core")
    for name in variables:
        variables[name] = _as_numpy(model_state[prefix + name])

    descriptor = DescrptDPA4.deserialize(serialized)
    native = OfficialDpa4Model(descriptor, config)
    if device != "cpu":
        if not str(device).startswith("cuda") or not torch.cuda.is_available():
            raise RuntimeError(f"requested DPA4 device {device!r}, but CUDA is unavailable")
        # The local core accepts array-API tensors and keeps model variables as
        # host arrays. Inputs are placed on the requested device by the caller.
    return native, config


__all__ = ["OfficialDpa4Model", "load_official_dpa4_checkpoint"]
