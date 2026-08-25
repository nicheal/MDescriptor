# SPDX-License-Identifier: LGPL-3.0-or-later
"""High-level API: load a DPA4/DPA4C descriptor from a DeePMD checkpoint
and compute descriptors for atomic structures — pure NumPy, no torch.

Typical usage::

    from mdescriptor.descriptors.model_backed._vendor.dpa4desc.api import DescriptorEvaluator

    ev = DescriptorEvaluator.from_checkpoint("DPA4-Air-OMat24-v20260704.pt")
    desc = ev.compute(coord, atype, cell)   # (nf, nloc, dim_out) np.ndarray

``coord`` are Cartesian coordinates in Angstrom with shape ``(nf, natom, 3)``,
``atype`` integer type indices with shape ``(nf, natom)`` (see
``ev.type_map``), and ``cell`` an optional periodic cell with shape
``(nf, 3, 3)`` in Angstrom (``None`` = non-periodic).
"""

from __future__ import annotations

import inspect
import warnings
from collections.abc import Mapping
from typing import Any

import numpy as np

__all__ = [
    "DescriptorEvaluator",
    "load_descriptor",
]


_DPA4_CHECKPOINT_DERIVED = frozenset(
    {
        "wigner_calc.l1_sign_outer",
        "blocks.0.so2_conv.rotate_inv_rescale_full",
        "blocks.0.so2_conv.coeff_index_m",
        "blocks.0.so2_conv.degree_index_m",
        "blocks.1.so2_conv.rotate_inv_rescale_full",
        "blocks.1.so2_conv.coeff_index_m",
        "blocks.1.so2_conv.degree_index_m",
        "blocks.2.so2_conv.rotate_inv_rescale_full",
        "blocks.2.so2_conv.coeff_index_m",
        "blocks.2.so2_conv.degree_index_m",
        "_empty_tensor",
    }
)
_DPA4C_REQUIRED_TENSORS = frozenset(
    {
        "type_embedding.adam_type_embedding",
        "radial_basis.adam_freqs",
        "radial_embedding.layers.0.w",
        "radial_embedding.layers.1.w",
        "pair_film.network.layers.0.w",
        "pair_film.network.layers.1.w",
        "readout.gram_index",
        "readout.gram_scale",
        "readout.bispectrum_coupling",
        "readout.probe_index",
        "readout.probe_scale",
    }
)


def _resolve_parent(obj: Any, parts: list[str]) -> tuple[Any, str] | None:
    """Walk ``parts`` as attributes / list indices; return (parent, last).

    Returns None if any component — including the final one — is missing.
    """
    cur = obj
    for part in parts[:-1]:
        if isinstance(cur, (list, tuple)) and part.isdigit():
            idx = int(part)
            if idx >= len(cur):
                return None
            cur = cur[idx]
        elif hasattr(cur, part):
            cur = getattr(cur, part)
        else:
            return None
    last = parts[-1]
    if isinstance(cur, (list, tuple)) and last.isdigit():
        return (cur, last) if int(last) < len(cur) else None
    return (cur, last) if hasattr(cur, last) else None


def _assign_flat_weights(
    descriptor: Any,
    flat: dict[str, np.ndarray],
    strict: bool = False,
) -> tuple[list[str], list[str]]:
    """Assign flat ``state_dict``-style weights onto a dpmodel descriptor tree.

    Handles the reference-pt naming (``matrix``/``bias``) by falling back to
    the dpmodel NativeLayer names (``w``/``b``) when the literal attribute
    does not exist. Returns (assigned, skipped) key lists.
    """
    assigned: list[str] = []
    skipped: list[str] = []
    for key, value in flat.items():
        parts = key.split(".")
        found = _resolve_parent(descriptor, parts)
        if found is None and parts[-1] in ("matrix", "bias"):
            alt = "w" if parts[-1] == "matrix" else "b"
            found = _resolve_parent(descriptor, parts[:-1] + [alt])
        if found is None:
            skipped.append(key)
            continue
        parent, last = found
        old = parent[int(last)] if isinstance(parent, list) else getattr(parent, last)
        value = np.asarray(value)
        if hasattr(old, "shape") and tuple(old.shape) != tuple(value.shape):
            skipped.append(key)
            continue
        if strict and hasattr(old, "dtype") and np.dtype(old.dtype) != np.dtype(value.dtype):
            skipped.append(key)
            continue
        if strict and value.dtype.kind in "fc" and not np.isfinite(value).all():
            skipped.append(key)
            continue
        if isinstance(parent, list):
            parent[int(last)] = value
        else:
            setattr(parent, last, value)
        assigned.append(key)
    if strict and skipped:
        raise RuntimeError(f"unresolved descriptor weights: {skipped}")
    return assigned, skipped


class DescriptorEvaluator:
    """A standalone DPA4/DPA4C descriptor evaluator (NumPy dpmodel backend).

    Attributes
    ----------
    descriptor
        The underlying ``DescrptDPA4`` / ``DescrptDPA4C`` (dpmodel) instance.
    type_map : list[str]
        Element symbols indexed by the integer atom type.
    rcut : float
        Cutoff radius in Angstrom.
    dim_out : int
        Per-atom descriptor dimension (``descriptor.get_dim_out()``).
    """

    def __init__(
        self,
        descriptor: Any,
        type_map: list[str],
        sel: int = 128,
    ) -> None:
        self.descriptor = descriptor
        self.type_map = list(type_map)
        self.sel = int(sel)
        self.rcut = float(descriptor.rcut)
        self.dim_out = int(descriptor.get_dim_out())

    # ------------------------------------------------------------------ load
    @classmethod
    def from_checkpoint(
        cls,
        path: str,
        sel: int | None = None,
        strict: bool = False,
        checkpoint: Mapping[str, Any] | None = None,
    ) -> "DescriptorEvaluator":
        """Build a descriptor evaluator from a DeePMD-kit ``.pt`` checkpoint.

        The checkpoint is parsed without torch (:mod:`mdescriptor.descriptors.model_backed._vendor.dpa4desc.weights`).
        The descriptor configuration is read from
        ``model._extra_state.model_params`` and the weights from the
        ``model.*.atomic_model.descriptor.*`` state-dict entries.
        Fitting-net weights are ignored.

        Parameters
        ----------
        path
            Path to the ``.pt`` checkpoint file.
        sel
            Maximum neighbor count per atom used when building the dense
            neighbor list. Defaults to the value stored in the descriptor
            config (DPA4) or 128 with automatic growth (DPA4C, whose config
            is graph-native and carries no ``sel``).
        strict
            If True, raise when any descriptor weight cannot be assigned.
        """
        if checkpoint is None:
            from mdescriptor.descriptors.model_backed._vendor.dpa4desc.weights import (
                load_torch_checkpoint,
            )

            ckpt = load_torch_checkpoint(path)
        else:
            ckpt = checkpoint
        state = ckpt["model"]
        model_params: dict[str, Any] = state["_extra_state"]["model_params"]

        desc_cfg = dict(model_params["descriptor"])
        desc_type = desc_cfg.pop("type", model_params.get("type", "")).lower()
        type_map = list(model_params.get("type_map", []))
        ntypes = len(type_map) if type_map else int(desc_cfg.pop("ntypes"))

        if desc_type == "dpa4":
            from mdescriptor.descriptors.model_backed._vendor.dpa4desc.dpmodel.descriptor.dpa4 import (
                DescrptDPA4,
            )

            desc_cls = DescrptDPA4
        elif desc_type == "dpa4c":
            from mdescriptor.descriptors.model_backed._vendor.dpa4desc.dpmodel.descriptor.dpa4c import (
                DescrptDPA4C,
            )

            desc_cls = DescrptDPA4C
        else:
            raise ValueError(
                f"Unsupported descriptor type {desc_type!r}; "
                "expected 'dpa4' or 'dpa4c'."
            )

        # keep only constructor arguments (dpa4c has no **kwargs)
        sig = inspect.signature(desc_cls.__init__)
        if not any(
            p.kind is inspect.Parameter.VAR_KEYWORD
            for p in sig.parameters.values()
        ):
            desc_cfg = {k: v for k, v in desc_cfg.items() if k in sig.parameters}
        descriptor = desc_cls(ntypes=ntypes, type_map=type_map, **desc_cfg)

        # collect descriptor weights: prefix like "model.Default.atomic_model.descriptor."
        desc_sd: dict[str, np.ndarray] = {}
        prefix = None
        marker = ".descriptor."
        for key, value in state.items():
            if key == "_extra_state" or not isinstance(value, np.ndarray):
                continue
            if marker in key:
                if prefix is None:
                    prefix = key[: key.index(marker) + len(marker)]
                if key.startswith(prefix):
                    desc_sd[key[len(prefix) :]] = value
        if not desc_sd:
            raise RuntimeError(f"No descriptor weights found in {path}")
        if strict:
            # A strict load is a schema check, not merely a best-effort
            # assignment. DPA4 serializes its complete variable map; DPA4C
            # has a compact descriptor serializer, so keep its required
            # state names explicit here.
            if desc_type == "dpa4":
                serialized = descriptor.serialize()
                expected = serialized.get("@variables", {})
                expected_names = set(expected) if isinstance(expected, Mapping) else set()
                expected_names -= _DPA4_CHECKPOINT_DERIVED
            else:
                expected_names = _DPA4C_REQUIRED_TENSORS
            missing = sorted(expected_names - set(desc_sd))
            if missing:
                raise RuntimeError(f"missing descriptor weights: {missing}")
            if desc_type == "dpa4":
                unexpected = set(desc_sd) - expected_names
                unexpected -= _DPA4_CHECKPOINT_DERIVED
                if unexpected:
                    raise RuntimeError(f"unexpected descriptor weights: {sorted(unexpected)}")
                # These arrays are serialized by DeepMD as derived/export
                # metadata. The NumPy core reconstructs them from config and
                # intentionally has no writable attribute for them.
                desc_sd = {
                    key: value
                    for key, value in desc_sd.items()
                    if key in expected_names
                }
            for key, value in state.items():
                if not key.startswith(prefix) or key.endswith("._extra_state"):
                    continue
                if not isinstance(value, np.ndarray):
                    raise TypeError(f"descriptor weight {key!r} is not a NumPy tensor")
                if value.dtype.kind not in "biufc":
                    raise TypeError(f"descriptor weight {key!r} has unsupported dtype {value.dtype}")

        # version migration (DPA4): rewrite stored variables whose meaning
        # changed since the checkpoint version, then pin descriptor.version.
        # Checkpoints without a version_tensor (e.g. DPA4C) keep the fresh
        # module's own version semantics untouched.
        if "version_tensor" in desc_sd:
            version = float(desc_sd.pop("version_tensor").item())
            if hasattr(descriptor, "_migrate_variables"):
                descriptor.version = descriptor._migrate_variables(desc_sd, version)
            else:
                descriptor.version = version
        # control flags carried as tensors but owned by plain python attrs
        if "graph_lower_disabled" in desc_sd:
            descriptor._graph_lower_disabled = bool(
                desc_sd.pop("graph_lower_disabled").item()
            )
        desc_sd.pop("_empty_tensor", None)

        assigned, skipped = _assign_flat_weights(descriptor, desc_sd, strict=strict)
        if skipped:
            warnings.warn(
                f"descriptor weights skipped (no matching attribute): {skipped}"
            )

        if sel is None:
            cfg_sel = desc_cfg.get("sel")
            sel = int(cfg_sel if isinstance(cfg_sel, int) else 128)
        return cls(descriptor, type_map, sel=sel)

    # --------------------------------------------------------------- compute
    def symbols_to_atype(self, symbols: list[str]) -> np.ndarray:
        """Map element symbols to integer type indices via ``type_map``."""
        idx = {s: i for i, s in enumerate(self.type_map)}
        return np.array([idx[s] for s in symbols], dtype=np.int64)

    def _build_nlist(self, coord, atype, cell, sel):
        from mdescriptor.descriptors.model_backed._vendor.dpa4desc.dpmodel.utils.nlist import (
            extend_input_and_build_neighbor_list,
        )

        return extend_input_and_build_neighbor_list(
            coord, atype, self.rcut, [sel], mixed_types=True, box=cell
        )

    def compute(
        self,
        coord: np.ndarray,
        atype: np.ndarray,
        cell: np.ndarray | None = None,
        auto_grow_sel: bool = True,
    ) -> np.ndarray:
        """Compute per-atom descriptors.

        Parameters
        ----------
        coord
            Cartesian coordinates in Angstrom, shape ``(nf, natom, 3)``
            (a single frame may be given as ``(natom, 3)``).
        atype
            Integer atom types, shape ``(nf, natom)`` or ``(natom,)``.
        cell
            Periodic cell in Angstrom, shape ``(nf, 3, 3)`` or ``(3, 3)``;
            ``None`` for a non-periodic system.
        auto_grow_sel
            If the neighbor list fills all ``sel`` slots for any atom, the
            slot count is doubled (up to 4096) and the list rebuilt, so no
            neighbor within ``rcut`` is silently dropped.

        Returns
        -------
        np.ndarray
            Descriptor array with shape ``(nf, natom, dim_out)``.
        """
        coord = np.asarray(coord, dtype=np.float64)
        atype = np.asarray(atype, dtype=np.int64)
        if coord.ndim == 2:
            coord = coord[None]
        if atype.ndim == 1:
            atype = atype[None]
        if cell is not None:
            cell = np.asarray(cell, dtype=np.float64)
            if cell.ndim == 2:
                cell = cell[None]

        sel = self.sel
        while True:
            coord_ext, atype_ext, mapping, nlist = self._build_nlist(
                coord, atype, cell, sel
            )
            full_rows = (nlist >= 0).all(axis=-1).any()
            if not (auto_grow_sel and full_rows and sel < 4096):
                break
            sel *= 2

        out = self.descriptor.call(coord_ext, atype_ext, nlist, mapping)
        return np.asarray(out[0])

    def __call__(self, coord, atype, cell=None) -> np.ndarray:
        return self.compute(coord, atype, cell)


def load_descriptor(path: str, **kwargs: Any) -> DescriptorEvaluator:
    """Shortcut for :meth:`DescriptorEvaluator.from_checkpoint`."""
    return DescriptorEvaluator.from_checkpoint(path, **kwargs)
