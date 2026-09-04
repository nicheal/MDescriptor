"""DPA4 descriptor adapter with a native C++ core and NumPy fallback."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from ...models import DPA4_MODEL
from .dpa_common import DpaKernelBase


def _as_float32(value: Any) -> np.ndarray:
    """Materialize one checkpoint tensor for the native DPA4 ABI."""

    return np.ascontiguousarray(np.asarray(value, dtype=np.float32))


def _as_int64(value: Any) -> np.ndarray:
    """Materialize one integer checkpoint/index buffer for the native ABI."""

    return np.ascontiguousarray(np.asarray(value, dtype=np.int64))


def _native_payload(
    descriptor: Any,
    *,
    num_threads: int,
) -> dict[str, Any] | None:
    """Extract the default DPA4 graph into the flat C++ inference ABI."""

    if (
        getattr(descriptor, "version", 0) < 1.2
        or getattr(descriptor, "use_spin", None) is not None
        or getattr(descriptor, "charge_spin_embedding", None) is not None
        or getattr(descriptor, "exclude_types", None)
        or getattr(descriptor, "force_embedding", None) is not None
        or len(getattr(descriptor, "blocks", ())) != 3
        or getattr(descriptor, "readout_layers", 1) != 1
        or getattr(descriptor, "so3_readout", "") != "mlp"
    ):
        return None
    if any(
        int(getattr(block, "lmax", 0)) != 3
        or int(getattr(block, "node_lmax", 0)) != 3
        or int(getattr(block, "mmax", 0)) != 1
        or int(getattr(block, "kmax", 0)) != 1
        or int(getattr(block, "mixing_layers", 0)) != 4
        or int(getattr(block, "n_atten_head", 0)) != 1
        or bool(getattr(block, "edge_cartesian", True))
        or bool(getattr(block, "atten_f_mix", True))
        or bool(getattr(block, "use_atten_v_proj", True))
        or bool(getattr(block, "use_atten_o_proj", True))
        or bool(getattr(block, "node_wise_s2", True))
        or bool(getattr(block, "node_wise_so3", True))
        or not bool(getattr(block, "message_node_so3", False))
        or bool(getattr(block, "message_node_grid_mlp", True))
        or int(getattr(block, "message_node_grid_branch", 0)) != 0
        or not bool(getattr(block, "ffn_so3_grid", False))
        or int(getattr(block, "ffn_grid_branch", 0)) != 1
        for block in descriptor.blocks
    ):
        return None
    try:
        from mdescriptor import _native as native

        if not hasattr(native, "Dpa4Calculator"):
            return None
    except ImportError:
        return None

    variables = descriptor._variables()
    projector = descriptor.blocks[0].so2_conv.message_node_grid_product.projector
    grid_to = _as_float32(projector.to_grid_mat).reshape(-1)
    grid_from = _as_float32(projector.from_grid_mat).reshape(-1)
    kernels = descriptor.wigner_calc.small_order_kernels

    def value(name: str) -> np.ndarray:
        if name not in variables:
            raise KeyError(name)
        return _as_float32(variables[name]).reshape(-1)

    def norm_values(
        block: Any,
        attr_name: str,
        variable_prefix: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
        norm = getattr(block, attr_name)
        if norm.__class__.__name__ == "Identity":
            return (
                np.empty(0, dtype=np.float32),
                np.empty(0, dtype=np.float32),
                np.empty(0, dtype=np.float32),
                False,
            )
        base = f"{variable_prefix}{attr_name}."
        if base + "adam_scale" not in variables:
            # Identity normalizers are omitted from the flattened checkpoint
            # variable map by the vendor runtime.  Keep the ABI explicit and
            # let the native core treat missing tensors as a no-op.
            return (
                np.empty(0, dtype=np.float32),
                np.empty(0, dtype=np.float32),
                np.empty(0, dtype=np.float32),
                False,
            )
        return (
            value(base + "adam_scale"),
            value(base + "bias"),
            value(base + "balance_weight"),
            True,
        )

    blocks: list[dict[str, Any]] = []
    for block_index, block in enumerate(descriptor.blocks):
        prefix = f"blocks.{block_index}."
        pre_scale, pre_bias, pre_balance, pre_enabled = norm_values(
            block, "pre_so2_norm", prefix
        )
        post_scale, post_bias, post_balance, post_enabled = norm_values(
            block, "post_so2_norm", prefix
        )
        ffn_norm = block.pre_ffn_norms[0]
        if ffn_norm.__class__.__name__ == "Identity":
            ffn_scale = np.empty(0, dtype=np.float32)
            ffn_bias = np.empty(0, dtype=np.float32)
            ffn_balance = np.empty(0, dtype=np.float32)
            ffn_enabled = False
        else:
            ffn_scale = value(prefix + "pre_ffn_norms.0.adam_scale")
            ffn_bias = value(prefix + "pre_ffn_norms.0.bias")
            ffn_balance = value(prefix + "pre_ffn_norms.0.balance_weight")
            ffn_enabled = True
        so2_m0 = [value(prefix + f"so2_conv.so2_linears.{i}.weight_m0") for i in range(4)]
        so2_m1 = [value(prefix + f"so2_conv.so2_linears.{i}.weight_m.0") for i in range(4)]
        so2_gates = [
            value(prefix + f"so2_conv.non_linearities.{i}.gate_linear.weight")
            for i in range(3)
        ]
        blocks.append(
            {
                "pre_norm_enabled": pre_enabled,
                "post_norm_enabled": post_enabled,
                "ffn_norm_enabled": ffn_enabled,
                "pre_norm_scale": pre_scale,
                "pre_norm_bias": pre_bias,
                "pre_norm_balance": pre_balance,
                "post_norm_scale": post_scale,
                "post_norm_bias": post_bias,
                "post_norm_balance": post_balance,
                "ffn_norm_scale": ffn_scale,
                "ffn_norm_bias": ffn_bias,
                "ffn_norm_balance": ffn_balance,
                "pre_focus_weight": value(prefix + "so2_conv.pre_focus_mix.weight"),
                "post_focus_weight": value(prefix + "so2_conv.post_focus_mix.weight"),
                "radial_mixer_weight": value(prefix + "so2_conv.radial_degree_mixer.weight"),
                "radial_channel_basis": value(prefix + "so2_conv.radial_degree_mixer.channel_basis"),
                "so2_weight_m0": so2_m0,
                "so2_weight_m1": so2_m1,
                "so2_gate_weight": so2_gates,
                "attn_qk_scale": value(prefix + "so2_conv.attn_qk_norm.adam_scale"),
                "attn_q_weight": value(prefix + "so2_conv.attn_q_proj.weight"),
                "attn_k_weight": value(prefix + "so2_conv.attn_k_proj.weight"),
                "attn_output_gate_scale": value(prefix + "so2_conv.attn_output_gate_norm.adam_scale"),
                "attn_logit_weight": value(prefix + "so2_conv.adamw_attn_logit_w"),
                "attn_z_bias_raw": value(prefix + "so2_conv.adamw_attn_z_bias_raw"),
                "attn_gate_weight": value(prefix + "so2_conv.adamw_attn_gate_w"),
                "message_scalar_gate": value(prefix + "so2_conv.message_node_grid_product.scalar_gate.weight"),
                "message_frame_expand": value(prefix + "so2_conv.message_node_grid_product.frame_expand.weight"),
                "message_frame_contract": value(prefix + "so2_conv.message_node_grid_product.frame_contract.weight"),
                "message_residual_scale": value(prefix + "so2_conv.message_node_grid_product.residual_scale"),
                "ffn_linear1": value(prefix + "ffns.0.so3_linear_1.weight"),
                "ffn_linear2": value(prefix + "ffns.0.so3_linear_2.weight"),
                "ffn_scalar_gate": value(prefix + "ffns.0.act.scalar_gate.weight"),
                "ffn_grid_left": value(prefix + "ffns.0.act.grid_op.left_proj.weight"),
                "ffn_grid_right": value(prefix + "ffns.0.act.grid_op.right_proj.weight"),
                "ffn_grid_router": value(prefix + "ffns.0.act.grid_op.router.weight"),
                "ffn_grid_out": value(prefix + "ffns.0.act.grid_op.out_proj.weight"),
            }
        )

    return {
        "rcut": float(descriptor.rcut),
        "ntypes": int(descriptor.ntypes),
        "channels": int(descriptor.channels),
        "n_radial": int(descriptor.n_radial),
        "num_threads": int(num_threads),
        "type_embedding": _as_float32(descriptor.type_embedding.call()),
        "env_rbf_layer1": value("env_seed_embedding.rbf_proj_layer1.matrix"),
        "env_rbf_layer2": value("env_seed_embedding.rbf_proj_layer2.matrix"),
        "env_type_embedding": value("env_seed_embedding.env_type_embed.adam_type_embedding"),
        "env_g_layer1": value("env_seed_embedding.g_layer1.matrix"),
        "env_g_layer2": value("env_seed_embedding.g_layer2.matrix"),
        "env_output_projection": value("env_seed_embedding.output_proj.matrix"),
        "film_scale_norm": value("film_scale_norm.adam_scale"),
        "film_shift_norm": value("film_shift_norm.adam_scale"),
        "film_scale_strength_log": _as_float32(value("film_scale_strength_log")),
        "film_shift_strength_log": _as_float32(value("film_shift_strength_log")),
        "radial_freqs": value("radial_basis.adam_freqs"),
        "radial_layer1": value("radial_embedding.net.0.matrix"),
        "radial_norm_scale": value("radial_embedding.net.1.adam_scale"),
        "radial_layer2": value("radial_embedding.net.3.matrix"),
        "wigner_l2_tensor": _as_float32(kernels.C_l2).reshape(-1),
        "wigner_l3_coefficients": _as_float32(kernels.C_l3).reshape(-1),
        "wigner_l3_exponents": _as_int64(kernels.exp_l3).reshape(-1),
        "gie_row_index": _as_int64(variables["gie.non_scalar_row_index"]),
        "gie_m0_index": _as_int64(variables["gie.zonal_m0_col_index_for_row"]),
        "gie_radial_index": _as_int64(variables["gie.radial_slot_index_for_row"]),
        "grid_to": grid_to,
        "grid_from": grid_from,
        "blocks": blocks,
        "output_linear1": value("output_ffn.so3_linear_1.weight"),
        "output_linear2": value("output_ffn.so3_linear_2.weight"),
        "output_scalar_gate": value("output_ffn.act.scalar_gate.weight"),
        "output_grid_left": value("output_ffn.act.grid_op.left_proj.weight"),
        "output_grid_right": value("output_ffn.act.grid_op.right_proj.weight"),
        "output_grid_out": value("output_ffn.act.grid_op.out_proj.weight"),
    }


class Dpa4Kernel(DpaKernelBase):
    """Compute DPA4 through the native core when its default graph is supported."""

    name = "DPA4"
    checkpoint_descriptor = "DPA4"
    runtime_descriptor_name = "DescrptDPA4"
    native_calculator_name = "Dpa4Calculator"
    default_model = DPA4_MODEL

    def __init__(
        self,
        model_path: str | Path | None = None,
        num_threads: int | None = None,
        _checkpoint: Mapping[str, Any] | None = None,
    ) -> None:
        self._initialize_dpa(model_path, num_threads, _checkpoint)

    def _native_payload(
        self,
        descriptor: Any,
        *,
        num_threads: int,
    ) -> dict[str, Any] | None:
        return _native_payload(descriptor, num_threads=num_threads)

    def _labels(self) -> tuple[str, ...]:
        return tuple(
            f"dpa4:scalar,channel={index}"
            for index in range(int(self._native.descriptor.channels))
        )

    def _metadata(self) -> dict[str, Any]:
        descriptor = self._native.descriptor
        metadata = self._metadata_base(descriptor)
        metadata.update(
            {
                "device": "cpu",
                "feature_count": self.feature_count,
                "num_threads": self._metadata_num_threads,
            }
        )
        return metadata


__all__ = ["Dpa4Kernel"]
