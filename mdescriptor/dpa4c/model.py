"""Torch inference core for the DPA4C descriptor.

This module contains the small, inference-only part of the DPA4C graph
equations needed by MDescriptor.  It deliberately does not import DeepMD:
the checkpoint loader consumes the ordinary state dictionary emitted by the
PyTorch Exportable model and the forward pass uses only PyTorch tensor ops.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence

import torch
from torch import Tensor, nn


SUPPORTED_CHANNELS = (8, 16, 32, 64, 128)
SUPPORTED_LMAX = (2, 3, 4)


def _degree_channels(channels: int, lmax: int) -> list[int]:
    if channels not in SUPPORTED_CHANNELS:
        raise ValueError(f"channels must be one of {SUPPORTED_CHANNELS}, got {channels}")
    if lmax not in SUPPORTED_LMAX:
        raise ValueError(f"lmax must be one of {SUPPORTED_LMAX}, got {lmax}")
    exponent = channels.bit_length() - 1
    degree_one = max(4, 1 << ((exponent + 1) // 2))
    degree_two = max(4, degree_one >> 1)
    return [channels, degree_one, degree_two] + [1] * (lmax - 2)


def _degree_offsets(widths: Sequence[int]) -> tuple[int, ...]:
    offsets = [0]
    for degree, width in enumerate(widths):
        offsets.append(offsets[-1] + (2 * degree + 1) * int(width))
    return tuple(offsets)


def _moment_indices(widths: Sequence[int]) -> tuple[Tensor, Tensor]:
    channel_index: list[int] = []
    harmonic_index: list[int] = []
    for degree, width in enumerate(widths):
        for component in range(2 * degree + 1):
            channel_index.extend(range(width))
            harmonic_index.extend([degree * degree + component] * width)
    return torch.tensor(channel_index, dtype=torch.long), torch.tensor(harmonic_index, dtype=torch.long)


def _swiglu(value: Tensor) -> Tensor:
    width = (value.shape[-1] + 1) // 2
    gate = value[..., :width]
    return torch.nn.functional.silu(gate) * value[..., width:]


def _linear(value: Tensor, weight: Tensor) -> Tensor:
    return torch.matmul(value, weight)


def _swiglu_mlp(value: Tensor, weights: Sequence[Tensor]) -> tuple[Tensor, Tensor]:
    """Return the final output and the activated hidden state."""

    hidden = _swiglu(_linear(value, weights[0]))
    output = _linear(hidden, weights[1])
    return output, hidden


def _angular_basis(direction: Tensor, lmax: int) -> Tensor:
    """Build the normalized real Cartesian harmonics used by DPA4C."""

    x, y, z = direction[:, 0], direction[:, 1], direction[:, 2]
    squared_norm = x * x + y * y + z * z
    blocks = [torch.ones_like(x)[:, None]]
    if lmax >= 1:
        blocks.append(torch.stack([x, y, z], dim=-1))
    if lmax >= 2:
        sqrt_three = math.sqrt(3.0)
        blocks.append(
            torch.stack(
                [
                    sqrt_three * x * y,
                    sqrt_three * y * z,
                    0.5 * (3.0 * z * z - squared_norm),
                    sqrt_three * x * z,
                    0.5 * sqrt_three * (x * x - y * y),
                ],
                dim=-1,
            )
        )
    if lmax >= 3:
        blocks.append(
            torch.stack(
                [
                    math.sqrt(5.0 / 8.0) * y * (3.0 * x * x - y * y),
                    math.sqrt(15.0) * x * y * z,
                    math.sqrt(3.0 / 8.0) * y * (5.0 * z * z - squared_norm),
                    0.5 * z * (5.0 * z * z - 3.0 * squared_norm),
                    math.sqrt(3.0 / 8.0) * x * (5.0 * z * z - squared_norm),
                    0.5 * math.sqrt(15.0) * z * (x * x - y * y),
                    math.sqrt(5.0 / 8.0) * x * (x * x - 3.0 * y * y),
                ],
                dim=-1,
            )
        )
    if lmax >= 4:
        z_squared = z * z
        x2_minus_y2 = x * x - y * y
        blocks.append(
            torch.stack(
                [
                    0.5 * math.sqrt(35.0) * x * y * x2_minus_y2,
                    0.25 * math.sqrt(70.0) * y * z * (3.0 * x * x - y * y),
                    0.5 * math.sqrt(5.0) * x * y * (7.0 * z_squared - squared_norm),
                    0.25 * math.sqrt(10.0) * y * z * (7.0 * z_squared - 3.0 * squared_norm),
                    0.125 * (
                        35.0 * z_squared * z_squared
                        - 30.0 * z_squared * squared_norm
                        + 3.0 * squared_norm * squared_norm
                    ),
                    0.25 * math.sqrt(10.0) * x * z * (7.0 * z_squared - 3.0 * squared_norm),
                    0.25 * math.sqrt(5.0) * x2_minus_y2 * (7.0 * z_squared - squared_norm),
                    0.25 * math.sqrt(70.0) * x * z * (x * x - 3.0 * y * y),
                    0.125 * math.sqrt(35.0) * (x**4 - 6.0 * x * x * y * y + y**4),
                ],
                dim=-1,
            )
        )
    return torch.cat(blocks, dim=-1)


def _packed_l2_to_stf(packed: Tensor) -> Tensor:
    inv_sqrt_two = 1.0 / math.sqrt(2.0)
    inv_sqrt_six = 1.0 / math.sqrt(6.0)
    q0, q1, q2, q3, q4 = (packed[..., index] for index in range(5))
    qxy = q0 * inv_sqrt_two
    qyz = q1 * inv_sqrt_two
    qxz = q3 * inv_sqrt_two
    qxx = -q2 * inv_sqrt_six + q4 * inv_sqrt_two
    qyy = -q2 * inv_sqrt_six - q4 * inv_sqrt_two
    qzz = 2.0 * q2 * inv_sqrt_six
    return torch.stack(
        [
            torch.stack([qxx, qxy, qxz], dim=-1),
            torch.stack([qxy, qyy, qyz], dim=-1),
            torch.stack([qxz, qyz, qzz], dim=-1),
        ],
        dim=-2,
    )


def _triples(lmax: int) -> tuple[tuple[int, int, int], ...]:
    return tuple(
        (first, second, third)
        for first in range(1, lmax + 1)
        for second in range(first, lmax + 1)
        for third in range(second, lmax + 1)
        if third <= first + second and (first + second + third) % 2 == 0
    )


def _coupling_offsets(lmax: int) -> tuple[int, ...]:
    offsets = [0]
    for first, second, third in _triples(lmax):
        offsets.append(offsets[-1] + (2 * first + 1) * (2 * second + 1) * (2 * third + 1))
    return tuple(offsets)


@dataclass(frozen=True)
class DPA4CConfig:
    rcut: float
    ntypes: int
    channels: int
    lmax: int
    basis_type: str
    n_radial: int
    radial_modes: int
    use_amp: bool
    precision: str
    trainable: bool
    seed: int | tuple[int, ...] | None
    type_map: tuple[str, ...]
    use_spin: tuple[bool, ...] | None
    add_chg_spin_ebd: bool
    default_chg_spin: tuple[float, float] | None
    exclude_types: tuple[tuple[int, int], ...]


def _torch_precision(precision: str) -> torch.dtype:
    normalized = str(precision).lower()
    aliases = {
        "half": "float16",
        "single": "float32",
        "double": "float64",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized == "default":
        normalized = "float64"
    try:
        return {
            "float16": torch.float16,
            "float32": torch.float32,
            "float64": torch.float64,
            "bfloat16": torch.bfloat16,
        }[normalized]
    except KeyError as exc:
        raise ValueError(f"unsupported DPA4C precision {precision!r}") from exc


class DPA4CModel(nn.Module):
    """Frozen DPA4C descriptor reconstructed from a PyTorch checkpoint."""

    def __init__(self, config: DPA4CConfig, state: dict[str, Tensor]) -> None:
        super().__init__()
        self.config = config
        self.compute_dtype = _torch_precision(config.precision)
        if config.rcut <= 0.0 or config.n_radial <= 0 or config.radial_modes < 0:
            raise ValueError("invalid DPA4C cutoff, radial basis count, or radial mode count")
        if config.basis_type not in {"bessel", "gaussian"}:
            raise ValueError("DPA4C basis_type must be 'bessel' or 'gaussian'")
        self.widths = _degree_channels(config.channels, config.lmax)
        self.offsets = _degree_offsets(self.widths)
        self.register_buffer("type_embedding", state["type_embedding"].to(self.compute_dtype))
        self.register_buffer("radial_freqs", state["radial_basis.adam_freqs"].to(self.compute_dtype))
        self.register_buffer("radial_w0", state["radial_embedding.layers.0.w"].to(self.compute_dtype))
        self.register_buffer("radial_w1", state["radial_embedding.layers.1.w"].to(self.compute_dtype))
        self.register_buffer(
            "radial_mode_w",
            state.get(
                "radial_mode_head.w",
                torch.empty((self.radial_w0.shape[1] // 2, config.radial_modes), dtype=self.compute_dtype),
            ).to(self.compute_dtype),
        )
        self.register_buffer("pair_w0", state["pair_film.network.layers.0.w"].to(self.compute_dtype))
        self.register_buffer("pair_w1", state["pair_film.network.layers.1.w"].to(self.compute_dtype))
        self.register_buffer("gram_index", state["readout.gram_index"].to(torch.long))
        self.register_buffer("gram_scale", state["readout.gram_scale"].to(self.compute_dtype))
        self.register_buffer("bispectrum_coupling", state["readout.bispectrum_coupling"].to(self.compute_dtype))
        self.register_buffer("probe_index", state["readout.probe_index"].to(torch.long))
        self.register_buffer("probe_scale", state["readout.probe_scale"].to(self.compute_dtype))
        self.register_buffer("alignment_1", state["readout.channel_alignment.0.w"].to(self.compute_dtype))
        self.register_buffer("alignment_2", state["readout.channel_alignment.1.w"].to(self.compute_dtype))
        self.register_buffer(
            "probe_1",
            state.get(
                "readout.probe_projections.0.w",
                torch.eye(self.widths[1], dtype=self.compute_dtype),
            ).to(self.compute_dtype),
        )
        self.register_buffer(
            "probe_2",
            state.get(
                "readout.probe_projections.1.w",
                torch.eye(self.widths[2], dtype=self.compute_dtype),
            ).to(self.compute_dtype),
        )
        self.register_buffer("mean", state["mean"].to(self.compute_dtype))
        self.register_buffer("stddev", state["stddev"].to(self.compute_dtype))
        self.register_buffer("emask", state.get("emask.type_mask", torch.ones((config.ntypes + 1) ** 2, dtype=torch.int32)).to(torch.bool))
        channel_index, harmonic_index = _moment_indices(self.widths)
        self.register_buffer("channel_index", channel_index[self.config.channels :])
        self.register_buffer("harmonic_index", harmonic_index[self.config.channels :])

        self.spin_channels = 0 if config.use_spin is None else self.widths[2]
        if self.spin_channels:
            if len(config.use_spin or ()) != config.ntypes:
                raise ValueError(
                    "DPA4C use_spin must contain one flag per checkpoint type"
                )
            self.register_buffer(
                "spin_scale_anchor",
                state["pair_film.adam_spin_scale_anchor"].to(self.compute_dtype),
            )
            self.register_buffer(
                "spin_shift_anchor",
                state["pair_film.adam_spin_shift_anchor"].to(self.compute_dtype),
            )
            self.register_buffer(
                "spin_vector_weight",
                state["spin.adam_spin_vector_weight"].to(self.compute_dtype),
            )
            self.register_buffer(
                "spin_quadrupole_weight",
                state["spin.adam_spin_quadrupole_weight"].to(self.compute_dtype),
            )
            self.register_buffer("spin_gate", state["spin.spin_gate"].to(self.compute_dtype))
            self.register_buffer("spin_reference", state["spin.spin_reference"].to(self.compute_dtype))
            self.register_buffer("spin_vector_index", state["spin.vector_gram_index"].to(torch.long))
            self.register_buffer("spin_vector_scale", state["spin.vector_gram_scale"].to(self.compute_dtype))
            self.register_buffer("spin_quadrupole_index", state["spin.quadrupole_gram_index"].to(torch.long))
            self.register_buffer("spin_quadrupole_scale", state["spin.quadrupole_gram_scale"].to(self.compute_dtype))
            spin_mask = [1.0 if value else 0.0 for value in config.use_spin] + [0.0]
            self.register_buffer("spin_mask", torch.tensor(spin_mask, dtype=self.compute_dtype))

        self.charge_enabled = bool(config.add_chg_spin_ebd)
        if self.charge_enabled:
            self.register_buffer(
                "charge_embedding",
                state["charge_spin_embedding.charge_embedding.adam_type_embedding"].to(self.compute_dtype),
            )
            self.register_buffer(
                "multiplicity_embedding",
                state["charge_spin_embedding.spin_embedding.adam_type_embedding"].to(self.compute_dtype),
            )
            self.register_buffer(
                "charge_w0",
                state["charge_spin_embedding.network.layers.0.w"].to(self.compute_dtype),
            )
            self.register_buffer(
                "charge_w1",
                state["charge_spin_embedding.network.layers.1.w"].to(self.compute_dtype),
            )

        # OrderedPairFiLM is independent of geometry for the usual (uncharged)
        # descriptor.  Keep its finite type-pair table out of the edge path:
        # the raw checkpoint contains only the network weights, but evaluating
        # that network once per edge turns a 1.3M-edge batch into the dominant
        # CPU cost.  This cache is intentionally lazy so it is created after
        # the model has reached its requested device.
        self._pair_cache: tuple[
            Tensor,
            Tensor,
            Tensor | None,
            Tensor | None,
            Tensor | None,
        ] | None = None

        expected = self.get_dim_out()
        if self.mean.numel() != expected or self.stddev.numel() != expected:
            raise ValueError(
                f"DPA4C calibration width is {self.mean.numel()}, expected {expected}"
            )

    @property
    def feature_count(self) -> int:
        return self.get_dim_out()

    def get_dim_out(self) -> int:
        gram_dim = sum(width * (width + 1) // 2 for width in self.widths[1:])
        geometric = (
            self.config.channels
            + gram_dim
            + int(self.probe_index.numel())
            + self.widths[2] * 2
        )
        spin = 0
        if self.spin_channels:
            quadrupole_width = 2
            spin = (
                int(self.spin_vector_index.numel())
                + int(self.spin_quadrupole_index.numel())
                + quadrupole_width * self.widths[2]
                + 2 * self.spin_channels
            )
        return geometric + spin + 2 + self.config.channels

    def _cutoff(self, distance: Tensor) -> Tensor:
        u = torch.clamp((self.config.rcut - distance) / self.config.rcut, 0.0, 1.0)
        x = 1.0 - u
        series = 1.0 + x * (4.0 + x * (10.0 + x * (20.0 + x * 35.0)))
        return u**4 * series

    def _radial(self, distance: Tensor) -> tuple[Tensor, Tensor]:
        freqs = self.radial_freqs
        if self.config.basis_type == "bessel":
            radial_basis = freqs * torch.sinc(distance * freqs / math.pi)
        else:
            width = self.config.rcut / max(self.config.n_radial - 1, 1)
            radial_basis = torch.exp((distance - freqs) ** 2 * (-0.5 / (width * width)))
        radial_output, radial_hidden = _swiglu_mlp(radial_basis, (self.radial_w0, self.radial_w1))
        return radial_output, radial_hidden

    def _charge_conditioning(self, charge_spin: Tensor) -> tuple[Tensor, Tensor]:
        charge_index = charge_spin[:, 0].to(torch.long) + 100
        multiplicity_index = charge_spin[:, 1].to(torch.long)
        pair_input = torch.cat(
            [self.charge_embedding[charge_index], self.multiplicity_embedding[multiplicity_index]],
            dim=-1,
        )
        hidden = _swiglu(_linear(pair_input, self.charge_w0))
        output = _linear(hidden, self.charge_w1)
        return output[:, : self.config.channels], output[:, self.config.channels :]

    def _pair_conditioning(
        self,
        center_type: Tensor,
        neighbor_type: Tensor,
        hidden_bias: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor | None, Tensor | None]:
        table = self.type_embedding
        pair_input = torch.cat([table[center_type], table[neighbor_type]], dim=-1)
        pair_hidden_affine = _linear(pair_input, self.pair_w0)
        if hidden_bias is not None:
            pair_hidden_affine = pair_hidden_affine + hidden_bias
        pair_hidden = _swiglu(pair_hidden_affine)
        # OrderedPairFiLM uses a fixed 0.1 output scale on its SwiGLU head.
        logits = _linear(pair_hidden, self.pair_w1) * 0.1
        channels = self.config.channels
        shift_end = 2 * channels
        mixing_end = shift_end + channels * self.config.radial_modes
        base_shift = table[center_type] + table[neighbor_type]
        scale = 1.0 + torch.tanh(logits[:, :channels])
        shift = base_shift + torch.tanh(logits[:, channels:shift_end])
        if self.config.radial_modes:
            mixing = torch.tanh(logits[:, shift_end:mixing_end]).reshape(
                -1, channels, self.config.radial_modes
            )
        else:
            mixing = None
        if self.spin_channels:
            spin_scale_end = mixing_end + self.spin_channels
            spin_scale = torch.tanh(
                logits[:, mixing_end:spin_scale_end] + self.spin_scale_anchor
            )
            spin_shift = torch.tanh(
                logits[:, spin_scale_end:] + self.spin_shift_anchor
            )
        else:
            spin_scale = None
            spin_shift = None
        return scale, shift, mixing, spin_scale, spin_shift

    def _pair_conditioning_cache(
        self,
    ) -> tuple[Tensor, Tensor, Tensor | None, Tensor | None, Tensor | None]:
        """Build the condition-independent ordered type-pair cache once."""

        cache = self._pair_cache
        if cache is not None and cache[0].device == self.type_embedding.device:
            return cache
        type_count = self.config.ntypes + 1
        center_type = torch.arange(
            type_count,
            dtype=torch.long,
            device=self.type_embedding.device,
        ).repeat_interleave(type_count)
        neighbor_type = torch.arange(
            type_count,
            dtype=torch.long,
            device=self.type_embedding.device,
        ).repeat(type_count)
        cache = self._pair_conditioning(center_type, neighbor_type)
        self._pair_cache = cache
        return cache

    def _readout(self, moments: Tensor) -> Tensor:
        blocks = [
            moments[:, self.offsets[degree] : self.offsets[degree + 1]].reshape(
                moments.shape[0], 2 * degree + 1, self.widths[degree]
            )
            for degree in range(self.config.lmax + 1)
        ]
        aligned = list(blocks)
        aligned[1] = _linear(blocks[1], self.alignment_1) + blocks[1]
        aligned[2] = _linear(blocks[2], self.alignment_2) + blocks[2]
        projected = [
            _linear(aligned[1], self.probe_1),
            _linear(aligned[2], self.probe_2),
        ]
        projected.extend(aligned[3:])
        ranks = [self.widths[2], 2] + [1] * (self.config.lmax - 2)

        grams: list[Tensor] = []
        gram_offsets = [0]
        for width in self.widths[1:]:
            gram_offsets.append(gram_offsets[-1] + width * (width + 1) // 2)
        for degree, block in enumerate(aligned[1:]):
            gram = torch.matmul(block.transpose(1, 2), block).reshape(moments.shape[0], -1)
            start, end = gram_offsets[degree], gram_offsets[degree + 1]
            grams.append(gram[:, self.gram_index[start:end]] * self.gram_scale[start:end])

        bispectrum_parts: list[Tensor] = []
        coupling_offsets = _coupling_offsets(self.config.lmax)
        probe_offsets = [0]
        for first, second, third in _triples(self.config.lmax):
            rank_1, rank_2, rank_3 = ranks[first - 1], ranks[second - 1], ranks[third - 1]
            if first == second == third:
                count = rank_1 * (rank_1 + 1) * (rank_1 + 2) // 6
            elif first == second:
                count = rank_1 * (rank_1 + 1) // 2 * rank_3
            elif second == third:
                count = rank_1 * rank_2 * (rank_2 + 1) // 2
            else:
                count = rank_1 * rank_2 * rank_3
            probe_offsets.append(probe_offsets[-1] + count)

        for triple_index, (first, second, third) in enumerate(_triples(self.config.lmax)):
            if (first, second, third) == (1, 1, 2):
                vectors = projected[0].permute(0, 2, 1)
                tensors = _packed_l2_to_stf(projected[1].permute(0, 2, 1))
                tensor_vector = torch.matmul(
                    tensors[:, :, None, :, :], vectors[:, None, :, :, None]
                ).reshape(moments.shape[0], projected[1].shape[-1], projected[0].shape[-1], 3)
                full = torch.matmul(
                    vectors[:, None, :, :], tensor_vector.permute(0, 1, 3, 2)
                )
                full = full.permute(0, 2, 3, 1).reshape(moments.shape[0], -1)
                full = full * (-1.0 / math.sqrt(5.0))
                quartic = (tensor_vector * tensor_vector).sum(dim=-1).reshape(moments.shape[0], -1)
            else:
                start, end = coupling_offsets[triple_index], coupling_offsets[triple_index + 1]
                coupling = self.bispectrum_coupling[start:end].reshape(
                    2 * first + 1, 2 * second + 1, 2 * third + 1
                )
                value_1, value_2, value_3 = (projected[degree - 1] for degree in (first, second, third))
                n_nodes = moments.shape[0]
                rank_1, rank_2, rank_3 = value_1.shape[-1], value_2.shape[-1], value_3.shape[-1]
                first_part = torch.matmul(value_1.permute(0, 2, 1), coupling.reshape(coupling.shape[0], -1))
                first_part = first_part.reshape(n_nodes, rank_1, coupling.shape[1], coupling.shape[2])
                first_part = first_part.permute(0, 1, 3, 2).reshape(n_nodes, rank_1 * coupling.shape[2], coupling.shape[1])
                second_part = torch.matmul(first_part, value_2)
                second_part = second_part.reshape(n_nodes, rank_1, coupling.shape[2], rank_2)
                second_part = second_part.permute(0, 1, 3, 2).reshape(n_nodes, rank_1 * rank_2, coupling.shape[2])
                full = torch.matmul(second_part, value_3).reshape(n_nodes, rank_1 * rank_2 * rank_3)
            start, end = probe_offsets[triple_index], probe_offsets[triple_index + 1]
            bispectrum_parts.append(full[:, self.probe_index[start:end]] * self.probe_scale[start:end])

        return torch.cat([blocks[0][:, 0, :], *grams, *bispectrum_parts, quartic], dim=-1)

    def _spin_edge_payload(
        self,
        conditioned_spin: Tensor,
        atype: Tensor,
        src: Tensor,
        direction: Tensor,
        radial: Tensor,
        envelope: Tensor,
        spin_scale: Tensor,
        spin_shift: Tensor,
    ) -> Tensor:
        channels = self.spin_channels
        neighbor_spin = conditioned_spin[src]
        spin_amplitude = (
            radial[:, :channels] * spin_scale + spin_shift
        ) * envelope.square()[:, None]
        neighbor_gate = self.spin_mask[atype[src]][:, None]
        magnitude = (neighbor_spin * neighbor_spin).sum(dim=-1, keepdim=True)
        bond_spin = direction * (neighbor_spin * direction).sum(dim=-1, keepdim=True)
        quadrupole = _angular_basis(neighbor_spin, 2)[:, 4:9]
        return torch.cat(
            [
                spin_amplitude * magnitude,
                spin_amplitude * neighbor_gate,
                (neighbor_spin[:, :, None] * spin_amplitude[:, None, :]).reshape(
                    -1, 3 * channels
                ),
                (bond_spin[:, :, None] * spin_amplitude[:, None, :]).reshape(
                    -1, 3 * channels
                ),
                (
                    quadrupole[:, :, None]
                    * spin_amplitude[:, None, :1]
                ).reshape(-1, 5),
            ],
            dim=-1,
        )

    def _spin_onsite_payload(self, conditioned_spin: Tensor, atype: Tensor) -> Tensor:
        vector_weight = self.spin_vector_weight[atype]
        quadrupole_weight = self.spin_quadrupole_weight[atype]
        quadrupole = _angular_basis(conditioned_spin, 2)[:, 4:9]
        return torch.cat(
            [
                conditioned_spin * vector_weight[:, None],
                quadrupole * quadrupole_weight[:, None],
            ],
            dim=-1,
        )

    def _spin_readout(self, spin_moments: Tensor, degree_two: Tensor) -> Tensor:
        channels = self.spin_channels
        offset = 0
        magnitude = spin_moments[:, offset : offset + channels]
        offset += channels
        coordination = spin_moments[:, offset : offset + channels]
        offset += channels
        neighbor_vector = spin_moments[:, offset : offset + 3 * channels].reshape(
            spin_moments.shape[0], 3, channels
        )
        offset += 3 * channels
        neighbor_bond = spin_moments[:, offset : offset + 3 * channels].reshape(
            spin_moments.shape[0], 3, channels
        )
        offset += 3 * channels
        neighbor_tensor = spin_moments[:, offset : offset + 5].reshape(
            spin_moments.shape[0], 5, 1
        )
        offset += 5
        onsite_vector = spin_moments[:, offset : offset + 3]
        offset += 3
        onsite_tensor = spin_moments[:, offset : offset + 5]
        vector = torch.cat(
            [onsite_vector[:, :, None], neighbor_vector, neighbor_bond], dim=-1
        )
        quadrupole = torch.cat(
            [onsite_tensor[:, :, None], neighbor_tensor], dim=-1
        )

        def half_gram(block: Tensor, index: Tensor, scale: Tensor) -> Tensor:
            width = block.shape[-1]
            gram = torch.matmul(block.transpose(1, 2), block).reshape(
                block.shape[0], width * width
            )
            return gram[:, index] * scale

        return torch.cat(
            [
                half_gram(vector, self.spin_vector_index, self.spin_vector_scale),
                half_gram(
                    quadrupole,
                    self.spin_quadrupole_index,
                    self.spin_quadrupole_scale,
                ),
                torch.matmul(quadrupole.transpose(1, 2), degree_two).reshape(
                    spin_moments.shape[0], -1
                ),
                magnitude,
                coordination,
            ],
            dim=-1,
        )

    def forward(
        self,
        edge_vec: Tensor,
        src: Tensor,
        dst: Tensor,
        atype: Tensor,
        n_total: int,
        *,
        calibrate: bool = True,
        spin: Tensor | None = None,
        charge_spin: Tensor | None = None,
        frame_index: Tensor | None = None,
    ) -> Tensor:
        edge_vec = edge_vec.to(dtype=self.compute_dtype)
        src = src.to(dtype=torch.long)
        dst = dst.to(dtype=torch.long)
        atype = atype.to(dtype=torch.long)
        center_type = atype[dst]
        neighbor_type = atype[src]
        type_features = self.type_embedding[atype]
        edge_hidden_bias = None
        if self.charge_enabled:
            if charge_spin is None or frame_index is None:
                raise ValueError(
                    "a charge-conditioned DPA4C requires charge_spin and frame_index"
                )
            type_shift, frame_pair_bias = self._charge_conditioning(
                charge_spin.to(dtype=self.compute_dtype)
            )
            frame_index = frame_index.to(dtype=torch.long)
            type_features = type_features + type_shift[frame_index]
            edge_hidden_bias = frame_pair_bias[frame_index[dst]]
        distance = torch.sqrt((edge_vec * edge_vec).sum(dim=-1, keepdim=True) + 1.0e-14)
        direction = edge_vec / distance
        pair_index = center_type * (self.config.ntypes + 1) + neighbor_type
        edge_mask = self.emask[pair_index]
        envelope = self._cutoff(distance[:, 0]) * edge_mask.to(self.compute_dtype)
        radial, radial_hidden = self._radial(distance)
        if edge_hidden_bias is None:
            pair_cache = self._pair_conditioning_cache()
            scale = pair_cache[0].index_select(0, pair_index)
            shift = pair_cache[1].index_select(0, pair_index)
            mixing = (
                None
                if pair_cache[2] is None
                else pair_cache[2].index_select(0, pair_index)
            )
            spin_scale = (
                None
                if pair_cache[3] is None
                else pair_cache[3].index_select(0, pair_index)
            )
            spin_shift = (
                None
                if pair_cache[4] is None
                else pair_cache[4].index_select(0, pair_index)
            )
        else:
            scale, shift, mixing, spin_scale, spin_shift = self._pair_conditioning(
                center_type, neighbor_type, edge_hidden_bias
            )
        amplitude = radial * scale + shift
        if mixing is not None:
            modes = _linear(radial_hidden, self.radial_mode_w)
            amplitude = amplitude + (mixing * modes[:, None, :]).sum(dim=-1)
        amplitude = amplitude * envelope[:, None]
        basis = _angular_basis(direction, self.config.lmax) * edge_mask[:, None].to(self.compute_dtype)

        parts = [
            envelope.square()[:, None],
            envelope.square().square()[:, None],
            amplitude,
            amplitude[:, self.channel_index] * basis[:, self.harmonic_index] * envelope[:, None],
        ]
        conditioned_spin = None
        if self.spin_channels:
            if spin is None:
                raise ValueError("a spin-conditioned DPA4C requires per-atom spin vectors")
            spin = spin.to(dtype=self.compute_dtype)
            if spin.ndim != 2 or tuple(spin.shape) != (n_total, 3):
                raise ValueError(f"spin must have shape ({n_total}, 3)")
            conditioned_spin = spin * (
                self.spin_mask[atype] / self.spin_reference[atype]
            )[:, None]
            parts.append(
                self._spin_edge_payload(
                    conditioned_spin,
                    atype,
                    src,
                    direction,
                    radial,
                    envelope,
                    spin_scale,
                    spin_shift,
                )
            )
        payload = torch.cat(parts, dim=1)
        reduced = torch.zeros(
            (n_total, payload.shape[1]), dtype=payload.dtype, device=payload.device
        )
        reduced.index_add_(0, dst, payload)
        divisors = torch.sqrt(reduced[:, :2] + 0.25)
        scalar_end = 2 + self.config.channels
        geometric_moments = torch.cat(
            [
                reduced[:, 2:scalar_end] / divisors[:, :1],
                reduced[:, scalar_end:] / divisors[:, 1:],
            ],
            dim=-1,
        )
        spin_values = None
        if self.spin_channels:
            spin_edge_start = scalar_end + (self.offsets[-1] - self.config.channels)
            spin_edge_end = reduced.shape[1]
            spin_moments = reduced[:, spin_edge_start:spin_edge_end] / divisors[:, 1:]
            spin_moments = torch.cat(
                [
                    spin_moments,
                    self._spin_onsite_payload(conditioned_spin, atype),
                ],
                dim=-1,
            )
            degree_two = geometric_moments[
                :, self.offsets[2] : self.offsets[3]
            ].reshape(n_total, 5, self.widths[2])
            spin_values = self._spin_readout(spin_moments, degree_two)
        geometry_values = self._readout(geometric_moments)
        raw_parts = [geometry_values]
        if spin_values is not None:
            raw_parts.append(spin_values)
        raw_parts.extend([divisors, type_features])
        raw = torch.cat(raw_parts, dim=-1)
        if calibrate:
            raw = (raw - self.mean) / self.stddev
            if spin_values is not None:
                start = geometry_values.shape[-1]
                stop = start + spin_values.shape[-1]
                raw[:, start:stop] = raw[:, start:stop] * self.spin_gate
        return raw


def _descriptor_state(raw_state: dict[str, Any]) -> dict[str, Tensor]:
    state = raw_state.get("model", raw_state)
    result: dict[str, Tensor] = {}
    for key, value in state.items():
        marker = ".descriptor."
        if marker in key:
            name = key.split(marker, 1)[1]
            if name == "type_embedding.adam_type_embedding":
                name = "type_embedding"
            result[name] = value
    if not result:
        raise ValueError("checkpoint does not contain a DeepMD dpa4c descriptor")
    return result


def load_dpa4c_checkpoint(path: str, *, device: torch.device | str = "cpu") -> tuple[DPA4CModel, DPA4CConfig]:
    try:
        raw = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError as exc:
        raise RuntimeError("DPA4C requires a PyTorch version supporting weights_only checkpoint loading") from exc
    if not isinstance(raw, dict):
        raise ValueError("DPA4C checkpoint must contain a mapping")
    state = _descriptor_state(raw)
    model_state = raw.get("model", raw)
    extra = model_state.get("_extra_state", {}) if isinstance(model_state, dict) else {}
    model_params = extra.get("model_params", {}) if isinstance(extra, dict) else {}
    descriptor = model_params.get("descriptor", {})
    if descriptor.get("type") != "dpa4c":
        raise ValueError("checkpoint descriptor.type is not 'dpa4c'")
    type_map = tuple(str(value) for value in model_params.get("type_map", ()))
    if not type_map:
        type_map = tuple(f"type_{index}" for index in range(state["type_embedding"].shape[0] - 1))
    config = DPA4CConfig(
        rcut=float(descriptor["rcut"]),
        ntypes=len(type_map),
        channels=int(descriptor["channels"]),
        lmax=int(descriptor["lmax"]),
        basis_type=str(descriptor.get("basis_type", "bessel")).lower(),
        n_radial=int(descriptor["n_radial"]),
        radial_modes=int(descriptor.get("radial_modes", 0)),
        use_amp=bool(descriptor.get("use_amp", False)),
        precision=str(descriptor.get("precision", "float32")),
        trainable=bool(descriptor.get("trainable", True)),
        seed=descriptor.get("seed"),
        type_map=type_map,
        use_spin=None
        if descriptor.get("use_spin") is None
        else tuple(bool(v) for v in descriptor["use_spin"]),
        add_chg_spin_ebd=bool(descriptor.get("add_chg_spin_ebd", False)),
        default_chg_spin=None if descriptor.get("default_chg_spin") is None else tuple(float(v) for v in descriptor["default_chg_spin"]),
        exclude_types=tuple(tuple(int(v) for v in pair) for pair in descriptor.get("exclude_types", ())),
    )
    model = DPA4CModel(config, state).to(device)
    model.eval()
    return model, config


__all__ = ["DPA4CConfig", "DPA4CModel", "load_dpa4c_checkpoint"]
