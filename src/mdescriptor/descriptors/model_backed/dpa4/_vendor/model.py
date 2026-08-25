"""A small, self-contained SO(3)-equivariant DPA4 descriptor core.

The public calculator deliberately keeps this module behind a lazy PyTorch
boundary.  The implementation uses only PyTorch and does not import DeepMD.
Its checkpoint format is an MDescriptor-owned archive with a config and a
state dict; this keeps the model implementation independent from a training
runtime while making the serialization contract explicit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn


_FORMAT = "mdescriptor.dpa4.v1"
_DEGREE_WIDTHS = tuple(2 * degree + 1 for degree in range(4))


def _as_tuple(value: Any, *, name: str) -> tuple[Any, ...] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return (value,)
    return tuple(value)


@dataclass(frozen=True)
class Dpa4Config:
    """Configuration for the standalone DPA4 implementation.

    The default values are intentionally modest so a CPU inference smoke test
    remains cheap.  ``channels`` is also the output descriptor dimension.
    """

    ntypes: int
    type_map: tuple[str, ...]
    rcut: float = 6.0
    channels: int = 64
    lmax: int = 2
    n_radial: int = 16
    radial_modes: int = 1
    n_blocks: int = 3
    radial_hidden: int = 64
    basis_type: str = "bessel"
    precision: str = "float32"
    use_spin: tuple[bool, bool] | None = None
    add_chg_spin_ebd: bool = False
    default_chg_spin: tuple[float, float] | None = None
    exclude_types: tuple[tuple[int, int], ...] = field(default_factory=tuple)
    seed: int | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "Dpa4Config":
        raw = dict(value)
        type_map = _as_tuple(raw.get("type_map"), name="type_map")
        if type_map is None:
            ntypes = int(raw.get("ntypes", 0))
            type_map = tuple(str(index) for index in range(ntypes))
        ntypes = int(raw.get("ntypes", len(type_map)))
        if ntypes != len(type_map):
            raise ValueError("DPA4 ntypes must equal the length of type_map")
        use_spin = raw.get("use_spin")
        if use_spin is not None:
            use_spin = tuple(bool(item) for item in use_spin)
            if len(use_spin) != ntypes:
                raise ValueError("DPA4 use_spin must contain one flag per atom type")
        default_chg_spin = raw.get("default_chg_spin")
        if default_chg_spin is not None:
            default_chg_spin = tuple(float(item) for item in default_chg_spin)
            if len(default_chg_spin) != 2:
                raise ValueError("DPA4 default_chg_spin must be [charge, multiplicity]")
        excluded = tuple(
            tuple(int(item) for item in pair)
            for pair in (raw.get("exclude_types") or ())
        )
        result = cls(
            ntypes=ntypes,
            type_map=tuple(str(item) for item in type_map),
            rcut=float(raw.get("rcut", 6.0)),
            channels=int(raw.get("channels", 64)),
            lmax=int(raw.get("lmax", 2)),
            n_radial=int(raw.get("n_radial", 16)),
            radial_modes=int(raw.get("radial_modes", 1)),
            n_blocks=int(raw.get("n_blocks", 3)),
            radial_hidden=int(raw.get("radial_hidden", 64)),
            basis_type=str(raw.get("basis_type", "bessel")).lower(),
            precision=str(raw.get("precision", "float32")).lower(),
            use_spin=use_spin,
            add_chg_spin_ebd=bool(raw.get("add_chg_spin_ebd", False)),
            default_chg_spin=default_chg_spin,
            exclude_types=excluded,
            seed=None if raw.get("seed") is None else int(raw["seed"]),
        )
        result.validate()
        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            "ntypes": self.ntypes,
            "type_map": list(self.type_map),
            "rcut": self.rcut,
            "channels": self.channels,
            "lmax": self.lmax,
            "n_radial": self.n_radial,
            "radial_modes": self.radial_modes,
            "n_blocks": self.n_blocks,
            "radial_hidden": self.radial_hidden,
            "basis_type": self.basis_type,
            "precision": self.precision,
            "use_spin": None if self.use_spin is None else list(self.use_spin),
            "add_chg_spin_ebd": self.add_chg_spin_ebd,
            "default_chg_spin": self.default_chg_spin,
            "exclude_types": [list(pair) for pair in self.exclude_types],
            "seed": self.seed,
        }

    def validate(self) -> None:
        if self.ntypes <= 0:
            raise ValueError("DPA4 ntypes must be positive")
        if self.rcut <= 0:
            raise ValueError("DPA4 rcut must be positive")
        if self.channels <= 0 or self.n_radial <= 0 or self.n_blocks <= 0:
            raise ValueError("DPA4 channels, n_radial, and n_blocks must be positive")
        if self.radial_modes < 0:
            raise ValueError("DPA4 radial_modes must be non-negative")
        if self.radial_hidden <= 0:
            raise ValueError("DPA4 radial_hidden must be positive")
        if self.lmax < 0 or self.lmax > 3:
            raise ValueError("standalone DPA4 supports lmax in [0, 3]")
        if self.basis_type not in {"bessel", "gaussian"}:
            raise ValueError("DPA4 basis_type must be 'bessel' or 'gaussian'")
        if self.precision not in {"float32", "float64", "float16", "bfloat16"}:
            raise ValueError("unsupported DPA4 precision")
        for first, second in self.exclude_types:
            if first < 0 or first >= self.ntypes or second < 0 or second >= self.ntypes:
                raise ValueError("DPA4 exclude_types contains an invalid type index")


def _compute_dtype(precision: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "float64": torch.float64,
    }[precision]


def _smooth_cutoff(distance: Tensor, rcut: float) -> Tensor:
    """C3-like compact cutoff with zero value and slope at the boundary."""

    u = torch.clamp(1.0 - distance / rcut, min=0.0, max=1.0)
    return u * u * (3.0 - 2.0 * u)


def _real_angular_basis(direction: Tensor, lmax: int) -> list[Tensor]:
    """Return real spherical-harmonic blocks through degree three.

    The normalization makes the squared norm of each degree block rotationally
    invariant.  The formulas are polynomial Cartesian harmonics, so this path
    is differentiable and has no dependency on a spherical-harmonics package.
    """

    x, y, z = direction.unbind(dim=-1)
    x2, y2, z2 = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    result = [direction.new_ones((*direction.shape[:-1], 1)) * (1.0 / math.sqrt(4.0 * math.pi))]
    if lmax >= 1:
        result.append(torch.stack((x, y, z), dim=-1) * math.sqrt(3.0 / (4.0 * math.pi)))
    if lmax >= 2:
        result.append(
            torch.stack(
                (
                    math.sqrt(15.0 / (4.0 * math.pi)) * xy,
                    math.sqrt(15.0 / (4.0 * math.pi)) * yz,
                    math.sqrt(5.0 / (16.0 * math.pi)) * (2.0 * z2 - x2 - y2),
                    math.sqrt(15.0 / (4.0 * math.pi)) * xz,
                    math.sqrt(15.0 / (16.0 * math.pi)) * (x2 - y2),
                ),
                dim=-1,
            )
        )
    if lmax >= 3:
        result.append(
            torch.stack(
                (
                    math.sqrt(35.0 / (32.0 * math.pi)) * y * (3.0 * x2 - y2),
                    math.sqrt(105.0 / (4.0 * math.pi)) * xy * z,
                    math.sqrt(21.0 / (32.0 * math.pi)) * y * (4.0 * z2 - x2 - y2),
                    math.sqrt(7.0 / (16.0 * math.pi)) * z * (2.0 * z2 - 3.0 * x2 - 3.0 * y2),
                    math.sqrt(21.0 / (32.0 * math.pi)) * x * (4.0 * z2 - x2 - y2),
                    math.sqrt(105.0 / (16.0 * math.pi)) * (x2 - y2) * z,
                    math.sqrt(35.0 / (32.0 * math.pi)) * x * (x2 - 3.0 * y2),
                ),
                dim=-1,
            )
        )
    return result


class _RadialNetwork(nn.Module):
    def __init__(self, config: Dpa4Config, output_width: int) -> None:
        super().__init__()
        self.rcut = config.rcut
        self.n_radial = config.n_radial
        self.basis_type = config.basis_type
        self.register_buffer(
            "centers",
            torch.linspace(0.0, config.rcut, config.n_radial),
            persistent=False,
        )
        self.register_buffer(
            "frequencies",
            torch.arange(1, config.n_radial + 1, dtype=torch.get_default_dtype()) * math.pi / config.rcut,
            persistent=False,
        )
        self.network = nn.Sequential(
            nn.Linear(config.n_radial, config.radial_hidden),
            nn.SiLU(),
            nn.Linear(config.radial_hidden, output_width),
        )

    def forward(self, distance: Tensor, cutoff: Tensor) -> Tensor:
        if self.basis_type == "bessel":
            scaled = distance.unsqueeze(-1) * self.frequencies.to(distance)
            basis = torch.where(
                distance.unsqueeze(-1).abs() > 1.0e-8,
                torch.sin(scaled) / distance.unsqueeze(-1).clamp_min(1.0e-8),
                self.frequencies.to(distance),
            )
        else:
            width = self.rcut / max(self.n_radial, 1)
            basis = torch.exp(-((distance.unsqueeze(-1) - self.centers.to(distance)) / width) ** 2)
        return self.network(basis * cutoff.unsqueeze(-1))


class _InteractionBlock(nn.Module):
    def __init__(self, config: Dpa4Config, widths: Sequence[int]) -> None:
        super().__init__()
        self.radial_modes = max(config.radial_modes, 1)
        radial_width = len(widths) * self.radial_modes * config.channels
        self.radial = _RadialNetwork(config, radial_width)
        self.scalar_update = nn.Linear(config.channels, config.channels)
        self.scalar_message = nn.Linear(config.channels, config.channels, bias=False)
        self.degree_message = nn.ModuleList(
            nn.Linear(config.channels, config.channels, bias=False)
            for _ in widths[1:]
        )
        self.degree_update = nn.ModuleList(
            nn.Linear(config.channels, config.channels, bias=False)
            for _ in widths[1:]
        )
        self.gate = nn.Linear(config.channels, config.channels)
        self.scalar_norm = nn.LayerNorm(config.channels)

    def forward(
        self,
        features: list[Tensor],
        edge_vectors: Tensor,
        src: Tensor,
        dst: Tensor,
        edge_mask: Tensor,
        rcut: float,
    ) -> list[Tensor]:
        n_atoms = features[0].shape[0]
        distance = torch.linalg.vector_norm(edge_vectors, dim=-1)
        safe_direction = edge_vectors / distance.clamp_min(1.0e-8).unsqueeze(-1)
        cutoff = _smooth_cutoff(distance, rcut) * edge_mask.to(edge_vectors.dtype)
        angular = _real_angular_basis(safe_direction, len(features) - 1)
        radial = self.radial(distance, cutoff)
        radial = radial.reshape(
            edge_vectors.shape[0],
            len(features),
            self.radial_modes,
            features[0].shape[-1],
        )

        scalar_source = features[0][src, 0]
        scalar_aggregate = edge_vectors.new_zeros((n_atoms, features[0].shape[-1]))
        if scalar_source.numel():
            scalar_edge = self.scalar_message(scalar_source) * radial[:, 0].mean(dim=1)
            scalar_aggregate.index_add_(0, dst, scalar_edge)
        scalar = features[0][:, 0] + self.scalar_update(scalar_aggregate)
        scalar = self.scalar_norm(torch.nn.functional.silu(scalar))
        updated = [scalar.unsqueeze(1)]
        gate = torch.sigmoid(self.gate(scalar))

        for degree, (source_projection, update_projection) in enumerate(
            zip(self.degree_message, self.degree_update),
            start=1,
        ):
            source = features[degree][src]
            radial_channels = radial[:, degree].mean(dim=1)
            message = source_projection(source) * radial_channels[:, None, :]
            direction_message = self._directional_message(
                source_scalar=scalar_source,
                angular=angular[degree],
                radial=radial_channels,
                channels=source.shape[-1],
            )
            aggregate = edge_vectors.new_zeros(features[degree].shape)
            if source.numel():
                aggregate.index_add_(0, dst, message + direction_message)
            value = features[degree] + update_projection(aggregate)
            value = value * gate.unsqueeze(1)
            # Normalize each channel with an invariant norm over the magnetic
            # components.  LayerNorm over the last dimension would normalize
            # each m component independently and break SO(3) equivariance.
            value = value / torch.sqrt(value.square().mean(dim=1, keepdim=True) + 1.0e-8)
            updated.append(value)
        return updated

    @staticmethod
    def _directional_message(
        source_scalar: Tensor,
        angular: Tensor,
        radial: Tensor,
        channels: int,
    ) -> Tensor:
        if radial.shape[-1] == 0:
            return angular.new_zeros((angular.shape[0], angular.shape[1], channels))
        weights = radial[..., :channels]
        if weights.shape[-1] < channels:
            weights = torch.nn.functional.pad(weights, (0, channels - weights.shape[-1]))
        return source_scalar[:, None, :] * angular[:, :, None] * weights[:, None, :]


class Dpa4Model(nn.Module):
    """Standalone DPA4-style equivariant descriptor network."""

    def __init__(self, config: Dpa4Config | Mapping[str, Any]) -> None:
        if not isinstance(config, Dpa4Config):
            config = Dpa4Config.from_mapping(config)
        config.validate()
        super().__init__()
        self.config = config
        self.compute_dtype = _compute_dtype(config.precision)
        widths = _DEGREE_WIDTHS[: config.lmax + 1]
        self.feature_count = config.channels
        self.type_embedding = nn.Embedding(config.ntypes, config.channels)
        self.blocks = nn.ModuleList(
            _InteractionBlock(config, widths) for _ in range(config.n_blocks)
        )
        invariant_width = config.channels * (1 + config.lmax)
        self.readout = nn.Sequential(
            nn.Linear(invariant_width, config.radial_hidden),
            nn.SiLU(),
            nn.Linear(config.radial_hidden, config.channels),
        )
        self.spin_scalar = nn.Linear(1, config.channels, bias=False)
        self.spin_vector = nn.Parameter(torch.zeros(config.channels))
        self.charge_embedding = nn.Embedding(201, config.channels)
        self.multiplicity_embedding = nn.Embedding(100, config.channels)
        self.register_buffer(
            "exclude_mask",
            self._make_exclude_mask(config.ntypes, config.exclude_types),
            persistent=False,
        )
        if config.seed is not None:
            self._reset_with_seed(config.seed)

    @staticmethod
    def _make_exclude_mask(ntypes: int, excluded: Sequence[tuple[int, int]]) -> Tensor:
        mask = torch.ones((ntypes, ntypes), dtype=torch.bool)
        for source, target in excluded:
            mask[source, target] = False
        return mask

    def _reset_with_seed(self, seed: int) -> None:
        state = torch.random.get_rng_state()
        torch.manual_seed(seed)
        for module in self.modules():
            if module is not self and hasattr(module, "reset_parameters"):
                module.reset_parameters()
        torch.random.set_rng_state(state)

    def forward(
        self,
        edge_vectors: Tensor,
        src: Tensor,
        dst: Tensor,
        atype: Tensor,
        n_atoms: int | None = None,
        *,
        spin: Tensor | None = None,
        charge_spin: Tensor | None = None,
        frame_index: Tensor | None = None,
        calibrate: bool = True,
    ) -> Tensor:
        del calibrate
        if n_atoms is None:
            n_atoms = int(atype.shape[0])
        if atype.shape != (n_atoms,):
            raise ValueError("DPA4 atype must have one type index per atom")
        dtype = self.compute_dtype
        edge_vectors = edge_vectors.to(dtype=dtype)
        src = src.to(dtype=torch.long)
        dst = dst.to(dtype=torch.long)
        atype = atype.to(dtype=torch.long)
        if edge_vectors.ndim != 2 or edge_vectors.shape[-1] != 3:
            raise ValueError("DPA4 edge_vectors must have shape (n_edges, 3)")
        if edge_vectors.shape[0] != src.shape[0] or src.shape != dst.shape:
            raise ValueError("DPA4 graph arrays have inconsistent edge counts")
        if atype.numel() and (int(atype.min()) < 0 or int(atype.max()) >= self.config.ntypes):
            raise ValueError("DPA4 atype contains an out-of-range type index")

        scalar = self.type_embedding(atype)
        if self.config.use_spin is not None:
            if spin is None or spin.shape != (n_atoms, 3):
                raise ValueError(f"spin must have shape ({n_atoms}, 3)")
            spin = spin.to(dtype=dtype)
            scalar = scalar + self.spin_scalar(torch.linalg.vector_norm(spin, dim=-1, keepdim=True))
        features = [scalar.unsqueeze(1)]
        for degree in range(1, self.config.lmax + 1):
            features.append(edge_vectors.new_zeros((n_atoms, 2 * degree + 1, self.config.channels)))
        if self.config.use_spin is not None and self.config.lmax >= 1:
            features[1] = features[1] + spin[:, :, None] * self.spin_vector[None, None, :]
        if self.config.add_chg_spin_ebd:
            if charge_spin is None:
                if self.config.default_chg_spin is None:
                    raise ValueError("charge_spin is required by this DPA4 checkpoint")
                charge_spin = edge_vectors.new_tensor(self.config.default_chg_spin).reshape(1, 2)
            charge_spin = charge_spin.to(dtype=dtype).reshape(-1, 2)
            if charge_spin.shape[0] == 1:
                state = charge_spin.expand(n_atoms, -1)
            else:
                if frame_index is None or frame_index.shape != (n_atoms,):
                    raise ValueError("DPA4 frame_index is required for per-structure charge_spin")
                frame_index = frame_index.to(dtype=torch.long)
                if frame_index.numel() and int(frame_index.max()) >= charge_spin.shape[0]:
                    raise ValueError("DPA4 frame_index exceeds charge_spin rows")
                state = charge_spin[frame_index]
            charge = state[:, 0].round().to(dtype=torch.long) + 100
            multiplicity = state[:, 1].round().to(dtype=torch.long)
            if (
                (charge < 0).any()
                or (charge >= 201).any()
                or (multiplicity < 0).any()
                or (multiplicity >= 100).any()
            ):
                raise ValueError("charge_spin is outside the supported embedding range")
            features[0] = features[0] + (
                self.charge_embedding(charge) + self.multiplicity_embedding(multiplicity)
            ).unsqueeze(1)

        if src.numel():
            edge_mask = self.exclude_mask[atype[src], atype[dst]]
        else:
            edge_mask = torch.empty((0,), dtype=torch.bool, device=edge_vectors.device)
        for block in self.blocks:
            features = block(features, edge_vectors, src, dst, edge_mask, self.config.rcut)

        invariants = [features[0][:, 0]]
        invariants.extend((value * value).sum(dim=1) for value in features[1:])
        return self.readout(torch.cat(invariants, dim=-1))

    def export_checkpoint(self, path: str | Path) -> None:
        """Write an MDescriptor-owned checkpoint that this model can reload."""

        torch.save(
            {
                "format": _FORMAT,
                "config": self.config.to_dict(),
                "state_dict": self.state_dict(),
            },
            str(path),
        )


def _torch_load(path: str | Path) -> Any:
    return torch.load(str(path), map_location="cpu", weights_only=True)


def load_native_dpa4_checkpoint(
    path: str | Path,
    *,
    device: str = "cpu",
) -> tuple[Dpa4Model, Dpa4Config]:
    """Load the legacy MDescriptor-owned archive for model-unit tests.

    The public DPA4 calculator no longer calls this loader: its input contract
    is the official DPA4 ``.pt`` archive handled by :mod:`.official`.
    """

    checkpoint = _torch_load(path)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("DPA4 checkpoint must be a mapping with format, config, and state_dict")
    if checkpoint.get("format") != _FORMAT:
        if "model_params" in checkpoint or "_extra_state" in checkpoint:
            raise ValueError(
                "the supplied archive is a DeepMD DPA4 checkpoint; standalone Dpa4Calculator "
                "requires an mdescriptor.dpa4.v1 checkpoint exported by Dpa4Model.export_checkpoint()"
            )
        raise ValueError(f"unsupported DPA4 checkpoint format: {checkpoint.get('format')!r}")
    config = Dpa4Config.from_mapping(checkpoint.get("config", {}))
    state_dict = checkpoint.get("state_dict")
    if not isinstance(state_dict, Mapping):
        raise ValueError("DPA4 checkpoint is missing a state_dict")
    model = Dpa4Model(config)
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise ValueError("DPA4 checkpoint state_dict does not match its config") from exc
    model.to(device=device, dtype=model.compute_dtype)
    model.eval()
    return model, config


def load_dpa4_checkpoint(
    path: str | Path,
    *,
    device: str = "cpu",
) -> tuple[Any, Dpa4Config]:
    """Load an official DPA4 ``.pt`` archive through the local core."""

    from .official import load_official_dpa4_checkpoint

    return load_official_dpa4_checkpoint(path, device=device)


__all__ = [
    "Dpa4Config",
    "Dpa4Model",
    "load_dpa4_checkpoint",
    "load_native_dpa4_checkpoint",
]
