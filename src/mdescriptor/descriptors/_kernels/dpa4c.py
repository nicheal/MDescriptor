"""DPA4C descriptor adapter with a native C++ core and NumPy fallback."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from ...models import DPA4C_MODEL
from .dpa_common import DpaKernelBase


def _as_float32(value: Any) -> np.ndarray:
    """Materialize one validated checkpoint tensor for the native backend."""

    return np.ascontiguousarray(np.asarray(value, dtype=np.float32))


def _native_payload(descriptor: Any, *, calibrate: bool, num_threads: int) -> dict[str, Any] | None:
    """Extract the DPA4C tensor ABI understood by the C++ core.

    The public loader still owns checkpoint validation and deserialization.  A
    native calculator is selected only for the graph-native, spin-free,
    uncompressed configuration implemented by the current C++ core.  More
    specialised DPA4C variants continue through the reference NumPy path so
    the public API remains available for them.
    """

    if getattr(descriptor, "compress", False):
        return None
    if getattr(descriptor, "use_spin", None) is not None:
        return None
    if getattr(descriptor, "charge_spin_embedding", None) is not None:
        return None
    if getattr(descriptor, "exclude_types", None):
        return None

    try:
        from mdescriptor import _native as native

        if not hasattr(native, "Dpa4cCalculator"):
            return None
    except ImportError:
        return None

    radial_embedding = descriptor.radial_embedding
    pair_network = descriptor.pair_film.network
    if len(radial_embedding.layers) != 2 or len(pair_network.layers) != 2:
        return None

    alignment = descriptor.readout.channel_alignment
    if len(alignment) != 2:
        return None
    alignment_parts = [_as_float32(layer.w).reshape(-1) for layer in alignment]
    alignment_offsets = [0]
    for part in alignment_parts:
        alignment_offsets.append(alignment_offsets[-1] + int(part.size))

    projection_parts: list[np.ndarray] = []
    projection_offsets = [0]
    for projection in descriptor.readout.probe_projections:
        if projection is not None:
            projection_parts.append(_as_float32(projection.w).reshape(-1))
            projection_offsets.append(projection_offsets[-1] + int(projection_parts[-1].size))
        else:
            projection_offsets.append(projection_offsets[-1])

    return {
        "rcut": float(descriptor.rcut),
        "ntypes": int(descriptor.ntypes),
        "channels": int(descriptor.channels),
        "lmax": int(descriptor.lmax),
        "n_radial": int(descriptor.n_radial),
        "radial_modes": int(descriptor.radial_modes),
        "radial_hidden": int(radial_embedding.layers[0].w.shape[1] // 2),
        "pair_hidden": int(pair_network.layers[0].w.shape[1] // 2),
        "num_threads": int(num_threads),
        "calibrate": bool(calibrate),
        "type_embedding": _as_float32(descriptor.type_embedding.call()),
        "radial_freqs": _as_float32(descriptor.radial_basis.adam_freqs).reshape(-1),
        "radial_w0": _as_float32(radial_embedding.layers[0].w),
        "radial_w1": _as_float32(radial_embedding.layers[1].w),
        "radial_mode_w": _as_float32(
            descriptor.radial_mode_head.w
            if descriptor.radial_mode_head is not None
            else np.empty((0,), dtype=np.float32)
        ),
        "pair_w0": _as_float32(pair_network.layers[0].w),
        "pair_w1": _as_float32(pair_network.layers[1].w),
        "degree_channels": [int(width) for width in descriptor.degree_channels],
        "bispectrum_ranks": [int(rank) for rank in descriptor.readout.bispectrum_ranks],
        "readout_alignment": (
            np.concatenate(alignment_parts)
            if alignment_parts
            else np.empty((0,), dtype=np.float32)
        ),
        "readout_alignment_offsets": np.asarray(alignment_offsets, dtype=np.int64),
        "readout_projections": (
            np.concatenate(projection_parts)
            if projection_parts
            else np.empty((0,), dtype=np.float32)
        ),
        "readout_projection_offsets": np.asarray(projection_offsets, dtype=np.int64),
        "bispectrum_coupling": _as_float32(descriptor.readout.bispectrum_coupling),
        "coupling_offsets": np.asarray(descriptor.readout.coupling_offsets, dtype=np.int64),
        "degree_triples": [
            int(value)
            for triple in descriptor.readout.degree_triples
            for value in triple
        ],
        "probe_offsets": np.asarray(descriptor.readout.probe_offsets, dtype=np.int64),
        "probe_index": np.asarray(descriptor.readout.probe_index, dtype=np.int64),
        "probe_scale": _as_float32(descriptor.readout.probe_scale),
        "output_mean": _as_float32(descriptor.mean),
        "output_stddev": _as_float32(descriptor.stddev),
    }


class Dpa4cKernel(DpaKernelBase):
    """Compute DPA4C through the vendored Torch-free CPU inference core."""

    name = "DPA4C"
    checkpoint_descriptor = "DPA4C"
    runtime_descriptor_name = "DescrptDPA4C"
    native_calculator_name = "Dpa4cCalculator"
    default_model = DPA4C_MODEL
    configuration_defaults = {"calibrate": True}

    def __init__(
        self,
        model_path: str | Path | None = None,
        calibrate: bool = True,
        num_threads: int | None = None,
        _checkpoint: Mapping[str, Any] | None = None,
    ) -> None:
        self.calibrate = bool(calibrate)
        self._initialize_dpa(model_path, num_threads, _checkpoint)

    def _native_payload(
        self,
        descriptor: Any,
        *,
        num_threads: int,
    ) -> dict[str, Any] | None:
        descriptor._calibrate_output = self.calibrate
        return _native_payload(
            descriptor,
            calibrate=self.calibrate,
            num_threads=num_threads,
        )

    def _labels(self) -> tuple[str, ...]:
        descriptor = self._native.descriptor
        readout = descriptor.readout
        labels = [
            f"dpa4c:scalar,channel={index}"
            for index in range(int(descriptor.channels))
        ]
        for degree, width in enumerate(readout.degree_channels[1:], start=1):
            for row in range(int(width)):
                for column in range(row, int(width)):
                    labels.append(
                        f"dpa4c:gram,degree={degree},row={row},column={column}"
                    )
        labels.extend(
            f"dpa4c:bispectrum,index={index}"
            for index in range(int(readout.probe_index.size))
        )
        labels.extend(
            f"dpa4c:quartic,index={index}"
            for index in range(int(readout.quartic_dim))
        )
        spin = descriptor.spin
        if spin is not None:
            labels.extend(
                f"dpa4c:spin,vector_gram,index={index}"
                for index in range(int(spin.vector_gram_index.size))
            )
            labels.extend(
                f"dpa4c:spin,quadrupole_gram,index={index}"
                for index in range(int(spin.quadrupole_gram_index.size))
            )
            labels.extend(
                f"dpa4c:spin,cross_degree2,index={index}"
                for index in range(2 * int(spin.spin_channels))
            )
            labels.extend(
                f"dpa4c:spin,magnitude,index={index}"
                for index in range(int(spin.spin_channels))
            )
            labels.extend(
                f"dpa4c:spin,coordination,index={index}"
                for index in range(int(spin.spin_channels))
            )
        labels.extend(["dpa4c:divisor=scalar", "dpa4c:divisor=angular"])
        labels.extend(
            f"dpa4c:type_embedding,index={index}"
            for index in range(int(descriptor.channels))
        )
        return tuple(labels)

    def _metadata(self) -> dict[str, Any]:
        descriptor = self._native.descriptor
        metadata = self._metadata_base(descriptor)
        metadata.update(
            {
                "radial_modes": int(descriptor.radial_modes),
                "calibrated": self.calibrate,
                "device": "cpu",
                "feature_count": self.feature_count,
            }
        )
        return metadata


__all__ = ["Dpa4cKernel"]
