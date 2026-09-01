"""DPA4C descriptor adapter with a native C++ core and NumPy fallback."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from ...core.result import DescriptorResult
from ...models import DPA4C_MODEL
from ..model_backed.dpa import (
    compute_batch,
    load_dpa_checkpoint,
    new_runtime,
)
from ..model_backed.graph import _ATOMIC_SYMBOLS
from .dpa_common import compute_native_batch


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


def _type_numbers(type_map: Any) -> np.ndarray:
    """Translate the checkpoint type order to public atomic numbers."""

    numbers_by_symbol = {symbol: number for number, symbol in _ATOMIC_SYMBOLS.items()}
    return np.asarray(
        [int(numbers_by_symbol.get(str(symbol), -1)) for symbol in type_map],
        dtype=np.int32,
    )


class Dpa4cKernel:
    """Compute DPA4C through the vendored Torch-free CPU inference core."""

    name = "DPA4C"
    accepts_preloaded_checkpoint = True
    configuration_defaults = {"calibrate": True}

    def __init__(
        self,
        model_path: str | Path | None = None,
        calibrate: bool = True,
        num_threads: int | None = None,
        _checkpoint: Mapping[str, Any] | None = None,
    ) -> None:
        path = DPA4C_MODEL if model_path is None else Path(model_path)
        if not str(path):
            raise ValueError("DPA4C model path cannot be empty")
        self.model_path = str(path.expanduser())
        if _checkpoint is None:
            _info, checkpoint = load_dpa_checkpoint(
                Path(self.model_path),
                expected_descriptor="DPA4C",
            )
        else:
            checkpoint = _checkpoint
        self._native = new_runtime(Path(self.model_path), checkpoint)
        descriptor = self._native.descriptor
        if descriptor.__class__.__name__ != "DescrptDPA4C":
            raise ValueError("checkpoint did not construct a DPA4C descriptor")
        self.calibrate = bool(calibrate)
        if num_threads is None:
            num_threads = 1
        if isinstance(num_threads, bool) or not isinstance(num_threads, int) or num_threads <= 0:
            raise ValueError("num_threads must be a positive integer")
        self.num_threads = int(num_threads)
        descriptor._calibrate_output = self.calibrate
        self._cpp = None
        payload = _native_payload(
            descriptor,
            calibrate=self.calibrate,
            num_threads=self.num_threads,
        )
        self._cuda_model_payload = payload
        if payload is not None:
            from mdescriptor import _native as native

            self._cpp = native.Dpa4cCalculator(payload)
        self._closed = False

    def _cuda_payload(self) -> dict[str, Any]:
        """Return the private device handoff without changing public state."""

        return {
            "version": 1,
            "model": self._cuda_model_payload,
            "labels": self._labels(),
            "type_numbers": _type_numbers(self._native.type_map),
            "feature_count": self.feature_count,
        }

    @property
    def feature_count(self) -> int:
        return int(self._native.dim_out)

    @property
    def descriptor_dim(self) -> int:
        return self.feature_count

    def compute(self, value: Any, control: Any = None) -> DescriptorResult:
        if self._closed or self._native is None:
            raise RuntimeError("DPA4C descriptor is closed")
        if self._cpp is None:
            values = compute_batch(self._native, value, control=control)
        else:
            values = compute_native_batch(self._cpp, self._native, value, control)
        return DescriptorResult(
            values,
            "atom",
            value.ids,
            value.offsets.copy(),
            self._labels(),
            self._metadata(),
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._cpp is not None:
            self._cpp.close()
            self._cpp = None
        self._native = None

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
        return {
            "backend": (
                "mdescriptor-dpa4c-cpp"
                if self._cpp is not None
                else "mdescriptor-dpa4c-numpy"
            ),
            "descriptor": self.name,
            "type_map": tuple(self._native.type_map),
            "rcut": float(self._native.rcut),
            "channels": int(descriptor.channels),
            "lmax": int(descriptor.lmax),
            "basis_type": str(descriptor.basis_type),
            "n_radial": int(descriptor.n_radial),
            "radial_modes": int(descriptor.radial_modes),
            "precision": str(descriptor.precision),
            "calibrated": self.calibrate,
            "use_spin": getattr(descriptor, "use_spin", None),
            "add_chg_spin_ebd": bool(getattr(descriptor, "add_chg_spin_ebd", False)),
            "device": "cpu",
            "feature_count": self.feature_count,
        }


__all__ = ["Dpa4cKernel"]
