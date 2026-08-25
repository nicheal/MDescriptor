"""Pure NumPy DPA4C descriptor adapter."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ...core.result import DescriptorResult
from ...models import DPA4C_MODEL
from ..model_backed.dpa import (
    compute_batch,
    load_dpa_checkpoint,
    new_runtime,
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
        descriptor._calibrate_output = self.calibrate
        self._closed = False

    @property
    def feature_count(self) -> int:
        return int(self._native.dim_out)

    @property
    def descriptor_dim(self) -> int:
        return self.feature_count

    def compute(self, value: Any, control: Any = None) -> DescriptorResult:
        del control
        if self._closed or self._native is None:
            raise RuntimeError("DPA4C descriptor is closed")
        values = compute_batch(self._native, value)
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
            "backend": "mdescriptor-dpa4c-numpy",
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
