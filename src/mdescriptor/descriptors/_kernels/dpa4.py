"""Pure NumPy DPA4 descriptor adapter."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ...core.result import DescriptorResult
from ...models import DPA4_MODEL
from ..model_backed.dpa import (
    compute_batch,
    load_dpa_checkpoint,
    new_runtime,
)


class Dpa4Kernel:
    """Compute DPA4 through the vendored Torch-free CPU inference core."""

    name = "DPA4"
    accepts_preloaded_checkpoint = True

    def __init__(
        self,
        model_path: str | Path | None = None,
        _checkpoint: Mapping[str, Any] | None = None,
    ) -> None:
        path = DPA4_MODEL if model_path is None else Path(model_path)
        if not str(path):
            raise ValueError("DPA4 model path cannot be empty")
        self.model_path = str(path.expanduser())
        if _checkpoint is None:
            _info, checkpoint = load_dpa_checkpoint(
                Path(self.model_path),
                expected_descriptor="DPA4",
            )
        else:
            checkpoint = _checkpoint
        self._native = new_runtime(Path(self.model_path), checkpoint)
        descriptor = self._native.descriptor
        if descriptor.__class__.__name__ != "DescrptDPA4":
            raise ValueError("checkpoint did not construct a DPA4 descriptor")
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
            raise RuntimeError("DPA4 descriptor is closed")
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
        return tuple(
            f"dpa4:scalar,channel={index}"
            for index in range(int(self._native.descriptor.channels))
        )

    def _metadata(self) -> dict[str, Any]:
        descriptor = self._native.descriptor
        return {
            "backend": "mdescriptor-dpa4-numpy",
            "descriptor": self.name,
            "type_map": tuple(self._native.type_map),
            "rcut": float(self._native.rcut),
            "channels": int(descriptor.channels),
            "lmax": int(descriptor.lmax),
            "basis_type": str(getattr(descriptor, "basis_type", "unknown")),
            "n_radial": int(getattr(descriptor, "n_radial", 0)),
            "precision": str(getattr(descriptor, "precision", "float64")),
            "use_spin": getattr(descriptor, "use_spin", None),
            "add_chg_spin_ebd": bool(getattr(descriptor, "add_chg_spin_ebd", False)),
            "device": "cpu",
            "feature_count": self.feature_count,
        }


__all__ = ["Dpa4Kernel"]
