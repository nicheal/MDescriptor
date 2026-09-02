"""Private execution backends used by :mod:`mdescriptor.core.adapter`.

The public descriptor contract deliberately knows nothing about native ABI
objects.  In particular, the CPU extension and the optional CUDA extension do
not share control objects.  This module is the small private seam where the
adapter turns the public ``StructureBatch``/``ComputeControl`` objects into
backend calls.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

from .control import ComputeControl, _unwrap_native_control
from .errors import CancelledError, MDescriptorError
from .input import StructureBatch
from .result import DescriptorResult, pair_samples


@runtime_checkable
class BackendKernel(Protocol):
    """The implementation-independent kernel protocol behind an adapter."""

    @property
    def feature_count(self) -> int | None: ...

    def compute(
        self,
        batch: StructureBatch,
        control: ComputeControl | None = None,
    ) -> Any: ...

    def close(self) -> None: ...

    def metadata(self) -> Mapping[str, Any]: ...


class CpuBackend:
    """Adapt the existing CPU kernel without changing its private ABI."""

    def __init__(self, kernel: Any) -> None:
        self.kernel = kernel

    @property
    def feature_count(self) -> int | None:
        value = getattr(self.kernel, "feature_count", None)
        if value is None:
            return None
        return int(value)

    def compute(
        self,
        batch: StructureBatch,
        control: ComputeControl | None = None,
    ) -> Any:
        # Only this class unwraps the CPU extension's pybind control.  The
        # CUDA backend receives the public wrapper and therefore cannot depend
        # on the CPU extension's C++ ABI.
        return self.kernel.compute(batch, _unwrap_native_control(control))

    def close(self) -> None:
        close = getattr(self.kernel, "close", None)
        if callable(close):
            close()

    def metadata(self) -> Mapping[str, Any]:
        builder = getattr(self.kernel, "_metadata", None)
        if callable(builder):
            value = builder()
            if isinstance(value, Mapping):
                return value
        return {}


class CudaBackend:
    """Lazy proxy for the optional CUDA plugin.

    Construction only stores the normalized configuration.  Importing the
    plugin, checking the driver, selecting a device, and allocating a CUDA
    context all happen in ``compute`` as required by the public contract.
    """

    def __init__(
        self,
        name: str,
        options: Mapping[str, Any],
        feature_count: int | None = None,
    ) -> None:
        self.name = name
        self.options = dict(options)
        self._feature_count = None if feature_count is None else int(feature_count)
        self._implementation: Any = None
        self._closed = False

    @property
    def feature_count(self) -> int | None:
        if self._feature_count is not None:
            return self._feature_count
        implementation = self._implementation
        if implementation is None:
            return None
        value = getattr(implementation, "feature_count", None)
        if value is None:
            return None
        self._feature_count = int(value)
        return self._feature_count

    def _ensure_implementation(self) -> Any:
        if self._closed:
            raise RuntimeError("CUDA backend is closed")
        if self._implementation is None:
            from .._runtime import create_cuda_backend

            self._implementation = create_cuda_backend(self.name, self.options)
        return self._implementation

    def compute(
        self,
        batch: StructureBatch,
        control: ComputeControl | None = None,
    ) -> DescriptorResult:
        if control is not None and control.cancelled():
            raise CancelledError("descriptor computation was cancelled")
        try:
            implementation = self._ensure_implementation()
            # Local CPU kernels historically discover their feature dimension
            # on first compute.  Mirror that behavior for a plugin whose
            # constructor cannot be used as the CUDA validation kernel.
            _ = self.feature_count
            raw = implementation.compute(batch, control)
            value = getattr(implementation, "feature_count", None)
            if value is not None and int(value) > 0:
                self._feature_count = int(value)
        except CancelledError:
            raise
        except MDescriptorError:
            raise
        except MemoryError as exc:
            raise MDescriptorError(
                "CUDA backend ran out of memory",
                code="backend_out_of_memory",
                path=["execution", "device"],
            ) from exc
        except ImportError as exc:
            raise MDescriptorError(
                "CUDA backend is unavailable",
                code="device_unavailable",
                path=["execution", "device"],
            ) from exc
        except OSError as exc:
            raise MDescriptorError(
                "CUDA backend is unavailable",
                code="device_unavailable",
                path=["execution", "device"],
            ) from exc
        except ValueError as exc:
            raise MDescriptorError(
                "CUDA backend failed",
                code="backend_error",
                path=["execution", "device"],
                details={"exception": type(exc).__name__},
            ) from exc
        except RuntimeError as exc:
            if _looks_cancelled(exc):
                raise CancelledError("descriptor computation was cancelled") from exc
            raise MDescriptorError(
                "CUDA backend failed",
                code="backend_error",
                path=["execution", "device"],
                details={"exception": type(exc).__name__},
            ) from exc
        except AttributeError as exc:
            raise MDescriptorError(
                "CUDA backend failed",
                code="backend_error",
                path=["execution", "device"],
                details={"exception": type(exc).__name__},
            ) from exc
        if control is not None and control.cancelled():
            raise CancelledError("descriptor computation was cancelled")
        try:
            return _cuda_result(raw, batch, self.name)
        except MDescriptorError:
            raise
        except (TypeError, ValueError, KeyError, IndexError) as exc:
            raise MDescriptorError(
                "CUDA backend returned an invalid result",
                code="backend_error",
                details={"exception": type(exc).__name__},
            ) from exc

    def close(self) -> None:
        self._closed = True
        implementation = self._implementation
        if implementation is not None:
            close = getattr(implementation, "close", None)
            if callable(close):
                close()

    def metadata(self) -> Mapping[str, Any]:
        implementation = self._implementation
        if implementation is not None:
            builder = getattr(implementation, "metadata", None)
            if callable(builder):
                value = builder()
                if isinstance(value, Mapping):
                    return value
        return {
            "descriptor": self.name,
            "backend": "mdescriptor-cuda",
            "execution": {
                "device": "cuda",
                "num_threads": self.options.get("execution", {}).get("num_threads"),
            },
        }


def _looks_cancelled(value: BaseException) -> bool:
    text = str(value).lower()
    return "cancel" in text


def _cuda_result(raw: Any, batch: StructureBatch, name: str) -> DescriptorResult:
    """Normalize the small raw-result vocabulary exposed by CUDA plugins."""

    if isinstance(raw, DescriptorResult):
        return raw
    if not isinstance(raw, Mapping):
        raise MDescriptorError(
            "CUDA backend returned an invalid result",
            code="backend_error",
            details={"result_type": type(raw).__name__},
        )

    try:
        values = raw["values"]
        level = str(raw["level"])
    except KeyError as exc:
        raise MDescriptorError(
            "CUDA backend returned an incomplete result",
            code="backend_error",
        ) from exc
    row_offsets = raw.get("row_offsets")
    samples = raw.get("samples")
    if samples is None and "pair_records" in raw:
        if row_offsets is None:
            raise MDescriptorError(
                "CUDA pair result is missing row offsets",
                code="backend_error",
            )
        samples = pair_samples(raw["pair_records"], row_offsets, batch.offsets)
    if level == "atom" and row_offsets is None:
        row_offsets = batch.offsets.copy()
    labels = raw.get("labels")
    if labels is None:
        width = int(values.shape[1])
        labels = tuple(f"{name}:{index}" for index in range(width))
    atom_row_offsets = raw.get("atom_row_offsets")
    if level == "pair" and atom_row_offsets is None:
        atom_row_offsets = batch.offsets.copy()
    metadata = raw.get("metadata", {})
    return DescriptorResult(
        values,
        level,
        batch.ids,
        row_offsets,
        tuple(labels),
        metadata,
        samples=samples,
        _atom_row_offsets=atom_row_offsets,
    )


__all__ = ["BackendKernel", "CpuBackend", "CudaBackend"]
