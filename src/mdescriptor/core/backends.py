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

import numpy as np

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
        self._dynamic_matrix_width = (
            name in {"CoulombMatrix", "SineMatrix", "EwaldSumMatrix"}
            and self.options.get("n_atoms_max") is None
        )
        self._matrix_width: int | None = None

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
        if self._closed:
            raise RuntimeError("CUDA backend is closed")
        if control is not None and control.cancelled():
            raise CancelledError("descriptor computation was cancelled")
        empty_matrix = self._empty_matrix_result(batch, control)
        if empty_matrix is not None:
            return _cuda_result(empty_matrix, batch, self.name)
        try:
            self._prepare_matrix_width(batch)
            implementation = self._ensure_implementation()
            # Local CPU kernels historically discover their feature dimension
            # on first compute.  Mirror that behavior for a plugin whose
            # constructor cannot be used as the CUDA validation kernel.
            _ = self.feature_count
            raw = self._compute_in_structure_blocks(implementation, batch, control)
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

    def _compute_in_structure_blocks(
        self,
        implementation: Any,
        batch: StructureBatch,
        control: ComputeControl | None,
    ) -> Any:
        """Run CUDA work at structure boundaries and combine raw results.

        A CUDA kernel remains non-interruptible after launch, but long batches
        are submitted as bounded structure blocks.  The proxy control keeps
        the extension from resetting the public total for every block; the
        public control is advanced only after a block has completed.
        """

        if batch.structures == 0:
            if control is not None:
                control.reset(0)
            return implementation.compute(batch, _CudaBlockControl(control))

        # Matrix width is batch-derived when n_atoms_max is omitted.  Freeze
        # it before slicing so every block has the same public feature width.
        if control is not None:
            control.reset(batch.structures)
        block_size = 32
        results: list[Any] = []
        for start in range(0, batch.structures, block_size):
            if control is not None and control.cancelled():
                raise CancelledError("descriptor computation was cancelled")
            stop = min(start + block_size, batch.structures)
            atom_start = int(batch.offsets[start])
            block = _slice_structure_batch(batch, start, stop)
            result = implementation.compute(block, _CudaBlockControl(control))
            if isinstance(result, Mapping) and "pair_records" in result:
                # Pair kernels report global atom indices relative to the
                # block they received.  Restore the full-batch index space
                # before concatenating blocks; _cuda_result then derives
                # structure/local samples from one consistent offset table.
                records = np.asarray(result["pair_records"]).copy()
                if records.ndim != 2 or records.shape[1] < 2:
                    raise TypeError("CUDA backend returned invalid pair records")
                if records.shape[0] > 0:
                    records[:, :2] += atom_start
                result = dict(result)
                result["pair_records"] = records
            results.append(result)
            if control is not None:
                for _ in range(stop - start):
                    if control.cancelled():
                        raise CancelledError("descriptor computation was cancelled")
                    control.mark_completed()
        return _combine_cuda_block_results(results)

    def _prepare_matrix_width(self, batch: StructureBatch) -> None:
        """Resolve dynamic matrix padding before constructing the native backend."""

        if not self._dynamic_matrix_width:
            return
        counts = np.diff(batch.offsets)
        required = int(np.max(counts, initial=0))
        if self._matrix_width == required and self._implementation is not None:
            return
        self._matrix_width = required
        self.options["n_atoms_max"] = required
        # The C++ backend snapshots constructor options.  Recreate it when a
        # later batch resolves a different width, matching CPU per-batch shape.
        if self._implementation is not None:
            close = getattr(self._implementation, "close", None)
            if callable(close):
                close()
            self._implementation = None
            self._feature_count = None

    def _empty_matrix_result(
        self,
        batch: StructureBatch,
        control: ComputeControl | None,
    ) -> Mapping[str, Any] | None:
        """Return the zero-width all-empty matrix result without touching CUDA."""

        if self.name not in {"CoulombMatrix", "SineMatrix", "EwaldSumMatrix"}:
            return None
        counts = np.diff(batch.offsets)
        if counts.size and np.any(counts):
            return None
        if control is not None:
            if control.cancelled():
                raise CancelledError("descriptor computation was cancelled")
            control.reset(batch.structures)
            for _ in range(batch.structures):
                if control.cancelled():
                    raise CancelledError("descriptor computation was cancelled")
                control.mark_completed()
        # An omitted n_atoms_max is resolved per batch by the CPU kernels.
        # Keep that same contract on CUDA: a previous wider batch must not
        # silently pad a later smaller batch (or an all-empty batch).
        n_atoms_max = (
            0
            if self._dynamic_matrix_width
            else int(self.options.get("n_atoms_max") or 0)
        )
        permutation = str(self.options.get("permutation", "sorted_l2"))
        columns = n_atoms_max if permutation == "eigenspectrum" else n_atoms_max * n_atoms_max
        labels = self.options.get("_cuda_labels")
        if labels is None:
            labels = tuple(f"{self.name}:{index}" for index in range(columns))
        execution = self.options.get("execution")
        num_threads = getattr(execution, "num_threads", None)
        if isinstance(execution, Mapping):
            num_threads = execution.get("num_threads")
        return {
            "values": np.zeros((batch.structures, columns), dtype=np.float64),
            "level": "structure",
            "labels": tuple(labels),
            "metadata": {
                "descriptor": self.name,
                "backend": "mdescriptor-cuda",
                "execution": {"device": "cuda", "num_threads": num_threads},
            },
        }

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
        execution = self.options.get("execution")
        num_threads = getattr(execution, "num_threads", None)
        if isinstance(execution, Mapping):
            num_threads = execution.get("num_threads")
        return {
            "descriptor": self.name,
            "backend": "mdescriptor-cuda",
            "execution": {
                "device": "cuda",
                "num_threads": num_threads,
            },
        }


def _looks_cancelled(value: BaseException) -> bool:
    text = str(value).lower()
    return "cancel" in text


class _CudaBlockControl(ComputeControl):
    """Control adapter that prevents a native block from resetting progress."""

    def __init__(self, public: ComputeControl | None) -> None:
        self._public = public

    def reset(self, _total: int) -> None:
        return None

    def cancel(self) -> None:
        if self._public is not None:
            self._public.cancel()

    def cancelled(self) -> bool:
        return self._public is not None and self._public.cancelled()

    def completed(self) -> int:
        return 0

    def total(self) -> int:
        # Progress belongs to the public batch control; a native block must
        # never expose or mutate a second total.
        return 0

    def mark_completed(self) -> None:
        return None


def _slice_structure_batch(batch: StructureBatch, start: int, stop: int) -> StructureBatch:
    atom_start = int(batch.offsets[start])
    atom_stop = int(batch.offsets[stop])
    offsets = batch.offsets[start : stop + 1] - atom_start
    spins = None if batch.spins is None else batch.spins[atom_start:atom_stop]
    charge_spin = (
        None if batch.charge_spin is None else batch.charge_spin[start:stop]
    )
    return StructureBatch(
        batch.numbers[atom_start:atom_stop],
        batch.positions[atom_start:atom_stop],
        batch.cells[start:stop],
        batch.pbc[start:stop],
        offsets,
        batch.ids[start:stop],
        spins,
        charge_spin,
    )


def _combine_cuda_block_results(results: list[Any]) -> Any:
    if not results:
        return {}
    if len(results) == 1:
        return results[0]
    if not all(isinstance(result, Mapping) for result in results):
        raise TypeError("CUDA backend returned inconsistent block results")
    first = dict(results[0])
    values = [np.asarray(result["values"]) for result in results]
    first["values"] = np.concatenate(values, axis=0)

    if any("row_offsets" in result for result in results):
        combined_offsets = [0]
        row_base = 0
        for result, value in zip(results, values, strict=True):
            offsets = np.asarray(result.get("row_offsets", [0]), dtype=np.int64)
            if offsets.ndim != 1 or offsets.size == 0 or offsets[0] != 0:
                raise TypeError("CUDA backend returned invalid block row offsets")
            combined_offsets.extend((offsets[1:] + row_base).tolist())
            row_base += int(value.shape[0])
        first["row_offsets"] = np.asarray(combined_offsets, dtype=np.int64)

    if any("atom_row_offsets" in result for result in results):
        combined_atom_offsets = [0]
        atom_base = 0
        for result in results:
            offsets = np.asarray(result.get("atom_row_offsets", [0]), dtype=np.int64)
            if offsets.ndim != 1 or offsets.size == 0 or offsets[0] != 0:
                raise TypeError("CUDA backend returned invalid block atom offsets")
            combined_atom_offsets.extend((offsets[1:] + atom_base).tolist())
            atom_base += int(offsets[-1])
        first["atom_row_offsets"] = np.asarray(combined_atom_offsets, dtype=np.int64)

    if any("pair_records" in result for result in results):
        first["pair_records"] = np.concatenate(
            [np.asarray(result["pair_records"]) for result in results], axis=0
        )
    # Pair samples are derived from the combined records and full batch by
    # _cuda_result; retaining a block-local samples array would be incorrect.
    first.pop("samples", None)
    return first


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
