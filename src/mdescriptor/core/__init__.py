"""Stable, implementation-independent descriptor contracts."""

from .descriptor import Descriptor
from .errors import (
    CancelledError,
    ClosedDescriptorError,
    DescriptorConfigError,
    DescriptorError,
    DescriptorInputError,
    ModelLoadError,
    NativeCancelledError,
)
from .input import StructureBatch, StructureInput, batch_from_ase
from .options import ExecutionOptions, OutputOptions
from .result import DescriptorLevel, DescriptorResult

try:  # The native extension is optional until the package is built.
    from .._native import ComputeControl
except ImportError:  # pragma: no cover - exercised only from an unpacked tree
    ComputeControl = object  # type: ignore[misc,assignment]

__all__ = [
    "CancelledError",
    "ClosedDescriptorError",
    "ComputeControl",
    "Descriptor",
    "DescriptorConfigError",
    "DescriptorError",
    "DescriptorInputError",
    "DescriptorLevel",
    "DescriptorResult",
    "ExecutionOptions",
    "ModelLoadError",
    "OutputOptions",
    "StructureBatch",
    "StructureInput",
    "batch_from_ase",
]
