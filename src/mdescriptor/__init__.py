"""Batch-oriented periodic atomic descriptors.

The root package intentionally exposes only the stable contracts and
registry.  Algorithm implementations live under ``mdescriptor.descriptors``
and are imported lazily by the registry.
"""

from .core import (
    CancelledError,
    ClosedDescriptorError,
    ComputeControl,
    Descriptor,
    DescriptorConfigError,
    DescriptorError,
    DescriptorInputError,
    DescriptorLevel,
    DescriptorResult,
    ExecutionOptions,
    ModelLoadError,
    OutputOptions,
    StructureBatch,
    StructureInput,
    batch_from_ase,
)
from .registry import (
    AssetPolicy,
    BUILTIN_REGISTRY,
    DescriptorRegistry,
    DescriptorSpec,
    create_descriptor,
    get_descriptor,
    list_descriptors,
)

__all__ = [
    "AssetPolicy",
    "BUILTIN_REGISTRY",
    "CancelledError",
    "ClosedDescriptorError",
    "ComputeControl",
    "Descriptor",
    "DescriptorConfigError",
    "DescriptorError",
    "DescriptorInputError",
    "DescriptorLevel",
    "DescriptorRegistry",
    "DescriptorResult",
    "DescriptorSpec",
    "ExecutionOptions",
    "ModelLoadError",
    "OutputOptions",
    "StructureBatch",
    "StructureInput",
    "batch_from_ase",
    "create_descriptor",
    "get_descriptor",
    "list_descriptors",
]
