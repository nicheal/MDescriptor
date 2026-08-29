"""Stable, implementation-independent descriptor contracts."""

from .control import ComputeControl
from .descriptor import Descriptor
from .errors import (
    CancelledError,
    ClosedDescriptorError,
    DescriptorConfigError,
    DescriptorInputError,
    MDescriptorError,
    ModelLoadError,
    UnsupportedPeriodicityError,
)
from .input import StructureBatch, StructureInput, batch_from_ase, coerce_batch
from .options import (
    CONFIGURATION_SCHEMA_VERSION,
    DescriptorConfiguration,
    ExecutionOptions,
    JSONValue,
    OutputOptions,
)
from .result import RESULT_SCHEMA_VERSION, DescriptorLevel, DescriptorResult
from .species import (
    normalize_species,
    require_species,
    species_from_batch,
    validate_batch_species,
)

__all__ = [
    "CancelledError",
    "ClosedDescriptorError",
    "CONFIGURATION_SCHEMA_VERSION",
    "ComputeControl",
    "Descriptor",
    "DescriptorConfiguration",
    "DescriptorConfigError",
    "DescriptorInputError",
    "DescriptorLevel",
    "DescriptorResult",
    "RESULT_SCHEMA_VERSION",
    "ExecutionOptions",
    "JSONValue",
    "MDescriptorError",
    "ModelLoadError",
    "OutputOptions",
    "StructureBatch",
    "StructureInput",
    "UnsupportedPeriodicityError",
    "batch_from_ase",
    "coerce_batch",
    "normalize_species",
    "require_species",
    "species_from_batch",
    "validate_batch_species",
]
