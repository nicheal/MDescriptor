"""Stable MDescriptor contracts and explicit registry functions."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version

from .core import (
    CONFIGURATION_SCHEMA_VERSION,
    RESULT_SCHEMA_VERSION,
    CancelledError,
    ClosedDescriptorError,
    ComputeControl,
    Descriptor,
    DescriptorConfigError,
    DescriptorInputError,
    DescriptorLevel,
    DescriptorResult,
    ExecutionOptions,
    MDescriptorError,
    ModelLoadError,
    OutputOptions,
    StructureBatch,
    StructureInput,
)
from .registry import (
    DESCRIPTOR_INFO_SCHEMA_VERSION,
    AssetPolicy,
    DescriptorConfiguration,
    DescriptorInfo,
    DescriptorRegistry,
    DescriptorSpec,
    builtin_registry,
    create_descriptor,
    describe_descriptor,
    get_descriptor,
    list_descriptors,
    parse_descriptor_info,
)

API_VERSION = 1

try:
    from ._version import __version__
except ImportError:  # pragma: no cover - source tree without generated version
    try:
        __version__ = package_version("MDescriptor")
    except PackageNotFoundError:  # pragma: no cover - uninstalled source tree
        __version__ = "0+unknown"


def get_runtime_info() -> dict[str, str | int]:
    """Return package and public schema versions for backend run metadata."""

    return {
        "version": __version__,
        "api_version": API_VERSION,
        "configuration_schema_version": CONFIGURATION_SCHEMA_VERSION,
        "descriptor_info_schema_version": DESCRIPTOR_INFO_SCHEMA_VERSION,
        "result_schema_version": RESULT_SCHEMA_VERSION,
    }

__all__ = [
    "API_VERSION",
    "AssetPolicy",
    "CancelledError",
    "ClosedDescriptorError",
    "ComputeControl",
    "Descriptor",
    "DescriptorConfigError",
    "DescriptorConfiguration",
    "DescriptorInputError",
    "DescriptorInfo",
    "DescriptorLevel",
    "DescriptorRegistry",
    "DescriptorResult",
    "DescriptorSpec",
    "DESCRIPTOR_INFO_SCHEMA_VERSION",
    "CONFIGURATION_SCHEMA_VERSION",
    "ExecutionOptions",
    "MDescriptorError",
    "ModelLoadError",
    "OutputOptions",
    "RESULT_SCHEMA_VERSION",
    "StructureBatch",
    "StructureInput",
    "builtin_registry",
    "create_descriptor",
    "describe_descriptor",
    "get_descriptor",
    "get_runtime_info",
    "list_descriptors",
    "parse_descriptor_info",
    "__version__",
]
