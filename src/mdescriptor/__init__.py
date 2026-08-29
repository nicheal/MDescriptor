"""Stable MDescriptor contracts and explicit registry functions."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version

from ._runtime import preload_native, preload_native_binary
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

preload_native_binary()

from .registry import (  # noqa: E402  # preload native binary before registry metadata
    DESCRIPTOR_INFO_SCHEMA_VERSION,
    AssetPolicy,
    DescriptorConfiguration,
    DescriptorRegistry,
    DescriptorSpec,
    builtin_registry,
    create_descriptor,
    describe_descriptor,
    get_descriptor,
    list_descriptors,
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
    "preload_native",
    "__version__",
]
