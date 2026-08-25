"""The explicit descriptor registry and reconstruction factory."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..core.descriptor import Descriptor
from ..core.options import DescriptorConfiguration, ExecutionOptions, OutputOptions
from .builtins import builtin_registry
from .registry import DescriptorRegistry
from .spec import CAPABILITIES, AssetPolicy, DescriptorSpec


def list_descriptors(*, registry: DescriptorRegistry = builtin_registry) -> tuple[str, ...]:
    return registry.names()


def get_descriptor(
    name: str,
    *,
    registry: DescriptorRegistry = builtin_registry,
) -> type:
    return registry.get(name).load_class()


def create_descriptor(
    configuration: DescriptorConfiguration,
    *,
    registry: DescriptorRegistry = builtin_registry,
) -> Descriptor:
    """Rebuild one descriptor from its immutable versioned configuration."""

    if not isinstance(configuration, DescriptorConfiguration):
        raise TypeError("create_descriptor expects a DescriptorConfiguration")
    if configuration.descriptor not in registry:
        raise KeyError(f"unknown descriptor {configuration.descriptor!r}")
    values = _restore_parameters(configuration.parameters)
    return registry.get(configuration.descriptor).load_class()(**values)


def _restore_parameters(parameters: Any) -> dict[str, Any]:
    values = {str(key): value for key, value in parameters.items()}
    output = values.get("output")
    if isinstance(output, Mapping):
        values["output"] = OutputOptions(**dict(output))
    execution = values.get("execution")
    if isinstance(execution, Mapping):
        values["execution"] = ExecutionOptions(**dict(execution))
    model = values.get("model")
    if isinstance(model, Mapping) and model.get("__type__") == "ModelResource":
        from ..models import ModelResource

        values["model"] = ModelResource.from_dict(dict(model))
    return values


__all__ = [
    "AssetPolicy",
    "CAPABILITIES",
    "builtin_registry",
    "DescriptorConfiguration",
    "DescriptorRegistry",
    "DescriptorSpec",
    "create_descriptor",
    "get_descriptor",
    "list_descriptors",
]
