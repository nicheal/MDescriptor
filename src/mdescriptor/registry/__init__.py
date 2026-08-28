"""The explicit descriptor registry and reconstruction factory."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..core.descriptor import Descriptor
from ..core.errors import DescriptorConfigError
from ..core.options import DescriptorConfiguration, ExecutionOptions, OutputOptions
from .builtins import builtin_registry
from .info import (
    DESCRIPTOR_INFO_SCHEMA_VERSION,
    DescriptorInfo,
    parse_descriptor_info,
    validate_descriptor_parameters,
)
from .registry import DescriptorRegistry
from .spec import CAPABILITIES, AssetPolicy, DescriptorSpec


def list_descriptors(*, registry: DescriptorRegistry = builtin_registry) -> tuple[str, ...]:
    return registry.names()


def describe_descriptor(
    name: str,
    *,
    registry: DescriptorRegistry = builtin_registry,
) -> dict[str, Any]:
    """Return static, JSON-safe metadata for one registered descriptor.

    This function deliberately reads only the registry entry.  It does not
    import or instantiate the descriptor class, resolve model assets, or
    initialize a native runtime.
    """

    spec = registry.get(name)
    if spec.info is None:
        raise DescriptorConfigError(
            f"descriptor {spec.name!r} has no static DescriptorInfo",
            code="missing_descriptor_info",
            path=["descriptor", spec.name],
        )
    payload = spec.info.to_dict()
    asset = dict(payload.get("asset", {}))
    asset["policy"] = spec.asset_policy.value
    if "model" in spec.capabilities:
        asset.setdefault("parameter", "model")
    else:
        asset.setdefault("parameter", None)
    asset.setdefault("allow_external", spec.asset_policy is not AssetPolicy.NONE)
    asset.setdefault("bundled_resources", [])
    asset.setdefault("file_extensions", [])

    result: dict[str, Any] = {
        "schema_version": DESCRIPTOR_INFO_SCHEMA_VERSION,
        "name": spec.name,
        "display_name": payload["display_name"],
        "description": payload["description"],
        "category": payload["category"],
        "level": spec.level,
        "backend": spec.backend,
        "capabilities": sorted(spec.capabilities),
        "parameters": payload["parameters"],
        "execution": payload["execution"],
        "input": payload["input"],
        "output": payload["output"],
        "asset": asset,
    }
    return result


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
    spec = registry.get(configuration.descriptor)
    if spec.info is not None:
        validate_descriptor_parameters(
            configuration.descriptor,
            configuration.parameters,
            spec.info.parameters,
        )
    # ``DescriptorConfiguration`` freezes nested JSON values for immutability.
    # Rebuild through its public JSON view so descriptor constructors receive
    # ordinary dict/list values rather than mappingproxy/tuple implementations.
    values = _restore_parameters(configuration.to_dict()["parameters"])
    return spec.load_class()(**values)


def _restore_parameters(parameters: Any) -> dict[str, Any]:
    values = {str(key): value for key, value in parameters.items()}
    output = values.get("output")
    if isinstance(output, Mapping):
        values["output"] = _restore_option(
            output,
            OutputOptions,
            "output",
            {"dtype", "sparse"},
        )
    execution = values.get("execution")
    if isinstance(execution, Mapping):
        values["execution"] = _restore_option(
            execution,
            ExecutionOptions,
            "execution",
            {"device", "num_threads"},
        )
    model = values.get("model")
    if isinstance(model, str):
        from ..models import ModelResource

        # A plain JSON string is the GUI file-picker form: it is always an
        # explicit local path. Named/bundled resources use ModelResource's
        # tagged object form so the two meanings cannot be confused.
        try:
            values["model"] = ModelResource.explicit(model)
        except DescriptorConfigError as exc:
            raise DescriptorConfigError(
                str(exc),
                code=exc.code,
                path=["parameters", "model"],
                details=exc.details,
            ) from exc
    elif isinstance(model, Mapping) and model.get("__type__") == "ModelResource":
        from ..models import ModelResource

        try:
            values["model"] = ModelResource.from_dict(dict(model))
        except DescriptorConfigError as exc:
            raise DescriptorConfigError(
                str(exc),
                code=exc.code,
                path=["parameters", "model"],
                details=exc.details,
            ) from exc
    elif isinstance(model, Mapping):
        raise DescriptorConfigError(
            "serialized model must be a path string or a ModelResource object",
            code="invalid_parameter",
            path=["parameters", "model"],
        )
    return values


def _restore_option(
    value: Mapping[str, Any],
    option_type: type[Any],
    name: str,
    allowed: set[str],
) -> Any:
    unknown = set(value) - allowed
    if unknown:
        field = sorted(str(item) for item in unknown)[0]
        raise DescriptorConfigError(
            f"unsupported {name} option {field!r}",
            code="unknown_option",
            path=["parameters", name, field],
        )
    try:
        return option_type(**dict(value))
    except DescriptorConfigError as exc:
        raise DescriptorConfigError(
            str(exc),
            code=exc.code,
            path=["parameters", name],
            details=exc.details,
        ) from exc
    except (TypeError, ValueError) as exc:
        raise DescriptorConfigError(
            f"invalid {name} options: {exc}",
            code="invalid_option",
            path=["parameters", name],
        ) from exc


__all__ = [
    "AssetPolicy",
    "CAPABILITIES",
    "DESCRIPTOR_INFO_SCHEMA_VERSION",
    "builtin_registry",
    "DescriptorConfiguration",
    "DescriptorInfo",
    "DescriptorRegistry",
    "DescriptorSpec",
    "create_descriptor",
    "describe_descriptor",
    "get_descriptor",
    "list_descriptors",
    "parse_descriptor_info",
    "validate_descriptor_parameters",
]
