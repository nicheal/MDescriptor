"""Implementation-independent metadata for GUI and registry consumers."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from ..core.errors import DescriptorConfigError

DESCRIPTOR_INFO_SCHEMA_VERSION = 1

_SCHEMA_TYPES = frozenset(
    {
        "integer",
        "number",
        "boolean",
        "string",
        "enum",
        "species",
        "model",
        "array",
        "object",
    }
)
_SCHEMA_FIELDS = frozenset(
    {
        "type",
        "required",
        "default",
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "enum",
        "unit",
        "description",
        "items",
        "properties",
    }
)


@dataclass(frozen=True, slots=True)
class DescriptorInfo:
    """Static, JSON-safe information used to describe one descriptor.

    Identity and implementation fields belong to :class:`DescriptorSpec`.
    This type therefore contains only the descriptive payload that can be
    safely shown by a frontend without importing a descriptor implementation.
    Nested mappings are frozen at construction so registry entries remain
    immutable after registration.
    """

    display_name: str
    description: str
    category: str
    parameters: Mapping[str, Any] = field(default_factory=dict)
    execution: Mapping[str, Any] = field(default_factory=dict)
    input: Mapping[str, Any] = field(default_factory=dict)
    output: Mapping[str, Any] = field(default_factory=dict)
    asset: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in ("display_name", "description", "category"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise DescriptorConfigError(
                    f"descriptor info {field_name} must be a non-empty string",
                    code="invalid_descriptor_info",
                    path=[field_name],
                )

        for field_name in ("parameters", "execution", "input", "output", "asset"):
            value = getattr(self, field_name)
            if not isinstance(value, Mapping):
                raise DescriptorConfigError(
                    f"descriptor info {field_name} must be a JSON object",
                    code="invalid_descriptor_info",
                    path=[field_name],
                )

        try:
            frozen_parameters = _freeze_json(self.parameters)
            frozen_execution = _freeze_json(self.execution)
            frozen_input = _freeze_json(self.input)
            frozen_output = _freeze_json(self.output)
            frozen_asset = _freeze_json(self.asset)
        except (TypeError, ValueError) as exc:
            raise DescriptorConfigError(
                f"descriptor info is not JSON-safe: {exc}",
                code="invalid_descriptor_info",
            ) from exc

        if not isinstance(frozen_parameters, MappingProxyType):  # pragma: no cover
            raise DescriptorConfigError("descriptor info parameters must be a JSON object")
        for name, schema in frozen_parameters.items():
            if not isinstance(schema, Mapping):
                raise DescriptorConfigError(
                    f"descriptor parameter {name!r} must be a JSON object",
                    code="invalid_descriptor_info",
                    path=["parameters", name],
                )
            _validate_schema(schema, ["parameters", name])

        object.__setattr__(self, "display_name", self.display_name.strip())
        object.__setattr__(self, "description", self.description.strip())
        object.__setattr__(self, "category", self.category.strip())
        object.__setattr__(self, "parameters", frozen_parameters)
        object.__setattr__(self, "execution", frozen_execution)
        object.__setattr__(self, "input", frozen_input)
        object.__setattr__(self, "output", frozen_output)
        object.__setattr__(self, "asset", frozen_asset)

    def to_dict(self) -> dict[str, Any]:
        """Return a mutable JSON-safe copy of the descriptive payload."""

        return {
            "display_name": self.display_name,
            "description": self.description,
            "category": self.category,
            "parameters": _thaw_json(self.parameters),
            "execution": _thaw_json(self.execution),
            "input": _thaw_json(self.input),
            "output": _thaw_json(self.output),
            "asset": _thaw_json(self.asset),
        }


def _validate_schema(value: Mapping[str, Any], path: list[str]) -> None:
    unknown = set(value) - _SCHEMA_FIELDS
    if unknown:
        names = ", ".join(sorted(str(item) for item in unknown))
        raise DescriptorConfigError(
            f"descriptor schema has unsupported field(s): {names}",
            code="invalid_descriptor_info",
            path=path,
        )

    schema_type = value.get("type")
    if schema_type is not None and schema_type not in _SCHEMA_TYPES:
        raise DescriptorConfigError(
            f"descriptor schema type {schema_type!r} is not supported",
            code="invalid_descriptor_info",
            path=[*path, "type"],
        )
    if "required" in value and not isinstance(value["required"], bool):
        raise DescriptorConfigError(
            "descriptor schema required must be a boolean",
            code="invalid_descriptor_info",
            path=[*path, "required"],
        )
    for field_name in ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"):
        constraint = value.get(field_name)
        if constraint is not None and (
            isinstance(constraint, bool)
            or not isinstance(constraint, (int, float))
            or (isinstance(constraint, float) and not math.isfinite(constraint))
        ):
            raise DescriptorConfigError(
                f"descriptor schema {field_name} must be a finite number",
                code="invalid_descriptor_info",
                path=[*path, field_name],
            )
    for field_name in ("unit", "description"):
        field_value = value.get(field_name)
        if field_value is not None and not isinstance(field_value, str):
            raise DescriptorConfigError(
                f"descriptor schema {field_name} must be a string",
                code="invalid_descriptor_info",
                path=[*path, field_name],
            )

    enum_values = value.get("enum")
    if enum_values is not None and not isinstance(enum_values, (list, tuple)):
        raise DescriptorConfigError(
            "descriptor schema enum must be an array",
            code="invalid_descriptor_info",
            path=[*path, "enum"],
        )

    items = value.get("items")
    if items is not None:
        if not isinstance(items, Mapping):
            raise DescriptorConfigError(
                "descriptor schema items must be an object",
                code="invalid_descriptor_info",
                path=[*path, "items"],
            )
        _validate_schema(items, [*path, "items"])

    properties = value.get("properties")
    if properties is not None:
        if not isinstance(properties, Mapping):
            raise DescriptorConfigError(
                "descriptor schema properties must be an object",
                code="invalid_descriptor_info",
                path=[*path, "properties"],
            )
        for name, schema in properties.items():
            if not isinstance(schema, Mapping):
                raise DescriptorConfigError(
                    f"descriptor schema property {name!r} must be an object",
                    code="invalid_descriptor_info",
                    path=[*path, "properties", str(name)],
                )
            _validate_schema(schema, [*path, "properties", str(name)])


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("JSON object keys must be strings")
        return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    if isinstance(value, float):
        raise TypeError("JSON values cannot contain non-finite numbers")
    raise TypeError(f"unsupported value type {type(value).__name__}")


def _thaw_json(value: Any) -> Any:
    if isinstance(value, MappingProxyType):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


__all__ = ["DESCRIPTOR_INFO_SCHEMA_VERSION", "DescriptorInfo"]
