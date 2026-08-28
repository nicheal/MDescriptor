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

# These names are accepted only by the direct Python constructor for
# backwards compatibility.  They are deliberately not part of DescriptorInfo
# and therefore can never become GUI fields or persisted canonical keys.
LEGACY_PARAMETER_ALIASES: dict[str, dict[str, str]] = {
    "ACSF": {
        "G2": "g2_params",
        "g2": "g2_params",
        "G3": "g3_params",
        "g3": "g3_params",
        "G4": "g4_params",
        "g4": "g4_params",
        "G5": "g5_params",
        "g5": "g5_params",
    },
    "SOAPTurbo": {"compress_mode": "compression"},
    "C00PSMLFF": {"cutoff": "r_cut", "n_max": "n_radial"},
    "MTP": {
        "cutoff": "max_dist",
        "l_max": "max_rank",
        "max_level": "max_rank",
        "level": "max_rank",
    },
}


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

    if "type" not in value:
        raise DescriptorConfigError(
            "descriptor schema must declare type",
            code="invalid_descriptor_info",
            path=[*path, "type"],
        )
    schema_type = value["type"]
    if not isinstance(schema_type, str) or schema_type not in _SCHEMA_TYPES:
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
    if schema_type == "enum" and not enum_values:
        raise DescriptorConfigError(
            "enum descriptor schemas must contain at least one value",
            code="invalid_descriptor_info",
            path=[*path, "enum"],
        )
    if enum_values is not None:
        try:
            _freeze_json(enum_values)
        except (TypeError, ValueError) as exc:
            raise DescriptorConfigError(
                f"descriptor schema enum is not JSON-safe: {exc}",
                code="invalid_descriptor_info",
                path=[*path, "enum"],
            ) from exc
        if schema_type != "enum":
            type_only_schema = {key: item for key, item in value.items() if key != "enum"}
            for index, enum_value in enumerate(enum_values):
                try:
                    _validate_parameter_value(
                        enum_value,
                        type_only_schema,
                        [*path, "enum", str(index)],
                    )
                except DescriptorConfigError as exc:
                    raise DescriptorConfigError(
                        f"descriptor schema enum value has the wrong type: {exc}",
                        code="invalid_descriptor_info",
                        path=exc.path,
                    ) from exc

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

    if "default" in value:
        try:
            _validate_parameter_value(value["default"], value, [*path, "default"])
        except DescriptorConfigError as exc:
            raise DescriptorConfigError(
                f"descriptor schema default is invalid: {exc}",
                code="invalid_descriptor_info",
                path=exc.path,
            ) from exc


def validate_descriptor_parameters(
    descriptor: str,
    parameters: Mapping[str, Any],
    schemas: Mapping[str, Any],
) -> None:
    """Validate a persisted descriptor configuration against its registry schema.

    ``DescriptorConfiguration`` validates the transport shape and JSON safety;
    this function validates the descriptor-specific shape.  It intentionally
    accepts the small legacy alias set used by the direct Python API while
    keeping canonical JSON keys equal to ``schemas``.
    """

    if not isinstance(parameters, Mapping):
        raise DescriptorConfigError(
            "descriptor configuration parameters must be an object",
            code="invalid_configuration",
            path=["parameters"],
        )
    aliases = LEGACY_PARAMETER_ALIASES.get(descriptor, {})
    allowed = set(schemas) | set(aliases) | {"output", "execution"}
    unknown = set(parameters) - allowed
    if unknown:
        name = sorted(str(item) for item in unknown)[0]
        raise DescriptorConfigError(
            f"unsupported descriptor parameter {name!r}",
            code="unknown_option",
            path=["parameters", name],
        )

    for alias, canonical in aliases.items():
        if (
            alias in parameters
            and canonical in parameters
            and parameters[alias] is not None
            and parameters[canonical] is not None
        ):
            raise DescriptorConfigError(
                f"descriptor parameters {alias!r} and {canonical!r} are aliases for the same option",
                code="conflicting_options",
                path=["parameters", alias],
                details={"canonical": canonical},
            )

    for name, schema in schemas.items():
        if not isinstance(schema, Mapping):  # DescriptorInfo already enforces this.
            continue
        source_name = name if name in parameters and parameters[name] is not None else ""
        if not source_name:
            source_name = next(
                (
                    alias
                    for alias, canonical in aliases.items()
                    if canonical == name
                    and alias in parameters
                    and parameters[alias] is not None
                ),
                name if name in parameters else "",
            )
        if not source_name:
            if schema.get("required", False):
                raise DescriptorConfigError(
                    f"missing required descriptor parameter {name!r}",
                    code="missing_required_parameter",
                    path=["parameters", name],
                )
            continue
        _validate_parameter_value(parameters[source_name], schema, ["parameters", source_name])

    for name in ("output", "execution"):
        if name in parameters and not isinstance(parameters[name], Mapping):
            raise DescriptorConfigError(
                f"descriptor {name} options must be an object",
                code="invalid_option_type",
                path=["parameters", name],
            )


def _validate_parameter_value(value: Any, schema: Mapping[str, Any], path: list[str]) -> None:
    """Validate one JSON value using the restricted descriptor schema dialect."""

    # ``None`` is the canonical JSON spelling for an omitted optional value.
    # Several kernels intentionally use it to request their own derived
    # default, so it is legal even when the semantic type is otherwise scalar.
    if value is None:
        if schema.get("required", False):
            raise DescriptorConfigError(
                "required descriptor parameter cannot be null",
                code="missing_required_parameter",
                path=path,
            )
        return

    schema_type = schema["type"]
    valid = True
    if schema_type == "integer":
        valid = (isinstance(value, int) and not isinstance(value, bool)) or (
            isinstance(value, float) and math.isfinite(value) and value.is_integer()
        )
    elif schema_type == "number":
        valid = (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and (not isinstance(value, float) or math.isfinite(value))
        )
    elif schema_type == "boolean":
        valid = isinstance(value, bool)
    elif schema_type in {"string", "enum"}:
        valid = isinstance(value, str) if schema_type == "string" else True
    elif schema_type == "species":
        valid = isinstance(value, (list, tuple)) and all(
            isinstance(item, int) and not isinstance(item, bool) for item in value
        )
    elif schema_type == "model":
        valid = isinstance(value, str) or isinstance(value, Mapping)
    elif schema_type == "array":
        # A few historical constructors use a scalar as a broadcast shorthand
        # for a per-species array. The GUI schema still emits the canonical
        # array form, while configuration loading keeps this compatibility
        # form so old manifests remain rebuildable.
        valid = isinstance(value, (list, tuple)) or (
            isinstance(value, (int, float)) and not isinstance(value, bool)
        )
    elif schema_type == "object":
        valid = isinstance(value, Mapping)
    if not valid:
        raise DescriptorConfigError(
            f"descriptor parameter value does not match type {schema_type!r}",
            code="invalid_parameter",
            path=path,
        )

    enum_values = schema.get("enum")
    if enum_values is not None and value not in enum_values:
        raise DescriptorConfigError(
            f"descriptor parameter value must be one of {list(enum_values)!r}",
            code="invalid_parameter",
            path=path,
        )

    if schema_type in {"integer", "number"}:
        numeric = value
        for field_name, predicate in (
            ("minimum", lambda bound: numeric < bound),
            ("maximum", lambda bound: numeric > bound),
            ("exclusiveMinimum", lambda bound: numeric <= bound),
            ("exclusiveMaximum", lambda bound: numeric >= bound),
        ):
            bound = schema.get(field_name)
            if bound is not None and predicate(bound):
                raise DescriptorConfigError(
                    f"descriptor parameter value violates {field_name}",
                    code="invalid_parameter",
                    path=path,
                )

    if schema_type == "array" and isinstance(value, (list, tuple)):
        item_schema = schema.get("items")
        if isinstance(item_schema, Mapping):
            for index, item in enumerate(value):
                _validate_parameter_value(item, item_schema, [*path, str(index)])
    elif schema_type == "object":
        properties = schema.get("properties")
        if isinstance(properties, Mapping):
            for name, item_schema in properties.items():
                if name in value and isinstance(item_schema, Mapping):
                    _validate_parameter_value(value[name], item_schema, [*path, str(name)])


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


__all__ = [
    "DESCRIPTOR_INFO_SCHEMA_VERSION",
    "DescriptorInfo",
    "LEGACY_PARAMETER_ALIASES",
    "validate_descriptor_parameters",
]
