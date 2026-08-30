"""Typed construction, execution, and output options.

The descriptor constructors deliberately accept the two option objects in this
module instead of an open mapping.  ``DescriptorConfiguration`` is the only
serialization seam for rebuilding an already-created descriptor.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal, TypeAlias

from .errors import DescriptorConfigError
from .json_value import freeze_json, thaw_json

JSONPrimitive: TypeAlias = str | int | float | bool | None
JSONValue: TypeAlias = JSONPrimitive | list["JSONValue"] | dict[str, "JSONValue"]

CONFIGURATION_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class OutputOptions:
    """How a descriptor result is represented."""

    dtype: Literal["float32", "float64"] = "float64"
    sparse: bool = False

    def __post_init__(self) -> None:
        if self.dtype not in {"float32", "float64"}:
            raise DescriptorConfigError("output dtype must be 'float32' or 'float64'")
        if not isinstance(self.sparse, bool):
            raise DescriptorConfigError("output sparse must be a boolean")


@dataclass(frozen=True, slots=True)
class ExecutionOptions:
    """Where and how a descriptor is evaluated."""

    device: str = "cpu"
    num_threads: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.device, str) or not self.device.strip():
            raise DescriptorConfigError("execution device cannot be empty")
        if self.device not in {"cpu", "cuda"}:
            raise DescriptorConfigError(
                "execution device must be exactly 'cpu' or 'cuda'",
                code="invalid_device",
                path=["device"],
                details={"supported": ["cpu", "cuda"]},
            )
        if isinstance(self.num_threads, bool):
            raise DescriptorConfigError("num_threads must be a positive integer or None")
        if self.num_threads is not None and (
            not isinstance(self.num_threads, int) or self.num_threads <= 0
        ):
            raise DescriptorConfigError("num_threads must be positive or None")


@dataclass(frozen=True, slots=True)
class DescriptorConfiguration:
    """Immutable, versioned JSON configuration for descriptor reconstruction."""

    schema_version: int
    descriptor: str
    parameters: Mapping[str, JSONValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (
            isinstance(self.schema_version, bool)
            or not isinstance(self.schema_version, int)
            or self.schema_version != CONFIGURATION_SCHEMA_VERSION
        ):
            raise DescriptorConfigError(
                f"unsupported descriptor configuration schema {self.schema_version!r}"
            )
        if not isinstance(self.descriptor, str):
            raise DescriptorConfigError("descriptor configuration name must be a string")
        name = self.descriptor.strip()
        if not name or any(character.isspace() for character in name):
            raise DescriptorConfigError("descriptor configuration name must be a non-empty token")
        if not isinstance(self.parameters, Mapping):
            raise DescriptorConfigError(
                "descriptor configuration parameters must be a mapping"
            )
        values = dict(self.parameters)
        try:
            frozen = freeze_json(values)
        except (TypeError, ValueError) as exc:
            raise DescriptorConfigError(f"descriptor configuration is not JSON-safe: {exc}") from exc
        if not isinstance(frozen, MappingProxyType):  # pragma: no cover - defensive
            raise DescriptorConfigError("descriptor configuration parameters must be an object")
        object.__setattr__(self, "descriptor", name)
        object.__setattr__(self, "parameters", frozen)

    def to_dict(self) -> dict[str, JSONValue]:
        """Return the canonical JSON-safe representation."""

        return {
            "schema_version": self.schema_version,
            "descriptor": self.descriptor,
            "parameters": thaw_json(self.parameters),
        }

    @classmethod
    def from_dict(cls, value: Any) -> DescriptorConfiguration:
        """Parse the exact versioned configuration object."""

        if not isinstance(value, Mapping):
            raise DescriptorConfigError("descriptor configuration must be a JSON object")
        expected = {"schema_version", "descriptor", "parameters"}
        unknown = set(value) - expected
        missing = expected - set(value)
        if unknown or missing:
            names = ", ".join(sorted(str(item) for item in unknown | missing))
            raise DescriptorConfigError(f"invalid descriptor configuration fields: {names}")
        parameters = value["parameters"]
        if not isinstance(parameters, Mapping):
            raise DescriptorConfigError("descriptor configuration parameters must be a JSON object")
        return cls(value["schema_version"], value["descriptor"], parameters)


__all__ = [
    "CONFIGURATION_SCHEMA_VERSION",
    "DescriptorConfiguration",
    "ExecutionOptions",
    "JSONValue",
    "OutputOptions",
]
