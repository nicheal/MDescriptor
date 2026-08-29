"""Descriptor definition records used by discovery, tests, and docs."""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TypeAlias

from .info import DescriptorInfo


class AssetPolicy(str, Enum):
    NONE = "none"
    OPTIONAL = "optional"
    REQUIRED = "required"


DescriptorClass: TypeAlias = type[Any]
CAPABILITIES = frozenset(
    {"sparse", "model", "spin", "charge_spin", "num_threads", "cooperative_cancel"}
)


@dataclass(frozen=True, slots=True)
class DescriptorSpec:
    name: str
    import_path: str
    asset_policy: AssetPolicy
    backend: str
    level: str
    capabilities: frozenset[str] = field(default_factory=frozenset)
    optional_extra: str | None = None
    info: DescriptorInfo | None = None
    descriptor_version: str = "1"
    execution_engine: str | None = None

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        if not name or any(character.isspace() for character in name):
            raise ValueError("descriptor spec name must be a non-empty token")
        module, separator, attribute = self.import_path.partition(":")
        if not separator or not module or not attribute:
            raise ValueError("descriptor spec import_path must be 'module:attribute'")
        try:
            policy = AssetPolicy(self.asset_policy)
        except ValueError as exc:
            raise ValueError(f"unknown asset policy {self.asset_policy!r}") from exc
        backend = str(self.backend).strip()
        if not backend:
            raise ValueError("descriptor spec backend must be a non-empty token")
        descriptor_version = str(self.descriptor_version).strip()
        if not descriptor_version:
            raise ValueError("descriptor spec version must be a non-empty token")
        execution_engine = (
            backend
            if self.execution_engine is None
            else str(self.execution_engine).strip()
        )
        if not execution_engine:
            raise ValueError("descriptor spec execution_engine must be a non-empty token")
        level = str(self.level)
        if level not in {"atom", "structure", "pair"}:
            raise ValueError("descriptor spec level must be atom, structure, or pair")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "asset_policy", policy)
        object.__setattr__(self, "backend", backend)
        object.__setattr__(self, "descriptor_version", descriptor_version)
        object.__setattr__(self, "execution_engine", execution_engine)
        object.__setattr__(self, "level", level)
        capabilities = frozenset(str(item) for item in self.capabilities)
        unknown = capabilities - CAPABILITIES
        if unknown:
            raise ValueError(
                f"unknown descriptor capability(s): {', '.join(sorted(unknown))}"
            )
        object.__setattr__(self, "capabilities", capabilities)
        if self.optional_extra is not None:
            object.__setattr__(self, "optional_extra", str(self.optional_extra))
        if self.info is not None and not isinstance(self.info, DescriptorInfo):
            raise TypeError("descriptor spec info must be a DescriptorInfo or None")

    def load_class(self) -> DescriptorClass:
        module_name, separator, attribute = self.import_path.partition(":")
        if not separator or not module_name or not attribute:
            raise ValueError(f"invalid descriptor import path {self.import_path!r}")
        module = importlib.import_module(module_name)
        return getattr(module, attribute)
