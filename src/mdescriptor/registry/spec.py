"""Descriptor definition records used by discovery, tests, and docs."""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TypeAlias


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
        level = str(self.level)
        if level not in {"atom", "structure", "pair"}:
            raise ValueError("descriptor spec level must be atom, structure, or pair")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "asset_policy", policy)
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

    def load_class(self) -> DescriptorClass:
        module_name, separator, attribute = self.import_path.partition(":")
        if not separator or not module_name or not attribute:
            raise ValueError(f"invalid descriptor import path {self.import_path!r}")
        module = importlib.import_module(module_name)
        return getattr(module, attribute)
