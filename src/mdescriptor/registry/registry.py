"""Deterministic, explicitly extendable descriptor registry."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any

from .spec import DescriptorSpec


class DescriptorRegistry:
    """A named registry with no import-time discovery side effects."""

    def __init__(self, specs: Iterable[DescriptorSpec] = (), *, frozen: bool = False) -> None:
        self._specs: dict[str, DescriptorSpec] = {}
        self._aliases: dict[str, str] = {}
        self._frozen = False
        for spec in specs:
            self.register(spec)
        self._frozen = frozen

    def register(self, spec: DescriptorSpec) -> None:
        if self._frozen:
            raise TypeError("this descriptor registry is immutable; create a child registry to extend it")
        if not isinstance(spec, DescriptorSpec):
            raise TypeError("registry entries must be DescriptorSpec instances")
        if spec.name in self._specs:
            raise ValueError(f"descriptor name {spec.name!r} is already registered")
        for alias in spec.aliases:
            if alias == spec.name or alias in self._specs or alias in self._aliases:
                raise ValueError(f"descriptor alias {alias!r} is already registered")
        if spec.name in self._aliases:
            raise ValueError(f"descriptor name {spec.name!r} is already registered as an alias")
        self._specs[spec.name] = spec
        self._aliases.update({alias: spec.name for alias in spec.aliases})

    @property
    def frozen(self) -> bool:
        """Whether this registry rejects further registrations."""

        return self._frozen

    def get(self, name: str) -> DescriptorSpec:
        try:
            return self._specs[name]
        except KeyError:
            canonical = self._aliases.get(name)
            if canonical is not None:
                return self._specs[canonical]
            raise KeyError(f"unknown descriptor {name!r}") from None

    def names(self) -> tuple[str, ...]:
        return tuple(self._specs)

    def specs(self) -> tuple[DescriptorSpec, ...]:
        return tuple(self._specs.values())

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and (name in self._specs or name in self._aliases)

    def __len__(self) -> int:
        return len(self._specs)

    def __iter__(self) -> Iterator[DescriptorSpec]:
        return iter(self._specs.values())

    def create(self, name: str, **config: Any) -> Any:
        return self.get(name).load_class()(**config)

    def child(self) -> "DescriptorRegistry":
        return DescriptorRegistry(self._specs.values())
