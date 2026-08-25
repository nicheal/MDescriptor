"""Deterministic, explicitly extendable descriptor registry."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

from .spec import DescriptorSpec


class DescriptorRegistry:
    """A named registry with no import-time discovery side effects."""

    def __init__(
        self,
        specs: Iterable[DescriptorSpec] = (),
        *,
        parent: DescriptorRegistry | None = None,
        frozen: bool = False,
    ) -> None:
        self._specs: dict[str, DescriptorSpec] = {}
        self._parent = parent
        self._frozen = False
        if parent is not None:
            if not isinstance(parent, DescriptorRegistry):
                raise TypeError("parent must be a DescriptorRegistry")
        for spec in specs:
            self.register(spec)
        self._frozen = frozen

    def register(self, spec: DescriptorSpec) -> None:
        if self._frozen:
            raise TypeError("this descriptor registry is immutable; create a child registry to extend it")
        if not isinstance(spec, DescriptorSpec):
            raise TypeError("registry entries must be DescriptorSpec instances")
        if spec.name in self:
            raise ValueError(f"descriptor name {spec.name!r} is already registered")
        self._specs[spec.name] = spec

    @property
    def frozen(self) -> bool:
        """Whether this registry rejects further registrations."""

        return self._frozen

    def get(self, name: str) -> DescriptorSpec:
        if name in self._specs:
            return self._specs[name]
        if self._parent is not None:
            return self._parent.get(name)
        raise KeyError(f"unknown descriptor {name!r}") from None

    def names(self) -> tuple[str, ...]:
        inherited = () if self._parent is None else self._parent.names()
        return inherited + tuple(self._specs)

    def specs(self) -> tuple[DescriptorSpec, ...]:
        inherited = () if self._parent is None else self._parent.specs()
        return inherited + tuple(self._specs.values())

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and (
            name in self._specs or (self._parent is not None and name in self._parent)
        )

    def __len__(self) -> int:
        return len(self.names())

    def __iter__(self) -> Iterator[DescriptorSpec]:
        return iter(self.specs())
