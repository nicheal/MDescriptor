"""Shared lazy namespace machinery for registry-backed descriptors."""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from typing import Any

from ..registry import builtin_registry


def install_registry_namespace(
    namespace: dict[str, Any],
    *,
    module_prefix: str | None = None,
) -> tuple[tuple[str, ...], Callable[[str], Any], Callable[[], list[str]]]:
    """Install lazy registry exports into a package namespace.

    ``module_prefix`` limits a sub-namespace to imports below one package;
    ``None`` exposes every built-in descriptor.  The returned functions are
    assigned to ``__getattr__`` and ``__dir__`` by the caller, keeping the
    import policy in one implementation while preserving normal module
    semantics.
    """

    prefix = None if module_prefix is None else module_prefix.rstrip(".")

    def belongs_to_namespace(import_path: str) -> bool:
        if prefix is None:
            return True
        return import_path == prefix or import_path.startswith(prefix + ".")

    def names() -> tuple[str, ...]:
        return tuple(
            spec.name
            for spec in builtin_registry
            if belongs_to_namespace(spec.import_path.split(":", 1)[0])
        )

    def module_getattr(name: str) -> Any:
        try:
            spec = builtin_registry.get(name)
        except KeyError as exc:
            raise AttributeError(name) from exc
        module_name, separator, attribute = spec.import_path.partition(":")
        if not separator or not belongs_to_namespace(module_name):
            raise AttributeError(name)
        value = getattr(import_module(module_name), attribute)
        namespace[name] = value
        return value

    def module_dir() -> list[str]:
        return sorted(set(namespace) | set(names()))

    return names(), module_getattr, module_dir


__all__ = ["install_registry_namespace"]
