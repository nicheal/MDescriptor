"""Lazy public algorithm namespace derived from the built-in registry."""

from __future__ import annotations

from importlib import import_module

from ..registry.builtins import builtin_registry


def _names() -> tuple[str, ...]:
    return builtin_registry.names()


__all__ = _names()


def __getattr__(name: str):
    try:
        spec = builtin_registry.get(name)
    except KeyError as exc:
        raise AttributeError(name) from exc
    module_name, separator, attribute = spec.import_path.partition(":")
    if not separator:
        raise AttributeError(name)
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_names()))
