"""Lazy standalone descriptor namespace derived from the registry."""

from __future__ import annotations

from importlib import import_module

from ...registry import builtin_registry


def _names() -> tuple[str, ...]:
    return tuple(
        spec.name
        for spec in builtin_registry
        if spec.import_path.split(":", 1)[0].startswith(
            "mdescriptor.descriptors.standalone"
        )
    )


__all__ = _names()


def __getattr__(name: str):
    try:
        spec = builtin_registry.get(name)
    except KeyError as exc:
        raise AttributeError(name) from exc
    module_name, separator, attribute = spec.import_path.partition(":")
    if not separator or not module_name.startswith("mdescriptor.descriptors.standalone"):
        raise AttributeError(name)
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_names()))
