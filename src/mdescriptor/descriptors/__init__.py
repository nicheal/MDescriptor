"""Lazy public algorithm namespace derived from the built-in registry."""

from ._namespace import install_registry_namespace as _install_registry_namespace

__all__, __getattr__, __dir__ = _install_registry_namespace(globals())

del _install_registry_namespace
