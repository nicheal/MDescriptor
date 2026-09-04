"""Lazy standalone descriptor namespace derived from the registry."""

from .._namespace import install_registry_namespace as _install_registry_namespace

__all__, __getattr__, __dir__ = _install_registry_namespace(
    globals(), module_prefix="mdescriptor.descriptors.standalone"
)

del _install_registry_namespace
