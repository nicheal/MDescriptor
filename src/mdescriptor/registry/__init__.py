"""Descriptor discovery and construction API."""

from .builtins import BUILTIN_REGISTRY, BUILTIN_SPECS
from .registry import DescriptorRegistry
from .spec import AssetPolicy, DescriptorSpec


def list_descriptors(*, registry: DescriptorRegistry = BUILTIN_REGISTRY) -> tuple[str, ...]:
    return registry.names()


def get_descriptor(name: str, *, registry: DescriptorRegistry = BUILTIN_REGISTRY) -> type:
    return registry.get(name).load_class()


def create_descriptor(name: str, *, registry: DescriptorRegistry = BUILTIN_REGISTRY, **config):
    return registry.create(name, **config)


__all__ = [
    "AssetPolicy",
    "BUILTIN_REGISTRY",
    "BUILTIN_SPECS",
    "DescriptorRegistry",
    "DescriptorSpec",
    "create_descriptor",
    "get_descriptor",
    "list_descriptors",
]
