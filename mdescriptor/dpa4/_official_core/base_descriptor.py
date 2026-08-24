# SPDX-License-Identifier: LGPL-3.0-or-later
"""Inference-only descriptor registry used by the official DPA4 port."""


class BaseDescriptor:
    _registry: dict[str, type] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(descriptor_cls: type) -> type:
            cls._registry[name] = descriptor_cls
            return descriptor_cls

        return decorator


__all__ = ["BaseDescriptor"]
