# SPDX-License-Identifier: LGPL-3.0-or-later
"""Small descriptor registry used by the project-local DPA4 port.

The upstream factory also wires descriptors into training and data-system
plugins. Those services are outside this inference-only module, so the port
keeps only the common registry needed by DPA4.
"""

from typing import Any


def make_base_descriptor(
    t_tensor: type[Any] | None = None,
    fwd_method_name: str = "forward",
) -> type:
    """Return the minimal inference descriptor base class."""

    class BaseDescriptor:
        _registry: dict[str, type] = {}

        @classmethod
        def register(cls, name: str):
            def decorator(descriptor_cls: type) -> type:
                cls._registry[name] = descriptor_cls
                return descriptor_cls

            return decorator

    return BaseDescriptor
