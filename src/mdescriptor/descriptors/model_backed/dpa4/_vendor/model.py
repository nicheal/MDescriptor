"""Removed Torch DPA4 model implementation.

The only DPA4 runtime is the shared pure-NumPy adapter in
``mdescriptor.descriptors.model_backed.dpa``.  This marker deliberately has
no compatibility classes or optional-runtime imports.
"""

from __future__ import annotations

__all__: list[str] = []
