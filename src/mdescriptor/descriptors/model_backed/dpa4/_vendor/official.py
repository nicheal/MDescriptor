"""Removed Torch DPA4 adapter.

The public DPA4 descriptor now uses the shared pure-NumPy implementation in
``mdescriptor.descriptors.model_backed.dpa``.  This module remains only as a
private import-time marker for source trees that still enumerate the vendor
directory; it intentionally does not provide the former Torch API.
"""

from __future__ import annotations

from mdescriptor.descriptors.model_backed.dpa import (
    DpaCheckpointInfo,
    load_dpa_checkpoint,
    new_runtime,
    validate_dpa_checkpoint_mapping,
)

__all__ = [
    "DpaCheckpointInfo",
    "load_dpa_checkpoint",
    "new_runtime",
    "validate_dpa_checkpoint_mapping",
]
