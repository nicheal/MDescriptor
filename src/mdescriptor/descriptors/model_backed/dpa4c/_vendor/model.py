"""Removed Torch DPA4C adapter.

DPA4C inference is implemented once by the private pure-NumPy seam at
``mdescriptor.descriptors.model_backed.dpa``.  Keeping a marker module makes
the removal explicit and prevents a second model implementation from being
reintroduced under the old path.
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
