# SPDX-License-Identifier: LGPL-3.0-or-later
"""Project-local array-API runtime for the official DPA4 inference core."""

from .common import (
    DEFAULT_PRECISION,
    GLOBAL_ENER_FLOAT_PRECISION,
    GLOBAL_NP_FLOAT_PRECISION,
    NativeOP,
    PRECISION_DICT,
    RESERVED_PRECISION_DICT,
)

__all__ = [
    "DEFAULT_PRECISION",
    "GLOBAL_ENER_FLOAT_PRECISION",
    "GLOBAL_NP_FLOAT_PRECISION",
    "NativeOP",
    "PRECISION_DICT",
    "RESERVED_PRECISION_DICT",
]
