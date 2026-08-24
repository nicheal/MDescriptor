# SPDX-License-Identifier: LGPL-3.0-or-later
"""Numerically safe vector norms for the local NumPy/PyTorch paths."""

from typing import Any

import array_api_compat


def safe_for_vector_norm(
    x: Any,
    axis: int | tuple[int, ...] = -1,
    keepdims: bool = False,
) -> Any:
    xp = array_api_compat.array_namespace(x)
    squared = xp.sum(x * x, axis=axis, keepdims=keepdims)
    return xp.sqrt(xp.maximum(squared, xp.asarray(0.0, dtype=x.dtype)))
