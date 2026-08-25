# SPDX-License-Identifier: LGPL-3.0-or-later
"""Small backend-neutral random helpers used by the array-API port."""

import numpy as np


def random(size: int) -> np.ndarray:
    return np.random.random(size)
