# SPDX-License-Identifier: LGPL-3.0-or-later

import numpy as np


def get_index_between_two_maps(
    old_type_map: list[str], new_type_map: list[str]
) -> tuple[np.ndarray, bool]:
    """Return old indices for a new type map and whether a type was added."""

    old_index = {name: index for index, name in enumerate(old_type_map)}
    has_new_type = any(name not in old_index for name in new_type_map)
    indices = np.asarray(
        [old_index.get(name, 0) for name in new_type_map],
        dtype=np.int64,
    )
    return indices, has_new_type
