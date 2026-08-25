# SPDX-License-Identifier: LGPL-3.0-or-later
"""Shared constants and minimal utilities (trimmed for CPU inference).

Only the pieces actually used by the descriptor computation are kept:
``VALID_ACTIVATION`` / ``VALID_PRECISION`` literal sets and ``j_get_type``.
The training-data / config-file / path-discovery helpers of upstream
``deepmd.common`` were removed.
"""
from typing import (
    Any,
    get_args,
)

try:
    from typing import Literal  # python >=3.8
except ImportError:
    from typing import Literal  # type: ignore

from mdescriptor.descriptors.model_backed._vendor.dpa4desc.env import (
    GLOBAL_NP_FLOAT_PRECISION,
)

__all__ = [
    "GLOBAL_NP_FLOAT_PRECISION",
    "VALID_ACTIVATION",
    "VALID_PRECISION",
    "j_get_type",
]

_PRECISION = Literal["default", "float16", "bfloat16", "float32", "float64"]
_ACTIVATION = Literal[
    "relu",
    "relu6",
    "softplus",
    "sigmoid",
    "tanh",
    "gelu",
    "gelu_tf",
    "silu",
    "silut",
    "none",
    "linear",
]
# get_args is new in py38
VALID_PRECISION: set[_PRECISION] = set(get_args(_PRECISION))
VALID_ACTIVATION: set[_ACTIVATION] = set(get_args(_ACTIVATION))


def j_get_type(data: dict, class_name: str = "object") -> str:
    """Get the type from the data.

    Parameters
    ----------
    data : dict
        the data
    class_name : str, optional
        the name of the class for error message, by default "object"

    Returns
    -------
    str
        the type
    """
    try:
        return data["type"]
    except KeyError as e:
        raise KeyError(f"the type of the {class_name} should be set by `type`") from e
