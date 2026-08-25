# SPDX-License-Identifier: LGPL-3.0-or-later
"""Global environment / precision settings (trimmed for CPU inference).

Keeps the float-precision policy of upstream ``deepmd.env``. The compiled
``deepmd.lib`` dependency, the thread-setting helpers and the LMDB worker
count policy are removed — none are used by descriptor computation.
"""
import os

import numpy as np

__all__ = [
    "GLOBAL_ENER_FLOAT_PRECISION",
    "GLOBAL_NP_FLOAT_PRECISION",
    "global_float_prec",
]

# FLOAT_PREC
dp_float_prec = os.environ.get("DP_INTERFACE_PREC", "high").lower()
if dp_float_prec in ("high", ""):
    # default is high
    GLOBAL_NP_FLOAT_PRECISION = np.float64
    GLOBAL_ENER_FLOAT_PRECISION = np.float64
    global_float_prec = "double"
elif dp_float_prec == "low":
    GLOBAL_NP_FLOAT_PRECISION = np.float32
    GLOBAL_ENER_FLOAT_PRECISION = np.float64
    global_float_prec = "float"
else:
    raise RuntimeError(
        f"Unsupported float precision option: {dp_float_prec}. Supported: high,"
        "low. Please set precision with environmental variable "
        "DP_INTERFACE_PREC."
    )
