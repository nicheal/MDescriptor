# SPDX-License-Identifier: LGPL-3.0-or-later
"""mdescriptor.descriptors.model_backed._vendor.dpa4desc: standalone DPA4/DPA4C descriptor computation package.

Extracted from deepmd-kit 3.2.0, trimmed to the CPU inference path: load a
DeePMD ``.pt`` checkpoint with the built-in pure-NumPy reader
(:mod:`mdescriptor.descriptors.model_backed._vendor.dpa4desc.weights`) and compute DPA4 / DPA4C descriptors for atomic
structures with the NumPy (dpmodel) backend. No torch, no GPU code.

Removed relative to upstream: the PyTorch backend (``pt_expt``), CUDA/Triton
kernels, training entry points, neighbor-statistics / update-sel machinery
and the training-data loading chain.
"""

__version__ = "0.3.1"

__all__ = [
    "DescrptDPA4",
    "DescrptDPA4C",
]


def __getattr__(name):
    if name == "DescrptDPA4":
        from mdescriptor.descriptors.model_backed._vendor.dpa4desc.dpmodel.descriptor.dpa4 import DescrptDPA4

        return DescrptDPA4
    if name == "DescrptDPA4C":
        from mdescriptor.descriptors.model_backed._vendor.dpa4desc.dpmodel.descriptor.dpa4c import DescrptDPA4C

        return DescrptDPA4C
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
