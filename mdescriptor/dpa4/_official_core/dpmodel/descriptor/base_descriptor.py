# SPDX-License-Identifier: LGPL-3.0-or-later
"""Inference-only descriptor registry."""

from .make_base_descriptor import make_base_descriptor

BaseDescriptor = make_base_descriptor()

__all__ = ["BaseDescriptor"]
