# SPDX-License-Identifier: LGPL-3.0-or-later
"""Pure-Python/NumPy reader for PyTorch ``.pt`` checkpoint files.

Implements just enough of the ``torch.save`` zip+pickle format to recover
tensor storages as NumPy arrays **without importing torch**:

- the archive is a zip whose ``<root>/data.pkl`` pickle describes the object
  graph and whose ``<root>/data/<key>`` entries hold raw tensor storages;
- tensors are reconstructed from ``torch._utils._rebuild_tensor_v2`` calls
  with ``persistent_load`` resolving ``('storage', ...)`` references.

Only plain dicts / lists / strings / numbers / NumPy arrays come back; any
other pickled global is replaced by an inert :class:`Stub` (checkpoints of
DeePMD-kit models contain only these, plus ``collections.OrderedDict``).
"""

from __future__ import annotations

import io
import pickle
import zipfile
from collections import (
    OrderedDict,
)
from typing import (
    Any,
)

import numpy as np

__all__ = [
    "load_torch_checkpoint",
]

#: torch storage class name -> numpy dtype
_STORAGE_DTYPES: dict[str, np.dtype] = {
    "DoubleStorage": np.dtype(np.float64),
    "FloatStorage": np.dtype(np.float32),
    "HalfStorage": np.dtype(np.float16),
    "LongStorage": np.dtype(np.int64),
    "IntStorage": np.dtype(np.int32),
    "ShortStorage": np.dtype(np.int16),
    "CharStorage": np.dtype(np.int8),
    "ByteStorage": np.dtype(np.uint8),
    "BoolStorage": np.dtype(np.bool_),
    "ComplexDoubleStorage": np.dtype(np.complex128),
    "ComplexFloatStorage": np.dtype(np.complex64),
}


class Stub:
    """Inert placeholder for unpickled objects we do not model."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        self.kwargs = kwargs

    def __call__(self, *args: Any, **kwargs: Any) -> Stub:
        return Stub(*args, **kwargs)

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"Stub{self.args!r}"


def _rebuild_tensor(
    storage: np.ndarray,
    storage_offset: int,
    size: tuple[int, ...],
    stride: tuple[int, ...],
    *args: Any,
) -> np.ndarray:
    """numpy equivalent of ``torch._utils._rebuild_tensor_v2/v3``."""
    if len(size) != len(stride) or any(int(dimension) < 0 for dimension in size):
        raise pickle.UnpicklingError("invalid tensor shape or stride")
    storage_offset = int(storage_offset)
    if storage_offset < 0:
        raise pickle.UnpicklingError("negative tensor storage offset")
    if any(int(step) < 0 for step in stride):
        raise pickle.UnpicklingError("negative tensor stride is not supported")
    if any(int(dimension) == 0 for dimension in size):
        return np.empty(tuple(int(dimension) for dimension in size), dtype=storage.dtype)
    last = storage_offset + sum(
        (int(dimension) - 1) * int(step)
        for dimension, step in zip(size, stride, strict=True)
    )
    if last >= storage.size:
        raise pickle.UnpicklingError("tensor view exceeds its storage")
    if len(size) == 0:
        return storage[storage_offset : storage_offset + 1].reshape(())
    itemsize = storage.dtype.itemsize
    base = storage[storage_offset:]
    out = np.lib.stride_tricks.as_strided(
        base,
        shape=tuple(size),
        strides=tuple(int(s) * itemsize for s in stride),
    )
    return out


class TorchCheckpointUnpickler(pickle.Unpickler):
    """Unpickler mapping torch globals onto NumPy reconstructions."""

    def __init__(self, file: io.BytesIO, archive: zipfile.ZipFile, root: str) -> None:
        super().__init__(file)
        self._archive = archive
        self._root = root

    # -- global resolution ------------------------------------------------
    def find_class(self, module: str, name: str) -> Any:
        full = f"{module}.{name}"
        if full == "collections.OrderedDict":
            return OrderedDict
        if full.startswith("torch._utils.") and name.startswith("_rebuild_tensor"):
            return _rebuild_tensor
        if full.startswith("torch._utils.") and name.startswith("_rebuild_parameter"):
            # Parameter / ParameterWithState: keep the wrapped tensor payload
            return lambda data, *args: data
        if module == "torch" and name.endswith("Storage"):
            return name  # marker; resolved by persistent_load
        # Keep the equivalent of ``torch.load(weights_only=True)`` strict:
        # no arbitrary stdlib or third-party global can be imported or called
        # while unpickling an untrusted archive.
        raise pickle.UnpicklingError(f"blocked checkpoint global {full!r}")

    # -- tensor storage resolution -----------------------------------------
    def persistent_load(self, pid: tuple) -> Any:
        tag, storage_name, key, _location, _numel = pid
        if tag != "storage":
            raise pickle.UnpicklingError(f"unsupported persistent id: {pid!r}")
        dtype = _STORAGE_DTYPES.get(storage_name)
        if dtype is None:
            raise pickle.UnpicklingError(f"unsupported storage type: {storage_name}")
        raw = self._archive.read(f"{self._root}/data/{key}")
        # little-endian on disk; astype makes a writable native-endian copy
        return np.frombuffer(raw, dtype=dtype.newbyteorder("<")).astype(dtype)


def load_torch_checkpoint(path: str) -> Any:
    """Load a ``torch.save`` checkpoint without torch.

    Parameters
    ----------
    path
        Path to the ``.pt`` / ``.pth`` file (zip archive format, the default
        since PyTorch 1.6).

    Returns
    -------
    Any
        The unpickled object with every tensor replaced by a NumPy array
        (little-endian, native dtype) and unknown objects by :class:`Stub`.
    """
    with zipfile.ZipFile(path) as archive:
        pkl_entries = [n for n in archive.namelist() if n.endswith("/data.pkl")]
        if not pkl_entries:
            raise ValueError(f"{path} is not a zip-format torch checkpoint")
        root = pkl_entries[0][: -len("/data.pkl")]
        data = archive.read(pkl_entries[0])
        return TorchCheckpointUnpickler(io.BytesIO(data), archive, root).load()
