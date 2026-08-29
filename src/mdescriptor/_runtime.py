"""Small platform runtime setup kept independent of descriptor imports."""

from __future__ import annotations

import ctypes
import importlib
import os
from pathlib import Path
from typing import Any

_DLL_DIRECTORIES: list[Any] = []
_NATIVE_HANDLES: list[Any] = []


def preload_native_binary() -> None:
    """Load the packaged native binary before an embedded host starts threads.

    Windows OpenBLAS builds can initialize a worker pool from DLL load code.
    Loading the extension binary during normal package startup moves that work
    before a GUI/stdio host starts its background pipe readers. The Python
    extension module is not initialized here, so static registry metadata stays
    independent of ``mdescriptor._native`` in ``sys.modules``.
    """

    if os.name != "nt":
        return
    loader = getattr(ctypes, "WinDLL", None)
    if loader is None:
        return
    package_directory = Path(__file__).resolve().parent
    add_dll_directory = getattr(os, "add_dll_directory", None)
    if add_dll_directory is not None:
        _DLL_DIRECTORIES.append(add_dll_directory(str(package_directory)))
    for path in sorted(package_directory.glob("_native*.pyd")):
        try:
            _NATIVE_HANDLES.append(loader(str(path)))
        except OSError:
            # Leave the normal import path to report a useful missing-runtime
            # error. Source checkouts without a built extension also land here.
            continue
        break


def preload_native() -> None:
    """Initialize the Python native module from a single-threaded startup hook."""

    importlib.import_module("mdescriptor._native")


__all__ = ["preload_native"]
