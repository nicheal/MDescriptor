"""Small platform runtime setup kept independent of descriptor imports."""

from __future__ import annotations

import ctypes
import importlib
import importlib.machinery
import importlib.util
import os
from pathlib import Path
from typing import Any

_DLL_DIRECTORIES: list[Any] = []
_NATIVE_HANDLES: list[Any] = []
_NATIVE_PRELOAD_ATTEMPTED = False


def native_extension_available() -> bool:
    """Return whether the private native module resolves to an extension binary."""

    # On Windows, an extension file can exist while one of its DLL
    # dependencies is unavailable.  ``preload_native_binary`` performs the
    # loadability check before the registry is assembled, so do not fall back
    # to the filename-only probe after that check has run.
    if os.name == "nt" and _NATIVE_PRELOAD_ATTEMPTED:
        return bool(_NATIVE_HANDLES)

    try:
        spec = importlib.util.find_spec("mdescriptor._native")
    except (ImportError, ValueError):
        return False
    if spec is None or spec.origin is None:
        return False
    origin = str(spec.origin)
    return any(origin.endswith(suffix) for suffix in importlib.machinery.EXTENSION_SUFFIXES)


def preload_native_binary() -> None:
    """Load the packaged native binary before an embedded host starts threads.

    Windows OpenBLAS builds can initialize a worker pool from DLL load code.
    Loading the extension binary during normal package startup moves that work
    before a GUI/stdio host starts its background pipe readers. The Python
    extension module is not initialized here, so static registry metadata stays
    independent of ``mdescriptor._native`` in ``sys.modules``.
    """

    global _NATIVE_PRELOAD_ATTEMPTED

    if os.name != "nt":
        return
    _NATIVE_PRELOAD_ATTEMPTED = True
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
