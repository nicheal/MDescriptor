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
_CUDA_FACTORY: Any = None
_CUDA_LOAD_ERROR: BaseException | None = None


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


def _cuda_factory() -> Any:
    """Resolve the optional CUDA backend without touching it during import."""

    global _CUDA_FACTORY, _CUDA_LOAD_ERROR
    if _CUDA_FACTORY is not None:
        return _CUDA_FACTORY
    if _CUDA_LOAD_ERROR is not None:
        raise _CUDA_LOAD_ERROR

    try:
        module = importlib.import_module("mdescriptor._cuda")
    except (ImportError, OSError) as exc:
        _CUDA_LOAD_ERROR = exc
        raise

    factory = getattr(module, "create_backend", None)
    if factory is None:
        factory = getattr(module, "backend_factory", None)
    if factory is None:
        backend_type = getattr(module, "CudaBackend", None)
        if backend_type is not None:
            factory = backend_type
    if not callable(factory):
        error = ImportError(
            "the CUDA plugin does not expose create_backend(name, options)"
        )
        _CUDA_LOAD_ERROR = error
        raise error
    _CUDA_FACTORY = factory
    return factory


def create_cuda_backend(name: str, options: dict[str, Any]) -> Any:
    """Create one CUDA backend instance on the first CUDA computation.

    The public error is deliberately structured here, while the original
    import exception remains chained for diagnostics and is never exposed as a
    stable API string.
    """

    try:
        factory = _cuda_factory()
    except (ImportError, OSError) as exc:
        from .core.errors import MDescriptorError

        raise MDescriptorError(
            "CUDA backend is unavailable",
            code="device_unavailable",
            path=["execution", "device"],
            details={"exception": type(exc).__name__},
        ) from exc
    try:
        return factory(name, dict(options))
    except (ImportError, OSError) as exc:
        from .core.errors import MDescriptorError

        raise MDescriptorError(
            "CUDA backend is unavailable",
            code="device_unavailable",
            path=["execution", "device"],
            details={"exception": type(exc).__name__},
        ) from exc
    except MemoryError as exc:
        from .core.errors import MDescriptorError

        raise MDescriptorError(
            "CUDA backend ran out of memory",
            code="backend_out_of_memory",
            path=["execution", "device"],
        ) from exc
    except Exception as exc:
        from .core.errors import MDescriptorError

        raise MDescriptorError(
            "CUDA backend failed to initialize",
            code="backend_error",
            path=["execution", "device"],
            details={"exception": type(exc).__name__},
        ) from exc


def load_cuda_plugin() -> Any:
    """Load the optional CUDA plugin on demand (private runtime hook)."""

    return _cuda_factory()


__all__ = [
    "create_cuda_backend",
    "load_cuda_plugin",
    "native_extension_available",
    "preload_native",
]
