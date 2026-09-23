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
_CUDA_MODEL_SNAPSHOT_ABI: int | None = None


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

    global _CUDA_FACTORY, _CUDA_MODEL_SNAPSHOT_ABI
    if _CUDA_FACTORY is not None:
        return _CUDA_FACTORY

    module = importlib.import_module("mdescriptor._cuda")
    _CUDA_MODEL_SNAPSHOT_ABI = int(getattr(module, "MODEL_SNAPSHOT_ABI", 0))

    factory = getattr(module, "create_backend", None)
    if not callable(factory):
        error = ImportError("the CUDA plugin does not expose create_backend(name, options)")
        raise error
    _CUDA_FACTORY = factory
    return factory


def create_cuda_backend(name: str, options: dict[str, Any]) -> Any:
    """Create one CUDA backend instance on the first CUDA computation.

    The public error is deliberately structured here, while the original
    import exception remains chained for diagnostics and is never exposed as a
    stable API string.
    """

    from .core.errors import ModelLoadError, translate_backend_error

    try:
        factory = _cuda_factory()
    except (ImportError, OSError) as exc:
        raise translate_backend_error(
            exc,
            unavailable_message="CUDA backend is unavailable",
            failure_message="CUDA backend failed to initialize",
        ) from exc
    if options.get("model_data") is not None and _CUDA_MODEL_SNAPSHOT_ABI != 1:
        raise ModelLoadError(
            "CUDA plugin lacks immutable model snapshot support; rebuild MDescriptor"
        )
    try:
        return factory(name, dict(options))
    except Exception as exc:
        raise translate_backend_error(
            exc,
            unavailable_message="CUDA backend is unavailable",
            failure_message="CUDA backend failed to initialize",
        ) from exc


def create_cuda_predictor(name: str, options: dict[str, Any]) -> Any:
    """Create a prediction backend lazily with the descriptor CUDA error contract."""

    from .core.errors import ModelLoadError, translate_backend_error

    try:
        _cuda_factory()
        module = importlib.import_module("mdescriptor._cuda")
        factory = module.create_predictor
    except (ImportError, OSError, AttributeError) as exc:
        raise translate_backend_error(
            exc,
            unavailable_message="CUDA backend is unavailable",
            failure_message="CUDA predictor failed to initialize",
        ) from exc
    if options.get("model_data") is not None and _CUDA_MODEL_SNAPSHOT_ABI != 1:
        raise ModelLoadError(
            "CUDA plugin lacks immutable model snapshot support; rebuild MDescriptor"
        )
    try:
        return factory(name, dict(options))
    except Exception as exc:
        raise translate_backend_error(
            exc,
            unavailable_message="CUDA backend is unavailable",
            failure_message="CUDA predictor failed to initialize",
        ) from exc


__all__ = [
    "create_cuda_backend",
    "create_cuda_predictor",
    "native_extension_available",
    "preload_native",
]
