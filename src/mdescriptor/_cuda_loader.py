"""Private, lazy discovery of the optional CUDA extension.

The base package deliberately does not import the CUDA plugin at import time.
Tests and benchmark scripts may ask for it explicitly; keeping the discovery
logic here prevents each caller from growing a subtly different copy of the
same path and import handling.
"""

from __future__ import annotations

import importlib
import importlib.machinery
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any, TypeAlias

import mdescriptor

PathLike: TypeAlias = str | os.PathLike[str]


class CudaPluginUnavailable(RuntimeError):
    """Raised when no usable source-tree or installed CUDA plugin exists."""


def load_cuda_plugin(*search_paths: PathLike) -> Path:
    """Load ``mdescriptor._cuda`` and return the directory that provided it.

    The optional ``MDESCRIPTOR_CUDA_PLUGIN_DIR`` override is tried first,
    followed by the caller's explicit paths (development build trees); only
    when those candidates are unavailable do we fall back to the extension
    installed beside this package.  A directory is considered a candidate
    only when it contains the extension module named ``_cuda``; this
    avoids importing an unrelated package path and keeps the base package
    driver-free.
    """

    candidates: list[Path] = []
    configured = os.environ.get("MDESCRIPTOR_CUDA_PLUGIN_DIR")
    if configured:
        candidates.append(Path(configured))
    candidates.extend(Path(path) for path in search_paths)

    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.expanduser().resolve()
        if candidate in seen or not _contains_cuda_extension(candidate):
            continue
        seen.add(candidate)
        candidate_text = str(candidate)
        loaded = sys.modules.get("mdescriptor._cuda")
        if loaded is not None:
            location = getattr(loaded, "__file__", None)
            loaded_path = Path(location).resolve().parent if location else None
            if loaded_path == candidate:
                return candidate
            raise CudaPluginUnavailable(
                "CUDA plugin is already loaded from "
                f"{loaded_path or '<unknown>'}; cannot switch to {candidate}"
            )
        original_package_path = list(mdescriptor.__path__)
        if candidate_text not in mdescriptor.__path__:
            mdescriptor.__path__.insert(0, candidate_text)
        else:
            mdescriptor.__path__.remove(candidate_text)
            mdescriptor.__path__.insert(0, candidate_text)
        importlib.invalidate_caches()
        try:
            module = _load_candidate(candidate)
        except (ImportError, OSError):
            # A broken candidate must not change future package discovery.
            # Restore the complete list so an existing entry keeps its order.
            mdescriptor.__path__[:] = original_package_path
            continue
        except BaseException:
            mdescriptor.__path__[:] = original_package_path
            raise
        location = getattr(module, "__file__", None)
        return Path(location).resolve().parent if location else candidate

    # If no explicit candidate was usable, fall back to the extension that
    # ships inside the installed package.  This import is intentionally last:
    # an editable checkout can have an old wheel on ``sys.path``, but an
    # explicit build directory must win above it.
    try:
        installed_module: Any = importlib.import_module("mdescriptor._cuda")
    except (ImportError, OSError):
        installed_module = None
    if installed_module is not None:
        location = getattr(installed_module, "__file__", None)
        if location:
            return Path(location).resolve().parent

    searched = ", ".join(str(path) for path in candidates) or "the installed package"
    raise CudaPluginUnavailable(f"CUDA plugin is not available; searched {searched}")


def _contains_cuda_extension(directory: Path) -> bool:
    """Return whether a candidate contains a platform extension module."""

    return _cuda_extension(directory) is not None


def _cuda_extension(directory: Path) -> Path | None:
    if not directory.is_dir():
        return None
    # Match the interpreter's complete extension suffix.  A prefix/glob
    # accepts unrelated files such as ``_cuda_test.so`` and can select an
    # incompatible ABI when several builds share a directory.
    for suffix in importlib.machinery.EXTENSION_SUFFIXES:
        path = directory / f"_cuda{suffix}"
        if path.is_file():
            return path
    dylib = directory / "_cuda.dylib"
    return dylib if dylib.is_file() else None


def _load_candidate(directory: Path) -> Any:
    """Load a candidate by filename, bypassing editable-package finders."""

    extension = _cuda_extension(directory)
    if extension is None:
        raise ImportError(f"no CUDA extension found in {directory}")
    spec = importlib.util.spec_from_file_location("mdescriptor._cuda", extension)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load CUDA extension from {extension}")
    module = importlib.util.module_from_spec(spec)
    module_name = "mdescriptor._cuda"
    previous = sys.modules.get(module_name)
    missing = object()
    previous_parent_attribute = getattr(mdescriptor, "_cuda", missing)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
        # ``exec_module`` alone does not perform the parent-package step that
        # the normal import machinery performs after successful execution.
        mdescriptor.__dict__["_cuda"] = module
    except BaseException:
        if previous is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous
        if previous_parent_attribute is missing:
            mdescriptor.__dict__.pop("_cuda", None)
        else:
            mdescriptor.__dict__["_cuda"] = previous_parent_attribute
        raise
    return module


__all__ = ["CudaPluginUnavailable", "load_cuda_plugin"]
