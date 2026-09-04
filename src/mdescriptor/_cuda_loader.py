"""Private, lazy discovery of the optional CUDA extension.

The base package deliberately does not import the CUDA plugin at import time.
Tests and benchmark scripts may ask for it explicitly; keeping the discovery
logic here prevents each caller from growing a subtly different copy of the
same path and import handling.
"""

from __future__ import annotations

import importlib
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
    followed by the caller's explicit paths.  Only when those candidates are
    unavailable do we fall back to an installed plugin.  A directory is
    considered a candidate only when it contains a platform extension matching
    ``_cuda*``; this avoids importing an unrelated package path and keeps the
    base package driver-free.
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
        if candidate_text not in mdescriptor.__path__:
            mdescriptor.__path__.insert(0, candidate_text)
        else:
            mdescriptor.__path__.remove(candidate_text)
            mdescriptor.__path__.insert(0, candidate_text)
        importlib.invalidate_caches()
        try:
            module = importlib.import_module("mdescriptor._cuda")
        except (ImportError, OSError):
            continue
        location = getattr(module, "__file__", None)
        return Path(location).resolve().parent if location else candidate

    # If no explicit candidate was usable, fall back to an installed plugin.
    # This import is intentionally last: an editable checkout can have an old
    # wheel on ``sys.path``, but an explicit build directory must win above it.
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

    return any(
        path.is_file()
        for pattern in ("_cuda*.so", "_cuda*.pyd", "_cuda*.dylib")
        for path in directory.glob(pattern)
    )


__all__ = ["CudaPluginUnavailable", "load_cuda_plugin"]
