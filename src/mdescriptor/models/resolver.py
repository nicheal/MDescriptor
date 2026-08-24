"""Deterministic model resource resolver with no implicit network access."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from os import PathLike

from ..core.errors import ModelLoadError
from .resource import ModelResource


class ModelResolver:
    """Resolve explicit paths or packaged defaults and validate their bytes."""

    def __init__(self, *, cache_dir: Path | None = None) -> None:
        self.cache_dir = cache_dir.expanduser() if cache_dir is not None else None

    def resolve(self, resource: ModelResource | str | PathLike[str]) -> Path:
        if not isinstance(resource, ModelResource):
            resource = ModelResource.from_value(resource)
        path = resource.path
        if not path.is_absolute() and not path.is_file() and self.cache_dir is not None:
            cached = self.cache_dir / path
            if cached.is_file():
                path = cached
        if not path.is_file():
            raise ModelLoadError(f"model resource does not exist: {path}")
        if resource.expected_sha256 is not None:
            digest = sha256(path.read_bytes()).hexdigest()
            if digest != resource.expected_sha256:
                raise ModelLoadError(
                    f"model resource checksum mismatch for {path}: "
                    f"expected {resource.expected_sha256}, got {digest}"
                )
        return path.resolve()

    def packaged(self, path: str | PathLike[str] | Path) -> Path:
        """Resolve a package-owned path without permitting network fallback."""

        return self.resolve(ModelResource.from_value(path))
