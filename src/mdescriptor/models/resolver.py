"""Deterministic model resource resolution with streaming integrity checks."""

from __future__ import annotations

import os
from dataclasses import dataclass
from hashlib import sha256
from os import PathLike
from pathlib import Path
from typing import Literal

from ..core.errors import ModelLoadError
from .resource import ModelResource


@dataclass(frozen=True, slots=True)
class ResolvedModel:
    """The concrete file identity selected by a resolver."""

    resource: ModelResource
    path: Path
    digest: str
    source: Literal["explicit", "cache", "package"]


class ModelResolver:
    """Resolve explicit paths or named resources without network access."""

    def __init__(
        self,
        *,
        cache_dir: Path | None = None,
        package_dir: Path | None = None,
    ) -> None:
        if cache_dir is None:
            configured = os.environ.get("MDESCRIPTOR_MODEL_CACHE")
            cache_dir = None if not configured else Path(configured)
        self.cache_dir = None if cache_dir is None else Path(cache_dir).expanduser()
        self.package_dir = (
            Path(package_dir).expanduser()
            if package_dir is not None
            else Path(__file__).resolve().parent / "assets"
        )

    def resolve(self, resource: ModelResource | str | PathLike[str]) -> ResolvedModel:
        if not isinstance(resource, ModelResource):
            resource = ModelResource.from_value(resource)
        if resource.path is not None:
            return self._resolve_file(resource, resource.path, "explicit")

        assert resource.name is not None  # guaranteed by ModelResource
        if self.cache_dir is not None:
            cached = self.cache_dir / resource.name
            if cached.exists():
                if not cached.is_file():
                    raise ModelLoadError(f"cached model resource is not a file: {cached}")
                # A present but corrupt cache entry is a hard failure.  Falling
                # through to a packaged artifact would hide a deployment error.
                return self._resolve_file(resource, cached, "cache")

        packaged = self.package_dir / resource.name
        if not packaged.is_file():
            raise ModelLoadError(
                f"named model resource {resource.name!r} is not present in the cache or package"
            )
        return self._resolve_file(resource, packaged, "package")

    def packaged(self, resource: ModelResource | str | PathLike[str]) -> ResolvedModel:
        """Resolve a package-owned resource explicitly, bypassing the cache."""

        if not isinstance(resource, ModelResource):
            resource = ModelResource.from_value(resource)
        if resource.path is not None:
            path = resource.path
        else:
            assert resource.name is not None
            path = self.package_dir / resource.name
        if not path.is_file():
            raise ModelLoadError(f"packaged model resource does not exist: {path}")
        return self._resolve_file(resource, path, "package")

    @staticmethod
    def digest(path: Path) -> str:
        """Return a streaming SHA-256 digest for one model file."""

        try:
            with path.open("rb") as stream:
                checksum = sha256()
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    checksum.update(chunk)
        except OSError as exc:
            raise ModelLoadError(f"cannot read model resource {path}: {exc}") from exc
        return checksum.hexdigest()

    def _resolve_file(
        self,
        resource: ModelResource,
        path: Path,
        source: Literal["explicit", "cache", "package"],
    ) -> ResolvedModel:
        path = path.expanduser().resolve()
        if not path.is_file():
            raise ModelLoadError(f"model resource does not exist: {path}")
        actual = self.digest(path)
        expected = resource.expected_sha256
        if expected is not None and actual != expected:
            raise ModelLoadError(
                f"model resource checksum mismatch for {path}: expected {expected}, got {actual}"
            )
        return ResolvedModel(resource, path, actual, source)


__all__ = ["ModelResolver", "ModelResource", "ResolvedModel"]
