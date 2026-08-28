"""Explicit and named model resource identities."""

from __future__ import annotations

import re
from dataclasses import dataclass
from os import PathLike, fspath
from pathlib import Path
from typing import Any

from ..core.errors import DescriptorConfigError


@dataclass(frozen=True, slots=True)
class ModelResource:
    """A model reference that is either an explicit path or a named asset.

    The two forms are intentionally mutually exclusive.  Explicit paths are
    resolved exactly as supplied; named resources participate in the configured
    cache/package lookup performed by :class:`ModelResolver`.
    """

    path: Path | None = None
    name: str | None = None
    expected_sha256: str | None = None
    identifier: str | None = None

    def __post_init__(self) -> None:
        has_path = self.path is not None
        has_name = self.name is not None
        if has_path == has_name:
            raise DescriptorConfigError(
                "ModelResource must contain exactly one of path or name"
            )
        if has_path:
            raw_path = fspath(self.path)  # type: ignore[arg-type]
            if isinstance(raw_path, bytes):
                raw_path = raw_path.decode()
            if not str(raw_path).strip():
                raise DescriptorConfigError("model resource path cannot be empty")
            object.__setattr__(self, "path", Path(self.path).expanduser())  # type: ignore[arg-type]
        else:
            resource_name = str(self.name).strip()
            if not resource_name or resource_name.startswith(("/", "\\")):
                raise DescriptorConfigError("model resource name cannot be empty or absolute")
            parts = Path(resource_name).parts
            if ".." in parts:
                raise DescriptorConfigError("model resource name cannot escape its asset directory")
            object.__setattr__(self, "name", resource_name)

        digest = self.expected_sha256
        if digest is not None:
            digest = str(digest).lower()
            if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
                raise DescriptorConfigError(
                    "expected_sha256 must be a 64-character hexadecimal digest"
                )
        object.__setattr__(self, "expected_sha256", digest)
        if self.identifier is not None:
            identifier = str(self.identifier).strip()
            if not identifier:
                raise DescriptorConfigError("model resource identifier cannot be empty")
            object.__setattr__(self, "identifier", identifier)

    @classmethod
    def explicit(
        cls,
        path: str | PathLike[str],
        *,
        expected_sha256: str | None = None,
        identifier: str | None = None,
    ) -> ModelResource:
        raw = fspath(path)
        if isinstance(raw, bytes):
            raw = raw.decode()
        if not str(raw).strip():
            raise DescriptorConfigError("model resource path cannot be empty")
        return cls(
            path=Path(raw).expanduser(),
            expected_sha256=expected_sha256,
            identifier=identifier,
        )

    @classmethod
    def named(
        cls,
        name: str,
        *,
        expected_sha256: str | None = None,
        identifier: str | None = None,
    ) -> ModelResource:
        return cls(
            name=name,
            expected_sha256=expected_sha256,
            identifier=identifier,
        )

    @classmethod
    def from_value(
        cls,
        value: str | PathLike[str],
        *,
        identifier: str | None = None,
    ) -> ModelResource:
        """Build an explicit resource for internal path-based callers."""

        return cls.explicit(value, identifier=identifier)

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {"__type__": "ModelResource"}
        if self.path is not None:
            value["path"] = str(self.path)
        if self.name is not None:
            value["name"] = self.name
        if self.expected_sha256 is not None:
            value["expected_sha256"] = self.expected_sha256
        if self.identifier is not None:
            value["identifier"] = self.identifier
        return value

    @classmethod
    def from_dict(cls, value: Any) -> ModelResource:
        if not isinstance(value, dict) or value.get("__type__") != "ModelResource":
            raise DescriptorConfigError("invalid serialized ModelResource")
        allowed = {"__type__", "path", "name", "expected_sha256", "identifier"}
        unknown = set(value) - allowed
        if unknown:
            raise DescriptorConfigError(
                f"invalid serialized ModelResource field(s): {', '.join(sorted(unknown))}"
            )
        if ("path" in value) == ("name" in value):
            raise DescriptorConfigError("serialized ModelResource must contain path or name")
        return cls(
            path=Path(value["path"]).expanduser() if "path" in value else None,
            name=value.get("name"),
            expected_sha256=value.get("expected_sha256"),
            identifier=value.get("identifier"),
        )


__all__ = ["ModelResource"]
