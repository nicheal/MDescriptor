"""Model resource identities and path normalization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from os import PathLike, fspath
import re


@dataclass(frozen=True, slots=True)
class ModelResource:
    """An immutable model reference resolved before loading."""

    path: Path
    expected_sha256: str | None = None
    identifier: str | None = None

    def __post_init__(self) -> None:
        raw_path = fspath(self.path)
        if isinstance(raw_path, bytes):
            raw_path = raw_path.decode()
        if not str(raw_path).strip():
            raise ValueError("model resource path cannot be empty")
        path = Path(self.path).expanduser()
        if not str(path):
            raise ValueError("model resource path cannot be empty")
        digest = self.expected_sha256
        if digest is not None:
            digest = str(digest).lower()
            if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
                raise ValueError("expected_sha256 must be a 64-character hexadecimal digest")
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "expected_sha256", digest)
        if self.identifier is not None:
            object.__setattr__(self, "identifier", str(self.identifier))

    @classmethod
    def from_value(cls, value: str | PathLike[str] | Path, *, identifier: str | None = None) -> "ModelResource":
        raw_value = fspath(value)
        if isinstance(raw_value, bytes):
            raw_value = raw_value.decode()
        if not str(raw_value).strip():
            raise ValueError("model resource path cannot be empty")
        return cls(Path(value).expanduser(), identifier=identifier)
