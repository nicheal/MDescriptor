"""Run the CUDA suite with unavailable or stale binaries treated as failures."""

from __future__ import annotations

import argparse
import importlib.machinery
import importlib.util
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _native_extension(directory: Path) -> Path:
    if not directory.is_dir():
        raise ImportError(f"native extension directory does not exist: {directory}")
    for suffix in importlib.machinery.EXTENSION_SUFFIXES:
        candidate = directory / f"_native{suffix}"
        if candidate.is_file():
            return candidate
    raise ImportError(f"no supported _native extension in {directory}")


def _preload_native(directory: Path) -> Path:
    """Load the selected native extension before pytest collects descriptors."""

    extension = _native_extension(directory)
    import mdescriptor

    spec = importlib.util.spec_from_file_location("mdescriptor._native", extension)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load _native from {extension}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["mdescriptor._native"] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop("mdescriptor._native", None)
        raise
    mdescriptor.__dict__["_native"] = module
    return extension


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run GPU tests as a required CUDA gate.",
    )
    parser.add_argument(
        "--plugin-dir",
        type=Path,
        required=True,
        help="directory containing the exact _cuda extension",
    )
    parser.add_argument(
        "--native-dir",
        type=Path,
        help="directory containing the exact _native extension (defaults to plugin-dir)",
    )
    args = parser.parse_args(argv)

    plugin_dir = args.plugin_dir.expanduser().resolve()
    native_dir = (args.native_dir or args.plugin_dir).expanduser().resolve()
    environment = os.environ.copy()
    environment.update(
        {
            "MDESCRIPTOR_REQUIRE_GPU": "1",
            "MDESCRIPTOR_CUDA_PLUGIN_DIR": str(plugin_dir),
            "MDESCRIPTOR_NATIVE_PLUGIN_DIR": str(native_dir),
            "MDESCRIPTOR_EXPECTED_NATIVE_PLUGIN_DIR": str(native_dir),
        }
    )
    try:
        native_extension = _preload_native(native_dir)
    except (ImportError, OSError) as error:
        print(f"required native preload failed: {error}", file=sys.stderr)
        return 1
    os.environ.update(environment)
    print(f"required CUDA plugin: {plugin_dir}")
    print(f"required native extension: {native_extension.resolve()}")
    return int(
        pytest.main(
            [
                "--import-mode=importlib",
                "--strict-markers",
                "-m",
                "gpu",
                str(ROOT / "tests"),
            ]
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
