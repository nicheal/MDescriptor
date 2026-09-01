"""Validate the contents and ELF closure of an MDescriptor-CUDA wheel."""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path
from zipfile import ZipFile

RUNTIME_LIBRARY_NAMES = ("libcudart", "libcublas", "libcublasLt")


def _readelf(path: Path) -> str:
    try:
        result = subprocess.run(
            ["readelf", "-d", str(path)],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        raise SystemExit("readelf is required to verify the CUDA wheel") from exc
    if result.returncode != 0:
        raise SystemExit(f"readelf failed for {path}: {result.stderr}")
    return result.stdout


def verify(wheel: Path) -> None:
    with ZipFile(wheel) as archive:
        names = set(archive.namelist())
        runtime_files = sorted(
            name
            for name in names
            if name.startswith("mdescriptor/.cuda_libs/")
            and any(
                Path(name).name.startswith(prefix + ".so.")
                for prefix in RUNTIME_LIBRARY_NAMES
            )
        )
        expected_runtime_count = len(RUNTIME_LIBRARY_NAMES)
        if len(runtime_files) != expected_runtime_count:
            raise SystemExit(
                f"expected {expected_runtime_count} CUDA runtime files, found {runtime_files}"
            )
        if "mdescriptor/licenses/NVIDIA-CUDA-EULA.txt" not in names:
            raise SystemExit("CUDA wheel is missing NVIDIA-CUDA-EULA.txt")

        extensions = sorted(
            name
            for name in names
            if name.startswith("mdescriptor/_cuda") and name.endswith((".so", ".pyd"))
        )
        if len(extensions) != 1:
            raise SystemExit(f"expected one CUDA extension, found {extensions}")

        metadata_names = sorted(name for name in names if name.endswith(".dist-info/METADATA"))
        if len(metadata_names) != 1:
            raise SystemExit(f"expected one wheel METADATA file, found {metadata_names}")
        metadata = archive.read(metadata_names[0]).decode("utf-8")
        if "Name: MDescriptor-CUDA\n" not in metadata:
            raise SystemExit("wheel metadata does not identify MDescriptor-CUDA")

        with tempfile.TemporaryDirectory(prefix="mdescriptor-cuda-wheel-") as temporary:
            root = Path(temporary)
            for name in [extensions[0], *runtime_files]:
                destination = root / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(archive.read(name))
            plugin = root / extensions[0]
            dynamic = _readelf(plugin)
            if "$ORIGIN/.cuda_libs" not in dynamic:
                raise SystemExit("CUDA extension does not contain the private runtime RPATH")
            try:
                resolved = subprocess.run(
                    ["ldd", str(plugin)],
                    check=False,
                    capture_output=True,
                    text=True,
                )
            except OSError as exc:
                raise SystemExit("ldd is required to verify the CUDA wheel") from exc
            if resolved.returncode != 0 or "not found" in resolved.stdout:
                raise SystemExit(f"unresolved CUDA wheel dependencies:\n{resolved.stdout}")
            for library in runtime_files:
                library_name = Path(library).name
                if f".cuda_libs/{library_name}" not in resolved.stdout:
                    raise SystemExit(
                        f"{library_name} was not resolved from the wheel's .cuda_libs directory"
                    )

    print(f"verified {wheel} ({len(RUNTIME_LIBRARY_NAMES)} CUDA runtime libraries)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("wheel", type=Path)
    args = parser.parse_args(argv)
    if not args.wheel.is_file():
        parser.error(f"wheel does not exist: {args.wheel}")
    verify(args.wheel)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
