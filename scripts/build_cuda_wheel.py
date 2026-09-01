"""Build the independent MDescriptor-CUDA wheel.

The wheel is built from ``packaging/cuda/pyproject.toml`` so the base
MDescriptor distribution stays CPU-only. CUDA architectures are deliberately
required at invocation time because they are part of the wheel compatibility
contract.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CUDA_PROJECT = ROOT / "packaging" / "cuda"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--arch",
        default=os.environ.get("CMAKE_CUDA_ARCHITECTURES"),
        help="CUDA architecture list, for example 75 or 75;86",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "dist")
    args = parser.parse_args(argv)
    if not args.arch:
        parser.error("--arch is required, for example --arch 75")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "build",
        str(CUDA_PROJECT),
        "--wheel",
        "--outdir",
        str(args.output_dir),
        f"-Ccmake.define.CMAKE_CUDA_ARCHITECTURES={args.arch}",
    ]
    return subprocess.run(command, cwd=ROOT, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
