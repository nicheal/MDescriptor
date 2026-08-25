"""Build an isolated reference wheel from the frozen source commit."""

from __future__ import annotations

import argparse
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COMMIT = "60dccbb"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--commit", default=DEFAULT_COMMIT)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="mdescriptor-reference-source-") as temp_name:
        temp = Path(temp_name)
        archive_path = temp / "source.tar"
        with archive_path.open("wb") as archive:
            subprocess.run(
                ["git", "archive", "--format=tar", args.commit],
                cwd=ROOT,
                check=True,
                stdout=archive,
            )
        source = temp / "source"
        source.mkdir()
        with tarfile.open(archive_path) as archive:
            archive.extractall(source)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "build",
                "--wheel",
                "--no-isolation",
                "--outdir",
                str(args.output_dir.resolve()),
            ],
            cwd=source,
            check=True,
        )

    wheels = tuple(args.output_dir.glob("*.whl"))
    if len(wheels) != 1:
        raise SystemExit(f"expected one reference wheel, found {len(wheels)}")
    print(f"reference wheel: {wheels[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
