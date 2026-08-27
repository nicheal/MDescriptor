"""Run the reproducible DPA native construction/steady-state benchmark.

The historical comparison runner remains the single source of truth for the
two locked workloads.  This wrapper adds process-level peak RSS, constructor
plus first-compute timing, and optional private stage timings from a profiling
build (``-DMDESCRIPTOR_DPA4_PROFILE=ON``).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "benchmarks" / "_work" / "run_dpa_comparison.py"
PROFILE_RE = re.compile(r"DPA4 profile (?P<label>[^.]+)\.(?P<stage>\S+) (?P<seconds>[0-9.]+) s")
COUNTER_RE = re.compile(r"DPA4 profile (?P<label>[^.]+)\.(?P<stage>\S+_count) (?P<value>\d+)")


def _peak_rss_kib(time_file: Path) -> int | None:
    if not time_file.is_file():
        return None
    text = time_file.read_text(encoding="utf-8", errors="replace")
    linux = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", text)
    if linux:
        return int(linux.group(1))
    macos = re.search(r"(\d+)\s+maximum resident set size", text)
    return None if macos is None else int(macos.group(1)) // 1024


def _private_stages(stderr: str) -> dict[str, dict[str, float]]:
    values: dict[str, dict[str, list[float]]] = {}
    for match in PROFILE_RE.finditer(stderr):
        label = match.group("label")
        stage = match.group("stage")
        values.setdefault(label, {}).setdefault(stage, []).append(float(match.group("seconds")))
    for match in COUNTER_RE.finditer(stderr):
        label = match.group("label")
        stage = match.group("stage")
        values.setdefault(label, {}).setdefault(stage, []).append(float(match.group("value")))
    return {
        label: {stage: statistics.median(samples) for stage, samples in stages.items()}
        for label, stages in values.items()
    }


def _run_case(
    *,
    descriptor: str,
    dataset: str,
    threads: int,
    mode: str,
    warmup: int,
    repeat: int,
    limit_frames: int,
    profile: bool,
    temporary: Path,
) -> dict[str, Any]:
    output = temporary / f"{descriptor.lower()}-{dataset}-{mode}-{threads}.json"
    time_file = temporary / f"time-{descriptor.lower()}-{mode}-{threads}.txt"
    command = [
        sys.executable,
        str(RUNNER),
        "--engine",
        "project",
        "--descriptor",
        descriptor,
        "--dataset",
        dataset,
        "--mode",
        mode,
        "--warmup",
        str(warmup),
        "--repeat",
        str(repeat),
        "--output",
        str(output),
    ]
    if limit_frames:
        command.extend(("--limit-frames", str(limit_frames)))
    environment = os.environ.copy()
    environment.update(
        {
            "DPA_BENCH_ENGINE": "project",
            "DPA_BENCH_THREADS": str(threads),
            "OMP_NUM_THREADS": str(threads),
            # The vendored scipy-openblas32 runtime is explicitly pinned to
            # one worker; only the DPA OpenMP pool is scanned here.
            "OPENBLAS_NUM_THREADS": "1",
        }
    )
    if profile:
        environment["MDESCRIPTOR_DPA4_PROFILE"] = "1"
    if sys.platform.startswith("linux") and Path("/usr/bin/time").exists():
        command = ["/usr/bin/time", "-v", "-o", str(time_file), *command]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise SystemExit(
            f"benchmark failed ({descriptor}, {dataset}, {threads}, {mode}):\n"
            f"{completed.stdout}\n{completed.stderr}"
        )
    record = json.loads(output.read_text(encoding="utf-8"))
    phases = record.get("phases_seconds", {})
    result: dict[str, Any] = {
        "threads": threads,
        "mode": mode,
        "warmup": warmup,
        "repeat": repeat,
        "median_kernel_seconds": record.get("median_kernel_seconds"),
        "p95_kernel_seconds": record.get("p95_kernel_seconds"),
        "kernel_samples_seconds": record.get("kernel_samples_seconds", []),
        "constructor_plus_first_compute_seconds": (
            float(phases.get("model", 0.0)) + float(phases.get("kernel", 0.0))
            if mode == "cold"
            else None
        ),
        "peak_rss_kib": _peak_rss_kib(time_file),
        "dataset_metadata": record.get("dataset_metadata", {}),
        "private_stages": _private_stages(completed.stderr),
    }
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--descriptor", choices=("DPA4", "DPA4C"), default="DPA4")
    parser.add_argument(
        "--dataset", choices=("two-structure-v1", "carbon_dataset_pbc"), default="two-structure-v1"
    )
    parser.add_argument("--threads", default="1,4,32", help="comma-separated OpenMP counts")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--limit-frames", type=int, default=0)
    parser.add_argument(
        "--profile", action="store_true", help="parse profiling-build stage timings"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if not RUNNER.is_file():
        raise SystemExit(f"benchmark runner is missing: {RUNNER}")
    threads = [int(value) for value in args.threads.split(",") if value]
    if not threads or any(value < 1 for value in threads):
        raise SystemExit("--threads must contain positive integers")
    with tempfile.TemporaryDirectory(prefix="mdescriptor-dpa-bench-") as directory:
        temporary = Path(directory)
        cases = {
            "cold": [
                _run_case(
                    descriptor=args.descriptor,
                    dataset=args.dataset,
                    threads=value,
                    mode="cold",
                    warmup=0,
                    repeat=1,
                    limit_frames=args.limit_frames,
                    profile=args.profile,
                    temporary=temporary,
                )
                for value in threads
            ],
            "steady": [
                _run_case(
                    descriptor=args.descriptor,
                    dataset=args.dataset,
                    threads=value,
                    mode="repeat",
                    warmup=args.warmup,
                    repeat=args.repeat,
                    limit_frames=args.limit_frames,
                    profile=args.profile,
                    temporary=temporary,
                )
                for value in threads
            ],
        }
    result = {
        "schema_version": 1,
        "descriptor": args.descriptor,
        "dataset": args.dataset,
        "engine": "project",
        "threads": threads,
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
