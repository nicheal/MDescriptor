"""Promote one accepted local benchmark snapshot into an independent test fixture."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEST_ROOT = ROOT / "tests" / "golden"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--tests-root", type=Path, default=DEFAULT_TEST_ROOT)
    parser.add_argument("--accept", action="store_true")
    args = parser.parse_args(argv)
    if not args.accept:
        raise SystemExit("refusing to promote a golden without explicit --accept")

    snapshot = args.snapshot.resolve()
    manifest = _load(snapshot / "manifest.json")
    accuracy = _load(snapshot / manifest["accuracy"])
    if not accuracy.get("passed"):
        raise SystemExit("cannot promote a benchmark snapshot whose accuracy check failed")
    descriptor = str(manifest["descriptor"])
    slug = descriptor.lower()
    target = args.tests_root / slug
    if target.exists():
        raise SystemExit(f"refusing to overwrite existing golden fixture: {target}")
    target.mkdir(parents=True)

    dataset_path = (snapshot / manifest["dataset"]["path"]).resolve()
    if not dataset_path.is_file():
        raise SystemExit(f"dataset does not exist: {dataset_path}")
    dataset_manifest = _load(dataset_path.with_name("manifest.json"))
    shutil.copy2(dataset_path, target / "input.npz")
    shutil.copy2(snapshot / manifest["files"]["reference"], target / "expected_output.npz")

    # The promoted fixture owns its input bytes.  Keep only immutable dataset
    # provenance here; a relative path into benchmarks would make tests depend
    # on the local benchmark tree at runtime.
    dataset = {
        "name": dataset_manifest["name"],
        "sha256": dataset_manifest["sha256"],
        "source": "promoted-local-dataset",
    }
    fixture = {
        "schema_version": 1,
        "descriptor": descriptor,
        "configuration": manifest["configuration"],
        "input": "input.npz",
        "input_ids": dataset_manifest["input"]["ids"],
        "expected_output": "expected_output.npz",
        "result": manifest["result"],
        "reference": manifest["reference"],
        "source_snapshot": manifest["snapshot"],
        "dataset": dataset,
        "nonperiodic": manifest["nonperiodic"],
        "tolerance": {
            "rtol": _load(snapshot / manifest["accuracy"])["rtol"],
            "atol": _load(snapshot / manifest["accuracy"])["atol"],
        },
    }
    (target / "manifest.json").write_text(
        json.dumps(fixture, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"promoted {descriptor} -> {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
