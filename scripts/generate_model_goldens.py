"""Regenerate official model goldens only with explicit human acceptance."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from mdescriptor import StructureBatch
from mdescriptor.descriptors import DPA4, DPA4C

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXTURE = ROOT / "tests" / "data" / "dpa4c_air_h2o_golden.json"


def _digest(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def _batch(payload: dict) -> StructureBatch:
    return StructureBatch(
        np.asarray(payload["numbers"], dtype=np.int32),
        np.asarray(payload["positions"], dtype=np.float64),
        np.asarray(payload["cells"], dtype=np.float64),
        np.asarray(payload["pbc"], dtype=np.int32),
        np.asarray(payload["offsets"], dtype=np.int64),
        tuple(payload.get("ids", [f"golden-{index}" for index in range(len(payload["offsets"]) - 1)])),
    )


def _write(path: Path, name: str, model: Path, batch: StructureBatch) -> None:
    descriptor = (DPA4 if name == "DPA4" else DPA4C)(model=model)
    result = descriptor.compute(batch)
    payload = {
        "schema_version": 1,
        "descriptor": name,
        "model": {"path": model.name, "sha256": _digest(model)},
        "numbers": batch.numbers.tolist(),
        "positions": batch.positions.tolist(),
        "cells": batch.cells.tolist(),
        "pbc": batch.pbc.tolist(),
        "offsets": batch.offsets.tolist(),
        "ids": list(batch.ids),
        "level": result.level.value,
        "feature_count": result.feature_count,
        "labels": list(result.labels),
        "samples": result.samples.tolist(),
        "values": np.asarray(result.values).tolist(),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dpa4-checkpoint", type=Path, required=True)
    parser.add_argument("--dpa4c-checkpoint", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "tests" / "data")
    parser.add_argument("--accept", action="store_true")
    args = parser.parse_args(argv)
    if not args.accept:
        raise SystemExit("refusing to update model goldens without explicit --accept")
    source = json.loads(args.fixture.read_text(encoding="utf-8"))
    batch = _batch(source)
    _write(args.output_dir / "dpa4_air_h2o_golden.json", "DPA4", args.dpa4_checkpoint, batch)
    _write(args.output_dir / "dpa4c_air_h2o_golden.json", "DPA4C", args.dpa4c_checkpoint, batch)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
