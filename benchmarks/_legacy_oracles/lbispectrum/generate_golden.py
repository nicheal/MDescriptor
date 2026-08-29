"""Generate the LBispectrum pytest golden through PyXtal-FF and LAMMPS."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from importlib.metadata import version as package_version
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "src"))

from run_comparison import (  # noqa: E402
    ReferenceRunner,
    _Structure,
)

from mdescriptor import DescriptorConfiguration, StructureBatch, create_descriptor  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--golden-dir",
        type=Path,
        default=ROOT / "tests/golden/lbispectrum",
    )
    args = parser.parse_args()
    installed_pyxtal_ff = package_version("pyxtal-ff")
    if installed_pyxtal_ff != "0.2.3":
        raise RuntimeError(f"expected pyxtal-ff==0.2.3, got {installed_pyxtal_ff}")
    manifest_path = args.golden_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    with np.load(args.golden_dir / manifest["input"]) as arrays:
        numbers = np.asarray(arrays["numbers"], dtype=np.int32)
        positions = np.asarray(arrays["positions"], dtype=np.float64)
        cells = np.asarray(arrays["cells"], dtype=np.float64)
        pbc = np.asarray(arrays["pbc"], dtype=np.int32)
        offsets = np.asarray(arrays["offsets"], dtype=np.int64)
    ids = tuple(manifest["input_ids"])
    batch = StructureBatch(numbers, positions, cells, pbc, offsets, ids)
    structures = [
        _Structure(
            numbers[offsets[index] : offsets[index + 1]],
            positions[offsets[index] : offsets[index + 1]],
            cells[index],
            pbc[index],
        )
        for index in range(len(ids))
    ]
    parameters = manifest["configuration"]["parameters"]
    profile = {number: {"r": 1.75, "w": 1.0} for number in sorted(set(map(int, numbers)))}
    runner = ReferenceRunner(structures, profile)
    try:
        values = runner.compute()
    finally:
        runner.close()

    descriptor = create_descriptor(DescriptorConfiguration.from_dict(manifest["configuration"]))
    try:
        contract = descriptor.compute(batch)
    finally:
        descriptor.close()
    if values.shape != contract.values.shape:
        raise RuntimeError(f"oracle/project shape mismatch: {values.shape} != {contract.values.shape}")

    output_path = args.golden_dir / manifest["expected_output"]
    np.savez_compressed(output_path, values=values, samples=np.asarray(contract.samples, dtype=np.int64))
    adapter = ROOT / "benchmarks/_legacy_oracles/lbispectrum/run_comparison.py"
    generator = Path(__file__).resolve()
    manifest["reference"] = {
        "kind": "external_upstream",
        "oracle": "LBispectrum",
        "backend": "PyXtal-FF Bispectrum + LAMMPS ML-SNAP compute sna/atom",
        "source_archive": ".deps/lammps-stable.tar.gz",
        "source_sha256": _sha256(ROOT / ".deps/lammps-stable.tar.gz"),
        "pyxtal_ff": "pyxtal-ff==0.2.3",
        "adapter": "benchmarks/_legacy_oracles/lbispectrum/run_comparison.py",
        "adapter_sha256": _sha256(adapter),
        "generator": "benchmarks/_legacy_oracles/lbispectrum/generate_golden.py",
        "generator_sha256": _sha256(generator),
        "configuration": {
            "twojmax": int(parameters["twojmax"]),
            "diagonal": int(parameters["diagonal"]),
            "rfac0": float(parameters["rfac0"]),
            "rmin0": float(parameters["rmin0"]),
            "rcutfac": float(parameters["rcutfac"]),
            "profile": "r=1.75, w=1.0 for every element",
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
