"""Generate the MTP pytest golden from official MLIP-4 source code."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "src"))

from run_comparison import _load_two, _run_official  # noqa: E402

from mdescriptor import DescriptorConfiguration, create_descriptor  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _portable(value):
    """Replace checkout-specific absolute paths before committing the manifest."""
    if isinstance(value, str):
        root = str(ROOT.resolve()) + "/"
        if value.startswith(root):
            return "${PROJECT_ROOT}/" + value[len(root) :]
        return value
    if isinstance(value, dict):
        return {key: _portable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_portable(item) for item in value]
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-exe", type=Path, required=True)
    parser.add_argument(
        "--golden-dir",
        type=Path,
        default=ROOT / "tests/golden/mtp",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / "tests/data/mlip4_mtp6_carbon.json",
    )
    args = parser.parse_args()
    args.model.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [str(args.official_exe), "generate-model", str(args.model)],
        cwd=ROOT,
        check=True,
    )

    manifest_path = args.golden_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    batch, _ = _load_two()
    with tempfile.TemporaryDirectory(prefix="mdescriptor-mlip4-golden-") as temporary:
        official_values, _ = _run_official(
            args.official_exe,
            args.model,
            batch,
            Path(temporary),
            "golden",
        )

    configuration = copy.deepcopy(manifest["configuration"])
    parameters = configuration["parameters"]
    parameters["model"] = "${PROJECT_ROOT}/tests/data/mlip4_mtp6_carbon.json"
    parameters["species"] = [1, 6, 8, 24, 25, 26, 27, 28]
    project_configuration = copy.deepcopy(configuration)
    project_configuration["parameters"]["model"] = str(args.model.resolve())
    descriptor = create_descriptor(DescriptorConfiguration.from_dict(project_configuration))
    try:
        contract = descriptor.compute(batch)
    finally:
        descriptor.close()
    if official_values.shape != contract.values.shape:
        raise RuntimeError(
            f"oracle/project shape mismatch: {official_values.shape} != {contract.values.shape}"
        )
    if not np.allclose(contract.values, official_values, rtol=1e-9, atol=1e-11):
        raise RuntimeError("official MLIP-4 values do not match the project descriptor")

    output_path = args.golden_dir / manifest["expected_output"]
    np.savez_compressed(
        output_path,
        values=official_values,
        samples=np.asarray(contract.samples, dtype=np.int64),
    )
    manifest["configuration"] = configuration
    manifest["result"] = {
        "feature_count": int(contract.feature_count),
        "labels": list(contract.labels),
        "structure_ids": list(contract.structure_ids),
        "row_offsets": None if contract.row_offsets is None else contract.row_offsets.tolist(),
        "level": contract.level.value,
        "metadata": _portable(contract.metadata),
    }
    adapter = ROOT / "benchmarks/_legacy_oracles/mtp/official_mlip4_mtp.cpp"
    generator = Path(__file__).resolve()
    manifest["reference"] = {
        "kind": "external_upstream",
        "oracle": "MTP",
        "backend": "official MLIP-4 MTP::AccumulateSiteEnergyGrads",
        "source_archive": ".deps/mlip-4-main.zip",
        "source_sha256": _sha256(ROOT / ".deps/mlip-4-main.zip"),
        "adapter": "benchmarks/_legacy_oracles/mtp/official_mlip4_mtp.cpp",
        "adapter_sha256": _sha256(adapter),
        "generator": "benchmarks/_legacy_oracles/mtp/generate_golden.py",
        "generator_sha256": _sha256(generator),
        "model": "tests/data/mlip4_mtp6_carbon.json",
        "feature_extraction": "trailing MTP basis-gradient entries after radial/species parameters",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {output_path}")
    print(f"wrote {args.model}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
