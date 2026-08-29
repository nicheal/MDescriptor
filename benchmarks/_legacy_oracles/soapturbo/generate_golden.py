"""Generate the SOAPTurbo pytest golden from soap_turbo-master ``get_soap``.

The expected array is produced by the Fortran oracle; MDescriptor is queried
only for the result contract (labels, samples, and metadata) that pytest also
checks.  This keeps the golden value independent of the implementation under
test while retaining the normal public-API assertions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "src"))

from run_comparison import OfficialSoapTurbo, _official_function, _reference_inputs  # noqa: E402

from mdescriptor import DescriptorConfiguration, StructureBatch, create_descriptor  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _batch(path: Path, ids: tuple[str, ...]) -> StructureBatch:
    with np.load(path) as arrays:
        return StructureBatch(
            np.asarray(arrays["numbers"], dtype=np.int32),
            np.asarray(arrays["positions"], dtype=np.float64),
            np.asarray(arrays["cells"], dtype=np.float64),
            np.asarray(arrays["pbc"], dtype=np.int32),
            np.asarray(arrays["offsets"], dtype=np.int64),
            ids,
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-library", type=Path, required=True)
    parser.add_argument(
        "--golden-dir",
        type=Path,
        default=ROOT / "tests/golden/soapturbo",
    )
    args = parser.parse_args()

    manifest_path = args.golden_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    ids = tuple(manifest["input_ids"])
    batch = _batch(args.golden_dir / manifest["input"], ids)
    parameters = dict(manifest["configuration"]["parameters"])
    species = [int(value) for value in parameters["species"]]
    config = {
        key: parameters[key]
        for key in (
            "alpha_max",
            "l_max",
            "rcut_hard",
            "rcut_soft",
            "nf",
            "atom_sigma_r",
            "atom_sigma_r_scaling",
            "atom_sigma_t",
            "atom_sigma_t_scaling",
            "amplitude_scaling",
            "central_weight",
            "basis",
            "compression",
        )
    }
    function = _official_function(args.official_library)
    inputs = _reference_inputs(batch, species, float(config["rcut_hard"]))
    values = OfficialSoapTurbo(function, species, config).compute(inputs)

    descriptor = create_descriptor(DescriptorConfiguration.from_dict(manifest["configuration"]))
    try:
        contract = descriptor.compute(batch)
    finally:
        descriptor.close()
    if values.shape != contract.values.shape:
        raise RuntimeError(f"oracle/project shape mismatch: {values.shape} != {contract.values.shape}")

    output_path = args.golden_dir / manifest["expected_output"]
    np.savez_compressed(output_path, values=values, samples=np.asarray(contract.samples, dtype=np.int64))
    adapter = ROOT / "benchmarks/_legacy_oracles/soapturbo/soap_turbo_reference.f90"
    generator = Path(__file__).resolve()
    manifest["reference"] = {
        "kind": "external_upstream",
        "oracle": "SOAPTurbo",
        "backend": "soap_turbo-master Fortran get_soap",
        "source_archive": ".deps/soap_turbo-master.zip",
        "source_sha256": _sha256(ROOT / ".deps/soap_turbo-master.zip"),
        "adapter": "benchmarks/_legacy_oracles/soapturbo/soap_turbo_reference.f90",
        "adapter_sha256": _sha256(adapter),
        "generator": "benchmarks/_legacy_oracles/soapturbo/generate_golden.py",
        "generator_sha256": _sha256(generator),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
