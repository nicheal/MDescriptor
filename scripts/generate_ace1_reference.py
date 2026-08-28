#!/usr/bin/env python3
"""Generate the ACE golden directly from ``ACE1.jl-main.zip``.

For a fresh machine, pass ``--instantiate`` once so Julia resolves the
dependencies declared by the archived ACE1 project.  A previously resolved
checkout can instead be supplied with ``--project``; this is useful for an
offline, bit-for-bit regeneration.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from external_reference import sha256

ROOT = Path(__file__).resolve().parents[1]
EVALUATOR = Path(__file__).with_name("ace1_reference.jl")
GENERATOR = Path(__file__).resolve()
DEFAULT_ARCHIVE = ROOT / ".deps" / "ACE1.jl-main.zip"
DEFAULT_FIXTURE = ROOT / "tests" / "golden" / "ace"

ELEMENT_SYMBOLS = (
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn "
    "Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La "
    "Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po "
    "At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg "
    "Cn Nh Fl Mc Lv Ts Og"
).split()


def _julia_float(value: float) -> str:
    value = float(value)
    if not np.isfinite(value):
        raise ValueError("ACE fixture contains a non-finite number")
    return repr(value)


def _julia_matrix(values: np.ndarray) -> str:
    rows = [" ".join(_julia_float(item) for item in row) for row in values]
    return "[" + "; ".join(rows) + "]"


def _write_request(path: Path, manifest: dict[str, Any], arrays: Any) -> None:
    parameters = manifest["configuration"]["parameters"]
    transform = parameters["trans"]
    if transform["type"] != "PolyTransform" or float(transform.get("a", 1.0)) != 1.0:
        raise ValueError("ACE1 0.12.5 baseline supports PolyTransform with a=1 only")
    species = []
    for number in parameters["species"]:
        if not 1 <= int(number) <= len(ELEMENT_SYMBOLS):
            raise ValueError(f"unsupported atomic number: {number}")
        species.append(":" + ELEMENT_SYMBOLS[int(number) - 1])

    cells = np.asarray(arrays["cells"], dtype=np.float64)
    cell_literal = ", ".join(_julia_matrix(cell) for cell in cells)
    pbc = np.asarray(arrays["pbc"], dtype=bool)
    pbc_literal = _julia_matrix(pbc.astype(np.int8)).replace("1.0", "true").replace("0.0", "false")
    degrees = parameters["D"]
    degree_literal = "nothing" if degrees is None else repr(degrees)
    lines = [
        f"const SPECIES = [{', '.join(species)}]",
        f"const N_ORDER = {int(parameters['N'])}",
        f"const R0 = {_julia_float(parameters['r0'])}",
        f"const TRANSFORM_P = {_julia_float(transform['p'])}",
        f"const TRANSFORM_R0 = {_julia_float(transform['r0'])}",
        f"const WL = {_julia_float(parameters['wL'])}",
        f"const MAX_DEGREE = {_julia_float(parameters['maxdeg'])}",
        f"const DEGREES = {degree_literal}",
        f"const R_CUT = {_julia_float(parameters['rcut'])}",
        f"const R_IN = {_julia_float(parameters['rin'])}",
        f"const P_CUT = {int(parameters['pcut'])}",
        f"const P_IN = {int(parameters['pin'])}",
        f"const CONSTANTS = {'true' if parameters['constants'] else 'false'}",
        f"const NUMBERS = {np.asarray(arrays['numbers'], dtype=np.int32).tolist()}",
        f"const POSITIONS = {_julia_matrix(np.asarray(arrays['positions'], dtype=np.float64))}",
        f"const CELLS = Matrix{{Float64}}[{cell_literal}]",
        f"const PBCS = {pbc_literal}",
        f"const OFFSETS = {(np.asarray(arrays['offsets'], dtype=np.int64) + 1).tolist()}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _read_values(path: Path) -> np.ndarray:
    with path.open(encoding="utf-8") as stream:
        shape = tuple(int(item) for item in stream.readline().split())
        values = np.loadtxt(stream, dtype=np.float64, ndmin=2)
    if len(shape) != 2 or values.shape != shape:
        raise ValueError(f"ACE1 evaluator returned {values.shape}, declared {shape}")
    return values


def _extract_version(project: Path) -> str:
    text = (project / "Project.toml").read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    if match is None:
        raise ValueError("ACE1 Project.toml has no version")
    return match.group(1)


def _resolve_project(archive: Path, requested: Path | None, temp: Path) -> Path:
    if requested is not None:
        return requested.resolve()
    with zipfile.ZipFile(archive) as package:
        package.extractall(temp)
    roots = [
        path.parent for path in temp.rglob("Project.toml") if path.parent.name == "ACE1.jl-main"
    ]
    if len(roots) != 1:
        raise ValueError(f"expected one ACE1.jl-main project in {archive}, found {len(roots)}")
    return roots[0]


def _instantiate(julia: str, project: Path, environment: dict[str, str]) -> None:
    program = (
        'using Pkg; try Pkg.Registry.add(Pkg.RegistrySpec(url="https://github.com/ACEsuit/ACEregistry.git")) '
        "catch error; @info error; end; Pkg.instantiate()"
    )
    subprocess.run([julia, f"--project={project}", "-e", program], check=True, env=environment)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--project", type=Path)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--julia", default="julia")
    parser.add_argument("--depot", type=Path)
    parser.add_argument("--instantiate", action="store_true")
    parser.add_argument("--accept", action="store_true", help="replace the committed golden")
    args = parser.parse_args()

    archive = args.source_archive.resolve()
    fixture = args.fixture.resolve()
    manifest_path = fixture / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    environment = os.environ.copy()
    if args.depot is not None:
        environment["JULIA_DEPOT_PATH"] = str(args.depot.resolve())

    with tempfile.TemporaryDirectory(prefix="mdescriptor-ace1-reference-") as directory:
        temp = Path(directory)
        project = _resolve_project(archive, args.project, temp)
        if args.instantiate:
            _instantiate(args.julia, project, environment)
        request = temp / "request.jl"
        output = temp / "values.txt"
        with np.load(fixture / manifest["input"]) as arrays:
            _write_request(request, manifest, arrays)
        completed = subprocess.run(
            [
                args.julia,
                f"--project={project}",
                "--startup-file=no",
                str(EVALUATOR),
                str(request),
                str(output),
            ],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )
        values = _read_values(output)
        ace1_version = _extract_version(project)

    from mdescriptor import DescriptorConfiguration, StructureBatch, create_descriptor

    with np.load(fixture / manifest["input"]) as arrays:
        batch = StructureBatch(
            np.asarray(arrays["numbers"], dtype=np.int32),
            np.asarray(arrays["positions"], dtype=np.float64),
            np.asarray(arrays["cells"], dtype=np.float64),
            np.asarray(arrays["pbc"], dtype=np.int32),
            np.asarray(arrays["offsets"], dtype=np.int64),
            tuple(manifest["input_ids"]),
        )
    descriptor = create_descriptor(DescriptorConfiguration.from_dict(manifest["configuration"]))
    try:
        current = descriptor.compute(batch)
    finally:
        descriptor.close()
    delta = np.abs(current.values - values)
    np.testing.assert_allclose(current.values, values, rtol=1e-10, atol=1e-12)
    samples = np.asarray(current.samples, dtype=np.int64)

    versions = dict(line.split("=", 1) for line in completed.stdout.splitlines() if "=" in line)
    manifest["reference"] = {
        "kind": "ace1_julia_source",
        "source": "ACE1.Utils.rpi_basis + ACE1.Descriptors.descriptors",
        "source_archive": str(archive.relative_to(ROOT)),
        "source_archive_sha256": sha256(archive),
        "ace1_version": ace1_version,
        "runtime_versions": versions,
        "generator": str(GENERATOR.relative_to(ROOT)),
        "generator_sha256": sha256(GENERATOR),
        "evaluator": str(EVALUATOR.relative_to(ROOT)),
        "evaluator_sha256": sha256(EVALUATOR),
        "verification": {
            "rtol": 1e-10,
            "atol": 1e-12,
            "max_abs": float(delta.max(initial=0.0)),
            "mae": float(delta.mean()) if delta.size else 0.0,
        },
    }
    if args.accept:
        np.savez_compressed(fixture / manifest["expected_output"], values=values, samples=samples)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(f"accepted ACE1 reference {values.shape} into {fixture}")
    else:
        print(f"verified ACE1 reference {values.shape}; pass --accept to replace the golden")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
