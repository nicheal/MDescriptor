"""Render and check the controlled descriptor inventory.

The registry is intentionally the only declaration seam.  This script is the
small, explicit documentation seam: ``--write`` updates the checked-in page
when a descriptor spec changes, while ``--check`` is the CI gate and never
modifies the working tree.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from mdescriptor import (
    DESCRIPTOR_INFO_SCHEMA_VERSION,
    builtin_registry,
    list_descriptors,
)

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "descriptor-inventory.md"


def render() -> str:
    specs = tuple(builtin_registry)
    names = ", ".join(spec.name for spec in specs)
    rows = []
    for spec in specs:
        group = "standalone" if ".standalone." in spec.import_path else "model_backed"
        capabilities = ", ".join(sorted(spec.capabilities)) or "—"
        extra = spec.optional_extra or "—"
        category = "—" if spec.info is None else spec.info.category
        parameters = (
            "—"
            if spec.info is None or not spec.info.parameters
            else ", ".join(spec.info.parameters)
        )
        rows.append(
            f"| {spec.name} | `{group}` | `{category}` | `{spec.asset_policy.value.upper()}` | "
            f"`{spec.backend}` | `{spec.level}` | {capabilities} | {parameters} | `{extra}` |"
        )
    return """# Descriptor inventory / 描述符清单

This page is generated from the immutable built-in registry.  It is a
controlled artifact: run `python scripts/check_descriptor_inventory.py
--write` when a registry spec changes, and keep the `--check` gate in CI.

<!-- registry-names: {names} -->

| Name | Directory group | Category | Asset policy | Backend | Level | Capabilities | Parameters | Extra |
|---|---|---|---|---|---|---|---|---|
{rows}

## Static descriptor metadata

Each built-in entry carries `DescriptorInfo` schema version
`{DESCRIPTOR_INFO_SCHEMA_VERSION}`.  Query it without importing an algorithm
implementation or resolving a model:

```python
import mdescriptor

metadata = mdescriptor.describe_descriptor("SOAP")
```

The returned object is JSON-safe and contains the canonical parameter schema,
execution devices, input periodicity, output representation, and model asset
policy.  Historical Python constructor aliases remain available to direct
callers but are intentionally omitted from the canonical GUI parameter list.
Each built-in parameter schema also exposes a GUI-facing ``display_name`` and
``description``.  The mapping key remains the canonical constructor name and
must be used when serializing values.

The canonical algorithm imports are:

```python
from mdescriptor.descriptors import SOAP, ACSF, ACE, MTP, NEP, DPA4, DPA4C
```

The root package exposes contracts, registry queries and the configuration
factory.  Algorithm classes are deliberately not re-exported from the root.

## Backend boundaries

- C++17/OpenMP kernels implement the standalone numerical formulas.
- `mdescriptor._native` is the private pybind11 extension name.
- The `MDESCRIPTOR_BUILD_CUDA` target installs `mdescriptor._cuda` alongside
  the CPU extension in the same package, covering all 28 built-in descriptors
  when a CUDA toolkit is available. The registry is the canonical capability
  source; the extension's extended dispatch table must cover the same names.
- Python adapters own input packing, lifecycle, typed options and result
  normalization.
- DPA4/DPA4C use the bundled NumPy checkpoint loader and fallback inference
  core; supported default graphs are executed by C++17/OpenMP, and official
  ``.pt`` files are parsed without importing Torch.
- NEP, DPA4 and DPA4C model files are resolved locally and verified by
  streaming SHA-256 before loading.

## Result contract

Every public class returns `DescriptorResult` from `compute(...)`.  Values are
two-dimensional and `samples` is a contiguous `int64` array with the fixed
shape `[structure]`, `[structure, local_atom]`, or
`[structure, local_atom_1, local_atom_2, shift_a, shift_b, shift_c]` according
to the output level.  Metadata uses the versioned JSON-safe schema; descriptors
retain configuration and metadata after `close()`.

Common representation and execution settings are passed as
`OutputOptions`/`ExecutionOptions`.  Sparse output is SciPy CSR and fails at
construction when the optional dependency is unavailable.

## Optional dependencies

The base import and DPA4/DPA4C computation do not import Torch.  Install
`.[ase]` for ASE input conversion and `.[sparse]` for SciPy CSR output.
Official `.pt` loads use the package's restricted NumPy checkpoint reader;
network downloads are not implemented.

## Native source map

| Family | Source |
|---|---|
| SOAP / ACSF / ACE / SOAPTurbo | `cpp/src/standalone/soap.cpp`, `acsf.cpp`, `ace.cpp`, `soap_turbo.cpp` |
| matrices / MBTR | `cpp/src/standalone/*_matrix.cpp`, `matrix_dispatch.cpp`, `mbtr.cpp` |
| local descriptors | `cpp/src/standalone/atomic_composition.cpp`, `neighbor_list.cpp`, `sorted_distances.cpp`, `spherical_expansion*.cpp` |
| rotational / MTP / C00PS | `cpp/src/standalone/ead.cpp`, `rotational_descriptors.cpp`, `mtp*.cpp`, `c00ps_mlff.cpp` |
| NEP | `cpp/src/common/nep.cpp` |
| shared helpers | `cpp/include/mdescriptor/detail/`, `cpp/src/common/` |

Algorithm numerical behavior is frozen during this layout refactor.  Formula
changes require a separate reference/golden update.
""".format(
        names=names,
        rows="\n".join(rows),
        DESCRIPTOR_INFO_SCHEMA_VERSION=DESCRIPTOR_INFO_SCHEMA_VERSION,
    ).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="fail when the generated page differs")
    mode.add_argument("--write", action="store_true", help="write the generated page")
    args = parser.parse_args(argv)

    expected = render()
    if args.write:
        DOC.write_text(expected, encoding="utf-8")
        return 0
    actual = DOC.read_text(encoding="utf-8")
    if actual != expected:
        match = re.search(r"<!-- registry-names: (.*?) -->", actual)
        documented = () if match is None else tuple(item.strip() for item in match.group(1).split(","))
        raise SystemExit(
            "descriptor inventory is stale; run scripts/check_descriptor_inventory.py --write "
            f"(documented={documented!r}, actual={list_descriptors()!r})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
