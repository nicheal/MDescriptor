# MTP upstream oracle

The oracle is compiled from `.deps/mlip-4-main.zip`.  The small helper calls
the official `MTP::AccumulateSiteEnergyGrads` and extracts only the trailing
MTP basis gradients; radial/species parameter gradients are not descriptor
features.  No MDescriptor descriptor implementation is linked into the helper.

Build and run from the repository root:

```bash
official_exe="$(benchmarks/_legacy_oracles/mtp/build.sh)"
.venv/bin/python benchmarks/_legacy_oracles/mtp/generate_golden.py \
  --official-exe "$official_exe"
```

For the larger local benchmark datasets, use the comparison runner:

```bash
.venv/bin/python benchmarks/_legacy_oracles/mtp/run_comparison.py \
  --official-exe "$official_exe"
```

The build uses the upstream static interface library.  Since the upstream
runtime creates a cache below `$HOME`, both build and runner redirect `HOME` to
the build/run scratch directory under `/tmp`.  Generated models, NDJSON inputs,
binaries and raw outputs remain build/results artifacts rather than oracle
source code.
