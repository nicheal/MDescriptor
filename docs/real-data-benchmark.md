# Local descriptor benchmarks

`benchmarks/` is a local, ignored experiment area. It is not part of the
source distribution and no test imports code or data from it. Reusable runners
and promotion tools live under `scripts/benchmarking/` and `scripts/`.

The controlled runner uses CPU, one thread, two warmup calls and five measured
calls. Each result records raw timings, median, p95, Python/NumPy/platform
information, descriptor configuration, input hash and reference provenance.
Performance reports are informative and are not cross-machine CI gates.

## Local layout

```text
benchmarks/
  _datasets/<dataset>-<sha256>/
    structures.npz
    manifest.json
  <descriptor>/
    <YYYYMMDD>-<version>-<git-sha>-rNN/
      manifest.json
      candidate_output.npz
      reference_output.npz
      accuracy.json
      performance.json
```

Accepted snapshots are append-only. Large inputs are stored once under
`_datasets/` and referenced from descriptor manifests by relative path and
SHA256. A failed or exploratory run belongs in a local temporary directory and
must not be promoted.

## Two-structure accuracy fixture

The current fixture contains a 32-atom FCC Cantor alloy (`Cr-Mn-Fe-Co-Ni`) and
a non-periodic cluster of three H₂O molecules. The generator uses an explicit
reference wheel and refuses to write without `--accept`:

```bash
.venv/bin/python scripts/build_reference_wheel.py \
  --commit HEAD --output-dir /tmp/mdescriptor-reference-wheel
.venv/bin/python scripts/generate_descriptor_goldens.py \
  --descriptor SOAP \
  --reference-wheel /tmp/mdescriptor-reference-wheel/<wheel>.whl \
  --accept
```

After accuracy has passed, promotion copies the input and reference output to
`tests/golden/<descriptor>/`. Runtime tests never follow the source snapshot
path and never read `benchmarks/`:

```bash
.venv/bin/python scripts/promote_descriptor_golden.py \
  --snapshot benchmarks/soap/<snapshot> --accept
```

`SineMatrix`, `EwaldSumMatrix`, `MBTR`, `LMBTR`, `ValleOganov` and the current
Lode implementation are recorded as periodic-only cases. Their periodic
output is tested; their non-periodic contract is separately recorded as an
expected rejection. DPA snapshots record the official evaluator for periodic
rows and the explicit current-adapter fallback required by its evaluator's
singular-cell limitation.

## Existing real-data experiments

Large extxyz experiments, third-party adapters and source-derived reference
oracles may remain locally under the descriptor directories or a local
`_legacy_oracles/` directory. They are not package assets, are not enumerated
as repository files, and should be retained only when their manifest or
reference implementation still has a reproducibility purpose.
