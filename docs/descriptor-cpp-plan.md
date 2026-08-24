# C++ descriptor architecture / C++ 描述符架构

状态：布局重构已实施，DPA 系列核心保留为后续独立阶段。

## Boundaries

`mdescriptor._native` is the only pybind11 extension exposed to Python
internals. Its C++17 kernels own periodic neighbor construction, cancellation,
finite-value validation and numerical loops. Python owns configuration parsing,
input conversion and result metadata.

The public Python layer is split into `descriptors/standalone` and
`descriptors/model_backed`. Standalone descriptors do not load models. NEP,
DPA4 and DPA4C resolve model resources explicitly; Torch is a runtime extra,
not a build dependency.

## Shared native helpers

Small value types and linear algebra live in the named header
`cpp/include/mdescriptor/detail/math3.hpp`. Descriptor-family headers retain
family-specific validation and dispatch (`descriptor_common.hpp`,
`extra_common.hpp`, `local_common.hpp`, `matrix_common.hpp`). New shared code
must be placed in a named domain header rather than a catch-all `utils` file.

## Build and release

The CMake source list is the single native build manifest. The Python package
uses the `src/` layout and installs the extension as `_native`.

Supported model-wheel target policy:

- CPython 3.10–3.14;
- Linux x86_64 and Windows x86_64;
- macOS arm64 (macOS 14+);
- Linux arm64 can be added later; Windows arm64 is not promised.

The release workflow builds sdist and wheels separately, tests the installed
artifact, and keeps Torch out of the base build. DPA tests that exercise
checkpoints run in a model-extra job.

## Contract tests

Native contract tests cover periodic input validation, cancellation, finite
outputs and atom/structure/pair row layouts. Python contract tests cover:

- lazy registry imports and immutable built-ins;
- `DescriptorResult` JSON-safe metadata and `feature_count`;
- close/context-manager behavior;
- explicit model resolution and strict `.pt` loading.

Numerical behavior is frozen for this refactor. The dedicated DPA4/DPA4C
implementation refactor is intentionally deferred until after this boundary is
stable.
