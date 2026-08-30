# GUI adaptation baseline

This document is the versioned contract for a GUI that discovers and runs
MDescriptor descriptors. It describes the public seams; private kernel
classes, native symbols, and model internals are not GUI interfaces.

The package exposes this document through `mdescriptor.gui_baseline()` and
reports `baseline_version: "2"` from `mdescriptor.get_runtime_info()`.

## Discovery and versions

`list_descriptors()` returns the registered descriptor names. For each name,
`describe_descriptor(name)` returns a JSON-safe object with these fields:

`schema_version`, `name`, `descriptor_version`, `display_name`, `description`,
`category`, `level`, `backend`, `execution_engine`, `capabilities`,
`parameters`, `execution`, `input`, `output`, and `asset`.

`schema_version` is the metadata-envelope version and is currently `3`.
`descriptor_version` identifies the individual descriptor contract and is
currently the string `"1"` for the built-ins. A GUI should check both before
persisting or reconstructing metadata-driven configurations.

`list_descriptors(detailed=True)` returns one JSON-safe summary record per
descriptor with `name`, `descriptor_version`, `display_name`, `category`, and
`level`. The default call remains the immutable tuple of names.

`backend` identifies the adapter/backend compatibility label. It is not a
promise about the currently selected native implementation. `execution_engine`
identifies the default numerical engine. For example, DPA4 and DPA4C expose
`backend: "numpy"` for their model/runtime adapter and
`execution_engine: "cpp"` when the native extension is available.

## Parameters and execution

Parameter schemas use JSON-safe values. Supported schema types are `integer`,
`number`, `boolean`, `string`, `enum`, `species`, `model`, `array`, and
`object`; object schemas may contain nested `properties`, and array schemas
may contain `items`. The `parameters` object is the canonical constructor
surface; implementation-only options must not be synthesized by the GUI.

The key in `parameters` is the canonical constructor/configuration name and
must be preserved when a configuration is serialized. Each built-in parameter
also provides a GUI-facing `display_name` and a `description`. The GUI should
use `display_name` as the field label and `description` as the tooltip, while
continuing to submit values under the mapping key. These presentation fields
are deliberately independent of names such as `r_cut`, `cutoff`, and `rcut`,
so descriptors can share a label without changing their numerical interfaces.
For nested object properties, the same fields may be present on the nested
schema.

The current built-ins advertise CPU execution and the first CUDA-capable
descriptors (`NeighborList`, `SphericalExpansion`, `SoapRadialSpectrum`, and
`SoapPowerSpectrum`) advertise `execution.devices == ["cpu", "cuda"]`.
The remaining built-ins advertise `execution.devices == ["cpu"]`. A GUI
should offer only the devices declared by the selected descriptor. Selecting
an undeclared device produces a structured `unsupported_device` configuration
error; CUDA plugin or driver failures use `device_unavailable`,
`backend_error`, or `backend_out_of_memory`.

CUDA execution is synchronous at the public boundary and returns the same
host-owned NumPy/CSR result contract as CPU execution. The CUDA plugin is
loaded lazily on the first CUDA `compute()` call, so static discovery and
`describe_descriptor()` do not require a CUDA driver. Cancellation is
cooperative and is reported as the public `CancelledError`.

## Input policy

`StructureBatch` is the canonical flattened input. GUI frame records can be
passed to `StructureBatch.from_frames()` when they expose `numbers`,
`positions`, `cell`, `pbc`, and `id`; the helper generates the flattened
arrays and cumulative offsets. Each frame is either
`isolated` (all PBC flags false) or `fully_periodic` (all flags true); partial
periodicity is invalid. A descriptor that only supports fully periodic input
rejects isolated frames at the public adapter boundary with
`UnsupportedPeriodicityError` (a `DescriptorInputError` subclass), code
`unsupported_periodicity`, and path `["input", "periodicity"]`.
LodeSphericalExpansion is periodic-only.

The GUI should use the declared `input.periodicity`, `input.mixed_periodicity`,
spin fields, and the structured error payload instead of inferring support
from implementation names or exception text. Descriptors declaring both
`isolated` and `fully_periodic` accept those frame kinds in one batch;
periodic-only descriptors reject a batch containing an isolated frame.

On Windows, import `mdescriptor` during single-threaded startup, before the GUI
starts any background stdin/stdio reader. The package preloads the native
binary at that point; an embedding host may also call
`mdescriptor.preload_native()` explicitly during startup.

## Results and lifecycle

Configurations and results carry their own schema versions. Descriptors are
closed explicitly after use; metadata snapshots remain available after close.
Unknown descriptor names, unsupported options, unavailable assets, and
unsupported input should be shown as structured errors rather than treated as
native implementation failures.
