"""Explicit boundary between public descriptors and private numeric kernels."""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from dataclasses import replace
from typing import Any, ClassVar

from .control import ComputeControl
from .descriptor import Descriptor
from .errors import (
    CancelledError,
    DescriptorConfigError,
    DescriptorInputError,
    NativeCancelledError,
)
from .input import StructureBatch
from .options import (
    CONFIGURATION_SCHEMA_VERSION,
    DescriptorConfiguration,
    ExecutionOptions,
    OutputOptions,
)
from .result import DescriptorResult, _json_safe, format_values

# The kernel option names are kept in one declaration so every public class
# gets the same explicit keyword-only boundary.  Kernels are private
# implementation details; callers never pass a config mapping or an open
# ``**kwargs`` bag.
_PUBLIC_OPTION_NAMES: dict[str, frozenset[str]] = {
    "SOAP": frozenset({
        "species", "rbf", "n_max", "l_max", "sigma", "average",
        "weighting", "r_cut", "compression",
    }),
    "ACSF": frozenset({
        "species", "r_cut",
        "g2_params", "G2", "g2", "g3_params", "G3", "g3",
        "g4_params", "G4", "g4", "g5_params", "G5", "g5",
    }),
    "SOAPTurbo": frozenset({
        "species", "alpha_max", "l_max", "rcut_hard", "rcut_soft",
        "nf", "radial_enhancement", "basis", "compression", "compress_mode",
        "atom_sigma_r", "atom_sigma_r_scaling",
        "atom_sigma_t", "atom_sigma_t_scaling", "amplitude_scaling",
        "central_weight", "central_species",
    }),
    "C00PSMLFF": frozenset({
        "species", "r_cut", "cutoff", "n_radial", "n_max", "l_max",
        "cutoff_function", "include_radial", "include_angular", "normalize_radial",
        "normalize_angular", "super_vector", "radial_weight", "angular_weight",
        "exclude_self_interaction",
    }),
    "SortedDistances": frozenset({
        "species", "cutoff", "max_neighbors", "separate_neighbor_types",
    }),
    "SphericalExpansion": frozenset({
        "species", "cutoff", "density_width", "max_radial", "max_angular",
    }),
    "SphericalExpansionByPair": frozenset({
        "species", "cutoff", "density_width", "max_radial", "max_angular",
    }),
    "SoapRadialSpectrum": frozenset({
        "species", "cutoff", "density_width", "max_radial", "max_angular",
    }),
    "SoapPowerSpectrum": frozenset({
        "species", "cutoff", "density_width", "max_radial", "max_angular",
    }),
    "MTP": frozenset({
        "species", "model", "min_dist",
        "max_dist", "r_cut", "cutoff", "radial_basis_size", "radial_funcs_count",
        "max_rank", "l_max", "max_level", "level", "radial_basis_type",
    }),
    "NEP": frozenset(),
    "DPA4": frozenset(),
    "DPA4C": frozenset({"calibrate"}),
    "SNAP": frozenset({"weights", "lmax", "rcut", "normalize_U"}),
    "LBispectrum": frozenset({
        "twojmax", "diagonal", "rfac0", "rmin0", "rcutfac", "element_profile",
        "element_radii", "weights", "rcut", "normalize_U",
    }),
    "LodeSphericalExpansion": frozenset({
        "species", "cutoff", "density_width", "max_radial", "max_angular",
        "k_cutoff", "exponent", "radial_radius",
    }),
    "ValleOganov": frozenset({
        "species", "function", "n", "sigma", "r_cut", "geometry", "grid", "weighting",
        "periodic", "normalize_gaussians", "normalization",
    }),
}


def adapt_result(result: Any) -> DescriptorResult:
    """Convert a kernel result into the one public result schema."""

    if isinstance(result, DescriptorResult):
        return result
    return DescriptorResult(
        result.values,
        result.level,
        result.structure_ids,
        result.row_offsets,
        result.labels,
        result.metadata,
        getattr(result, "samples", None),
        _atom_row_offsets=getattr(result, "_atom_row_offsets", None),
    )


def _coerce_options(value: Any, option_type: type[Any], name: str) -> Any:
    if isinstance(value, option_type):
        return value
    raise DescriptorConfigError(f"{name} must be {option_type.__name__}")


def _constructor_parameters(kernel_type: type[Any]) -> dict[str, inspect.Parameter]:
    try:
        signature = inspect.signature(kernel_type.__init__)
    except (TypeError, ValueError):  # pragma: no cover - unusual extension types
        return {}
    return {
        name: parameter
        for name, parameter in signature.parameters.items()
        if name != "self"
    }


def _accepts_keyword(parameters: Mapping[str, inspect.Parameter], name: str) -> bool:
    parameter = parameters.get(name)
    return parameter is not None or any(
        item.kind is inspect.Parameter.VAR_KEYWORD for item in parameters.values()
    )


def _has_explicit_keyword(parameters: Mapping[str, inspect.Parameter], name: str) -> bool:
    parameter = parameters.get(name)
    return parameter is not None and parameter.kind in {
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    }


def _apply_output(result: DescriptorResult, options: OutputOptions) -> DescriptorResult:
    """Apply the common output representation after any backend computation."""

    values = format_values(result.values, dtype=options.dtype, sparse=options.sparse)

    metadata = dict(result.metadata)
    metadata["output"] = {"dtype": options.dtype, "sparse": options.sparse}
    return replace(result, values=values, metadata=metadata)


class DescriptorAdapter(Descriptor):
    """Centralize lifecycle, options, and result handling for one kernel.

    The wrapped implementation is deliberately private. Public callers get
    only :class:`Descriptor`'s lifecycle and ``compute`` contract plus the
    explicitly declared ``feature_count`` property.
    """

    kernel_type: ClassVar[type[Any]]
    name: ClassVar[str]
    wrap_value_errors_as_config: ClassVar[bool] = True
    allowed_options: ClassVar[frozenset[str] | None] = None
    requires_species: ClassVar[bool] = False

    def _initialize(self, options: Mapping[str, Any]) -> None:
        """Initialize a kernel from already-bound public options."""

        super().__init__()
        kwargs = dict(options)
        output_value = kwargs.pop("output", None)
        execution_value = kwargs.pop("execution", None)
        # Generated public constructors use ``None`` as the sentinel for an
        # omitted keyword; let each kernel's declared default apply.
        kwargs = {key: value for key, value in kwargs.items() if value is not None}
        self._output_options = (
            OutputOptions()
            if output_value is None
            else _coerce_options(output_value, OutputOptions, "output")
        )
        self._execution_options = (
            ExecutionOptions()
            if execution_value is None
            else _coerce_options(execution_value, ExecutionOptions, "execution")
        )
        if self._output_options.sparse:
            try:
                import scipy.sparse  # noqa: F401
            except ImportError as exc:  # pragma: no cover - optional extra
                raise DescriptorConfigError(
                    "output.sparse requires the optional 'sparse' extra"
                ) from exc
        self._validate_public_call(kwargs)
        preloaded = getattr(self, "_preloaded_model_weights", None)
        if preloaded is not None and getattr(
            self.kernel_type, "accepts_preloaded_checkpoint", False
        ):
            kwargs["_checkpoint"] = preloaded
        parameters = _constructor_parameters(self.kernel_type)
        resolved_model_digest = getattr(self, "_resolved_model_digest", None)
        if resolved_model_digest is not None and _accepts_keyword(parameters, "model_digest"):
            # This implementation identity is deliberately absent from the
            # public constructor and configuration snapshot.
            kwargs["model_digest"] = resolved_model_digest
        if self._execution_options.num_threads is not None:
            if not _accepts_keyword(parameters, "num_threads"):
                raise DescriptorConfigError(
                    f"{self.name} does not support execution.num_threads"
                )
            kwargs["num_threads"] = self._execution_options.num_threads
        if self._execution_options.device != "cpu":
            if not _has_explicit_keyword(parameters, "device"):
                raise DescriptorConfigError(
                    f"{self.name} does not support execution.device={self._execution_options.device!r}"
                )
            kwargs["device"] = self._execution_options.device

        snapshot = dict(options)
        snapshot["output"] = self._output_options
        snapshot["execution"] = self._execution_options
        for key, default in getattr(self.kernel_type, "configuration_defaults", {}).items():
            if snapshot.get(key) is None:
                snapshot[key] = default
        self._configuration = DescriptorConfiguration(
            CONFIGURATION_SCHEMA_VERSION,
            self.name,
            _json_safe(snapshot),
        )

        try:
            self._kernel = self.kernel_type(**kwargs)
        except (DescriptorConfigError, DescriptorInputError):
            raise
        except TypeError as exc:
            raise DescriptorConfigError(
                f"invalid {self.name} configuration: {exc}"
            ) from exc
        except ValueError as exc:
            if not self.wrap_value_errors_as_config:
                raise
            raise DescriptorConfigError(
                f"invalid {self.name} configuration: {exc}"
            ) from exc
        self._feature_count: int | None = None
        try:
            resolved_features = int(self._kernel.feature_count)
        except (AttributeError, TypeError, ValueError, RuntimeError):
            resolved_features = 0
        if resolved_features > 0:
            self._feature_count = resolved_features
        self._snapshot_kernel_metadata()

    def _validate_public_call(self, kwargs: Mapping[str, Any]) -> None:
        signature = getattr(type(self), "__signature__", None)
        if signature is not None:
            bind_kwargs = dict(kwargs)
            # ModelBackedAdapter injects the resolved path for its private
            # numeric kernel. That implementation detail is deliberately
            # absent from the public signature and must not be treated as a
            # caller-supplied option.
            if getattr(self, "_resolved_model_path", False):
                bind_kwargs.pop(getattr(self, "model_keyword", "model_path"), None)
            try:
                signature.bind(**bind_kwargs)
            except TypeError as exc:
                raise DescriptorConfigError(
                    f"invalid {self.name} constructor arguments: {exc}"
                ) from exc
        allowed = self.allowed_options
        if allowed is None:
            return
        allowed_names = set(allowed)
        internal_model_name = getattr(self, "model_keyword", "model_path")
        if getattr(self, "_resolved_model_path", False):
            allowed_names.add(internal_model_name)
        unknown = set(kwargs) - allowed_names
        if unknown:
            names = ", ".join(sorted(str(item) for item in unknown))
            raise DescriptorConfigError(
                f"{self.name} received unsupported option(s): {names}"
            )
    # Kept as a narrow hook for model-backed adapters.  It is deliberately
    # not a public constructor and therefore cannot become another API path.
    def _initialize_public(self, options: Mapping[str, Any]) -> None:
        if self.requires_species and options.get("species") is None:
            raise DescriptorConfigError(
                f"{self.name} requires species= at construction time"
            )
        self._initialize(options)

    @property
    def feature_count(self) -> int | None:
        if self._feature_count is not None:
            return self._feature_count
        value = getattr(self._kernel, "feature_count", None)
        if value is None:
            return None
        value = int(value)
        if value > 0:
            self._feature_count = value
            return value
        return None

    def _adapt_result(self, result: Any) -> DescriptorResult:
        adapted = adapt_result(result)
        adapted = _apply_output(adapted, self._output_options)
        self._metadata_snapshot = dict(adapted.metadata)
        return adapted

    def _snapshot_kernel_metadata(self) -> None:
        """Capture diagnostic metadata before a native runtime can disappear."""

        builder = getattr(getattr(self, "_kernel", None), "_metadata", None)
        if not callable(builder):
            return
        try:
            value = builder()
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return
        if isinstance(value, Mapping):
            candidate = dict(_json_safe(value))
            if not self._metadata_snapshot:
                self._metadata_snapshot = self._canonical_metadata(candidate)

    def _canonical_metadata(self, candidate: Mapping[str, Any]) -> dict[str, Any]:
        """Build the same metadata envelope available from a computed result.

        Native metadata is often assembled lazily by ``compute``.  Closing a
        descriptor before its first call must nevertheless leave a useful,
        schema-valid snapshot, so the wrapper supplies the fixed fields here
        and keeps backend-specific values under ``details``.
        """

        raw = dict(candidate)
        descriptor = raw.pop("descriptor", self.name)
        backend = raw.pop("backend", "unknown")
        level = raw.pop("level", None)
        raw.pop("schema_version", None)
        raw.pop("feature_count", None)
        raw.pop("output", None)
        raw.pop("execution", None)
        raw.pop("dtype", None)
        raw.pop("sparse", None)
        raw.pop("device", None)
        raw.pop("num_threads", None)
        model = raw.pop("model", None)
        resource = getattr(self, "model_resource", None)
        if resource is not None:
            model = resource.to_dict()
        if level is None:
            try:
                from ..registry.builtins import builtin_registry

                level = builtin_registry.get(self.name).level
            except (ImportError, KeyError):  # pragma: no cover - direct private use
                level = "unknown"
        details = raw.pop("details", None)
        if details is not None and not isinstance(details, Mapping):
            raise TypeError("metadata details must be a JSON object")
        details_value = dict(details or {})
        details_value.update(raw)
        normalized: dict[str, Any] = {
            "schema_version": 1,
            "descriptor": descriptor,
            "backend": backend,
            "level": level,
            "feature_count": self.feature_count,
            "output": {
                "dtype": self._output_options.dtype,
                "sparse": self._output_options.sparse,
            },
            "execution": {
                "device": self._execution_options.device,
                "num_threads": self._execution_options.num_threads,
            },
        }
        if model is not None:
            normalized["model"] = model
        if details_value:
            normalized["details"] = details_value
        return _json_safe(normalized)

    def _compute_batch(
        self,
        batch: StructureBatch,
        *,
        control: ComputeControl | None = None,
    ) -> DescriptorResult:
        try:
            raw_result = self._kernel.compute(batch, control)
        except NativeCancelledError as exc:
            raise CancelledError("descriptor computation was cancelled") from exc
        except DescriptorInputError:
            raise
        except ValueError as exc:
            raise DescriptorInputError(str(exc)) from exc
        return self._adapt_result(raw_result)

    def close(self) -> None:
        if self.closed:
            return
        self._snapshot_kernel_metadata()
        close = getattr(self._kernel, "close", None)
        if close is not None:
            close()
        super().close()


def _public_signature(
    kernel_type: type[Any],
    base: type[DescriptorAdapter],
    allowed_options: frozenset[str] | None = None,
) -> inspect.Signature:
    """Expose a useful constructor signature without forwarding old names."""

    parameters = list(_constructor_parameters(kernel_type).values())
    if any(parameter.kind is inspect.Parameter.VAR_POSITIONAL for parameter in parameters):
        # A few older kernel subclasses used ``*args`` only to forward the
        # parent constructor. Reconstruct that parent signature so the public class
        # does not advertise an unbounded positional escape hatch.
        own_parameters = [
            parameter
            for parameter in parameters
            if parameter.kind
            not in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}
        ]
        for parent in kernel_type.__mro__[1:]:
            if parent is object:
                break
            parent_parameters = list(_constructor_parameters(parent).values())
            if not any(
                parameter.kind is inspect.Parameter.VAR_POSITIONAL
                for parameter in parent_parameters
            ):
                parent_parameters = [
                    parameter
                    for parameter in parent_parameters
                    if parameter.kind is not inspect.Parameter.VAR_KEYWORD
                ]
                names = {parameter.name for parameter in parent_parameters}
                parameters = [
                    *parent_parameters,
                    *[parameter for parameter in own_parameters if parameter.name not in names],
                    *[
                        parameter
                        for parameter in _constructor_parameters(kernel_type).values()
                        if parameter.kind is inspect.Parameter.VAR_KEYWORD
                    ],
                ]
                break
    model_backed = hasattr(base, "model_keyword")
    visible_options = set(allowed_options or ())
    if model_backed:
        visible_options.discard("model_path")
        visible_options.discard("model_file")
        visible_options.add("model")
    parameters = [
        parameter
        for parameter in parameters
        if parameter.kind
        not in {
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        }
        and parameter.name != "config"
        and parameter.name in visible_options
    ]
    existing_names = {parameter.name for parameter in parameters}
    if model_backed and "model" not in existing_names:
        parameters.insert(
            0,
            inspect.Parameter("model", inspect.Parameter.KEYWORD_ONLY, default=None),
        )
        existing_names.add("model")
    parameters.extend(
        inspect.Parameter(
            name,
            inspect.Parameter.KEYWORD_ONLY,
            default=None,
            annotation=Any,
        )
        for name in sorted(visible_options - existing_names)
    )
    # Common representation and execution controls are wrapper concerns, not
    # kernel arguments, but are part of every public constructor.
    for name in ("output", "execution"):
        if name not in {parameter.name for parameter in parameters}:
            parameters.append(
                inspect.Parameter(name, inspect.Parameter.KEYWORD_ONLY, default=None)
            )
    # Constructors are a keyword-only contract even when the underlying
    # kernel historically accepted positional arguments.
    parameters = [
        parameter.replace(kind=inspect.Parameter.KEYWORD_ONLY)
        for parameter in parameters
    ]
    try:
        return inspect.Signature(parameters)
    except ValueError:  # pragma: no cover - defensive for unusual signatures
        return inspect.Signature()


def adapter_class(
    name: str,
    kernel_type: type[Any],
    module: str,
    *,
    base: type[DescriptorAdapter] = DescriptorAdapter,
    default_model: Any = None,
    allowed_options: frozenset[str] | None = None,
    requires_species: bool | None = None,
) -> type[DescriptorAdapter]:
    """Create a named public adapter without repeating lifecycle boilerplate."""

    resolved_allowed = allowed_options
    if resolved_allowed is None:
        resolved_allowed = _PUBLIC_OPTION_NAMES.get(name)
    if resolved_allowed is None:
        resolved_allowed = frozenset(
            parameter.name
            for parameter in _constructor_parameters(kernel_type).values()
            if parameter.kind
            not in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}
        )

    signature = _public_signature(kernel_type, base, resolved_allowed)
    resolved_requires_species = (
        "species" in resolved_allowed
        if requires_species is None
        else bool(requires_species)
    )

    # Generate a real keyword-only ``__init__`` rather than relying solely on
    # ``__signature__``.  This makes positional calls and unknown options fail
    # at Python's boundary before they can reach a private kernel.
    parameter_names = [parameter.name for parameter in signature.parameters.values()]
    source = "def __init__(self, *, " + ", ".join(parameter_names) + "):\n"
    source += "    self._initialize_public({" + ", ".join(
        f"{name!r}: {name}" for name in parameter_names
    ) + "})\n"
    namespace: dict[str, Any] = {}
    exec(compile(source, f"<{module}.{name}.__init__>", "exec"), namespace)
    public_init = namespace["__init__"]
    public_init.__kwdefaults__ = {
        parameter.name: parameter.default
        for parameter in signature.parameters.values()
        if parameter.default is not inspect.Parameter.empty
    }

    return type(
        name,
        (base,),
        {
            "kernel_type": kernel_type,
            "name": name,
            "default_model": default_model,
            "loader_kind": name if hasattr(base, "model_keyword") else getattr(base, "loader_kind", "generic"),
            "allowed_options": resolved_allowed,
            "requires_species": resolved_requires_species,
            "__module__": module,
            "__doc__": kernel_type.__doc__ or f"{name} descriptor.",
            "__signature__": signature,
            "__init__": public_init,
        },
    )
