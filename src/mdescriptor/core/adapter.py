"""Explicit boundary between public descriptors and private numeric kernels."""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from copy import copy
from os import PathLike, fspath
from typing import Any, ClassVar, cast

from ..registry.info import (
    LEGACY_PARAMETER_ALIASES,
    _validate_parameter_value,
    validate_descriptor_parameters,
)
from .backends import BackendKernel, CpuBackend, CudaBackend
from .control import ComputeControl
from .descriptor import Descriptor, _input_error_path
from .errors import (
    DescriptorConfigError,
    DescriptorInputError,
    UnsupportedPeriodicityError,
)
from .input import StructureBatch
from .json_value import JSON_UNHANDLED, json_safe_value
from .options import (
    CONFIGURATION_SCHEMA_VERSION,
    DescriptorConfiguration,
    ExecutionOptions,
    OutputOptions,
)
from .result import DescriptorResult, _json_safe, _metadata_v1, format_values

# The registry is the canonical declaration of constructor options.  The
# aliases below are only a Python compatibility layer; they are not exposed
# as GUI schema fields or emitted in canonical configurations.  Kernels remain
# private implementation details, so their signatures are never used as the
# public option source for built-ins.
_IMPLEMENTATION_ONLY_OPTIONS = frozenset(
    {
        "config",
        "dtype",
        "sparse",
        "device",
        "num_threads",
        "model_path",
        "model_file",
        "model_digest",
        "_checkpoint",
    }
)


def adapt_result(result: Any) -> DescriptorResult:
    """Convert a kernel result into the one public result schema."""

    if not isinstance(result, DescriptorResult):
        raise TypeError(
            f"kernel compute must return DescriptorResult, got {type(result).__name__}"
        )
    return result


def _coerce_options(value: Any, option_type: type[Any], name: str) -> Any:
    if isinstance(value, option_type):
        return value
    raise DescriptorConfigError(
        f"{name} must be {option_type.__name__}",
        code="invalid_option_type",
        path=[name],
    )


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


def _builtin_info_field(name: str, attr: str) -> Mapping[str, Any] | None:
    """Read one static info field from the registry without importing kernels."""

    try:
        # Import lazily: the registry must not import descriptor implementations
        # while it is being assembled, and static metadata must stay lightweight.
        from ..registry.builtins import builtin_registry

        spec = builtin_registry.get(name)
    except (ImportError, KeyError):
        return None
    if spec.info is None:
        return None
    return getattr(spec.info, attr)


def _builtin_parameter_names(name: str) -> frozenset[str] | None:
    """Read canonical constructor names without importing a descriptor class."""

    parameters = _builtin_info_field(name, "parameters")
    if parameters is None:
        return None
    return frozenset(str(option) for option in parameters)


def _unsupported_input(
    descriptor: str,
    field: str,
    value: Any,
    supported: Any,
) -> DescriptorInputError:
    return _input_capability_error(
        descriptor,
        field=field,
        message=f"{descriptor} does not support input field {field!r}",
        code="unsupported_input",
        value=value,
        supported=supported,
    )


def _input_capability_error(
    descriptor: str,
    *,
    field: str,
    message: str,
    code: str,
    value: Any,
    supported: Any = None,
    error_type: type[DescriptorInputError] = DescriptorInputError,
) -> DescriptorInputError:
    details: dict[str, Any] = {"provided": value}
    if supported is not None:
        details["supported"] = list(supported)
    return error_type(
        message,
        code=code,
        path=["input", field],
        details=details,
    )


def _unsupported_periodicity(
    descriptor: str,
    provided: str,
    supported: tuple[str, ...],
) -> UnsupportedPeriodicityError:
    if supported == ("fully_periodic",):
        message = f"{descriptor} requires fully_periodic input"
    else:
        supported_text = ", ".join(supported) or "none"
        message = (
            f"{descriptor} does not support periodicity {provided!r}; "
            f"supported: {supported_text}"
        )
    return cast(
        UnsupportedPeriodicityError,
        _input_capability_error(
            descriptor,
            field="periodicity",
            message=message,
            code="unsupported_periodicity",
            value=provided,
            supported=supported,
            error_type=UnsupportedPeriodicityError,
        ),
    )


def _legacy_parameter_names(name: str) -> frozenset[str]:
    return frozenset(LEGACY_PARAMETER_ALIASES.get(name, {}))


def _schema_value_converter(value: Any) -> Any:
    """Normalize live objects that a direct-Python parameter may carry."""

    if isinstance(value, PathLike):
        raw = fspath(value)
        return raw.decode() if isinstance(raw, bytes) else str(raw)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    to_list = getattr(value, "tolist", None)
    if callable(to_list):
        return to_list()
    return JSON_UNHANDLED


def _schema_value(value: Any) -> Any:
    """Convert common direct-Python values to the JSON schema value shape.

    Values without a JSON representation (non-finite floats, live objects)
    pass through unchanged; the schema ladders and the kernels own them.
    """

    try:
        return json_safe_value(
            value,
            context="descriptor parameter",
            scalar_keys=True,
            converter=_schema_value_converter,
        )
    except TypeError:
        return value


def _configuration_validation_path(
    descriptor: str, options: Mapping[str, Any]
) -> list[str | int] | None:
    """Return the typed path of the first schema-invalid option, if any."""

    schemas = _builtin_info_field(descriptor, "parameters")
    if schemas is None:
        return None
    try:
        validate_descriptor_parameters(descriptor, _schema_value(options), schemas)
    except DescriptorConfigError as exc:
        path = list(exc.path or ())
        if path and path[0] == "parameters":
            path.pop(0)
        return path or None
    return None


def _canonicalize_legacy_options(options: Mapping[str, Any], name: str) -> dict[str, Any]:
    """Translate direct-Python aliases before taking a config snapshot."""

    canonical = dict(options)
    for alias, target in LEGACY_PARAMETER_ALIASES.get(name, {}).items():
        alias_value = canonical.get(alias)
        target_value = canonical.get(target)
        if alias_value is not None and target_value is not None:
            raise DescriptorConfigError(
                f"{name} received both {alias}= and {target}=",
                code="conflicting_options",
                path=["parameters", alias],
                details={"canonical": target},
            )
        if alias_value is not None:
            canonical[target] = alias_value
        canonical.pop(alias, None)
    return canonical


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
    output = {"dtype": options.dtype, "sparse": options.sparse}
    if values is result.values and result.metadata.get("output") == output:
        return result
    return result._replace_output(values, output)


class DescriptorAdapter(Descriptor):
    """Centralize lifecycle, options, and result handling for one kernel.

    The wrapped implementation is deliberately private. Public callers get
    only :class:`Descriptor`'s lifecycle and ``compute`` contract plus the
    explicitly declared ``feature_count`` property.
    """

    kernel_type: ClassVar[type[Any]]
    name: ClassVar[str]
    _kernel: Any
    _backend: BackendKernel
    wrap_value_errors_as_config: ClassVar[bool] = True
    allowed_options: ClassVar[frozenset[str] | None] = None
    requires_species: ClassVar[bool] = False
    supported_devices: ClassVar[tuple[str, ...]] = ("cpu",)
    input_capabilities: ClassVar[Mapping[str, Any] | None] = None

    def _bind_input_capabilities(self, capabilities: Mapping[str, Any]) -> None:
        """Bind input metadata supplied by a non-built-in registry."""

        self._registry_input_capabilities = capabilities

    def _bind_execution_capabilities(self, capabilities: Mapping[str, Any]) -> None:
        """Bind execution metadata supplied by a non-built-in registry."""

        self._registry_execution_capabilities = capabilities

    def _initialize(
        self,
        options: Mapping[str, Any],
        *,
        validate_public: bool = True,
    ) -> None:
        """Initialize a kernel from already-bound public options."""

        super().__init__()
        options = _canonicalize_legacy_options(options, self.name)
        if validate_public:
            self._validate_public_parameters(options)
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
        execution_capabilities = getattr(
            self, "_registry_execution_capabilities", None
        )
        if execution_capabilities is None:
            execution_capabilities = _builtin_info_field(self.name, "execution")
        if execution_capabilities is not None:
            declared_devices = execution_capabilities.get("devices", ("cpu",))
        else:
            declared_devices = self.supported_devices
        supported_devices = tuple(str(value) for value in declared_devices)
        if not supported_devices:
            raise DescriptorConfigError(
                f"{self.name} does not declare an execution device",
                code="invalid_descriptor_info",
                path=["execution", "devices"],
            )
        if self._execution_options.device not in supported_devices:
            supported = ", ".join(supported_devices) or "none"
            raise DescriptorConfigError(
                f"{self.name} does not support execution.device={self._execution_options.device!r}; "
                f"supported devices: {supported}",
                code="unsupported_device",
                path=["execution", "device"],
                details={"supported": list(supported_devices)},
            )
        if self._output_options.sparse:
            try:
                import scipy.sparse  # noqa: F401
            except ImportError as exc:  # pragma: no cover - optional extra
                raise DescriptorConfigError(
                    "output.sparse requires the optional 'sparse' extra",
                    code="missing_optional_dependency",
                    path=["output", "sparse"],
                ) from exc
        self._validate_public_call(kwargs)
        # Kernel implementations may still raise a plain ``TypeError`` or
        # ``ValueError`` for a schema-invalid option.  Recover the path from
        # the registry's typed schema before adding private implementation
        # arguments below, so those arguments cannot appear in public errors.
        configuration_path = _configuration_validation_path(self.name, kwargs)
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
        if (
            self._execution_options.num_threads is not None
            and self._execution_options.device == "cpu"
        ):
            if not _accepts_keyword(parameters, "num_threads"):
                raise DescriptorConfigError(
                    f"{self.name} does not support execution.num_threads",
                    code="unsupported_option",
                    path=["execution", "num_threads"],
                )
            kwargs["num_threads"] = self._execution_options.num_threads
        try:
            validation_kernel = self.kernel_type(**kwargs)
        except (DescriptorConfigError, DescriptorInputError):
            raise
        except TypeError as exc:
            raise DescriptorConfigError(
                f"invalid {self.name} configuration: {exc}",
                code="invalid_configuration",
                path=getattr(exc, "path", None)
                or configuration_path,
            ) from exc
        except ValueError as exc:
            if not self.wrap_value_errors_as_config:
                raise
            raise DescriptorConfigError(
                f"invalid {self.name} configuration: {exc}",
                code="invalid_configuration",
                path=getattr(exc, "path", None)
                or configuration_path,
            ) from exc

        snapshot = dict(options)
        canonicalize = getattr(validation_kernel, "_canonical_configuration", None)
        if callable(canonicalize):
            snapshot = dict(canonicalize(snapshot))
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

        backend: BackendKernel
        if self._execution_options.device == "cuda":
            # ``validation_kernel`` only validates and canonicalizes options;
            # it is never used for CUDA computation.  In particular, no CUDA
            # module or driver is touched on this construction path.
            resolved_features: int | None
            try:
                candidate = int(validation_kernel.feature_count)
            except (AttributeError, TypeError, ValueError, RuntimeError):
                candidate = 0
            resolved_features = candidate if candidate > 0 else None
            backend_options = dict(snapshot)
            backend_options["output"] = self._output_options
            backend_options["execution"] = self._execution_options
            # Keep the model tensor payload outside the public configuration
            # snapshot and result metadata.  The builder runs while the CPU
            # validation kernel is alive; plugin loading remains lazy until
            # the first compute call.
            cuda_payload_builder = getattr(validation_kernel, "_cuda_payload", None)
            try:
                if callable(cuda_payload_builder):
                    try:
                        cuda_payload = cuda_payload_builder()
                    except (
                        AttributeError,
                        KeyError,
                        TypeError,
                        ValueError,
                        RuntimeError,
                    ) as exc:
                        raise DescriptorConfigError(
                            f"failed to prepare {self.name} CUDA payload: {exc}",
                            code="invalid_configuration",
                            path=configuration_path,
                        ) from exc
                    # JSON-safe conversion applies only to the public options.
                    # Device model arrays must remain NumPy arrays for the
                    # private CUDA binding, and are deliberately never put in
                    # ``snapshot``.
                    backend_options = _json_safe(backend_options)
                    backend_options["_cuda_payload"] = cuda_payload
                else:
                    backend_options = _json_safe(backend_options)
                # CUDA result assembly lives outside the validation kernel, but
                # labels are part of the public result contract.  Preserve the
                # validated kernel's canonical ordering in a private option so
                # a CUDA implementation never has to recreate Python label
                # formatting (which is especially important for compressed,
                # high-order, and model-derived descriptors).
                labels_builder = getattr(validation_kernel, "_labels", None)
                if callable(labels_builder):
                    try:
                        backend_options["_cuda_labels"] = tuple(labels_builder())
                    except (AttributeError, KeyError, TypeError, ValueError, RuntimeError) as exc:
                        raise DescriptorConfigError(
                            f"failed to prepare {self.name} CUDA labels: {exc}",
                            code="invalid_configuration",
                            path=configuration_path,
                        ) from exc
                backend_options["_cuda_feature_count"] = resolved_features
                backend = CudaBackend(
                    self.name,
                    backend_options,
                    resolved_features,
                )
                self._kernel = backend
            finally:
                close_validation = getattr(validation_kernel, "close", None)
                if callable(close_validation):
                    close_validation()
        else:
            self._kernel = validation_kernel
            backend = CpuBackend(validation_kernel)
        self._backend = backend

        self._feature_count: int | None = None
        backend_features = self._backend.feature_count
        resolved_features = 0 if backend_features is None else int(backend_features)
        if resolved_features > 0:
            self._feature_count = resolved_features
        try:
            self._snapshot_kernel_metadata()
        except (TypeError, ValueError) as exc:
            raise DescriptorConfigError(
                f"invalid {self.name} configuration: {exc}",
                code="invalid_configuration",
                path=getattr(exc, "path", None)
                or configuration_path,
            ) from exc

    def _validate_public_parameters(self, options: Mapping[str, Any]) -> None:
        """Reject direct values that the kernel would silently coerce."""

        schemas = _builtin_info_field(self.name, "parameters")
        if schemas is None:
            return
        parameters = {
            key: value
            for key, value in _canonicalize_legacy_options(options, self.name).items()
            if key not in {"output", "execution"}
        }
        for name, schema in schemas.items():
            if name not in parameters or parameters[name] is None:
                continue
            _validate_parameter_value(
                _schema_value(parameters[name]),
                schema,
                ["parameters", name],
                direct=True,
            )

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
                    f"invalid {self.name} constructor arguments: {exc}",
                    code="invalid_configuration",
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
                f"{self.name} received unsupported option(s): {names}",
                code="unknown_option",
                path=["parameters"],
            )
    # Kept as a narrow hook for model-backed adapters.  It is deliberately
    # not a public constructor and therefore cannot become another API path.
    def _initialize_public(self, options: Mapping[str, Any]) -> None:
        if self.requires_species and options.get("species") is None:
            raise DescriptorConfigError(
                f"{self.name} requires species= at construction time",
                code="missing_required_parameter",
                path=["species"],
            )
        self._initialize(options)

    def __init__(self, **kwargs: Any) -> None:
        # Public constructors are a keyword-only contract (see the per-class
        # ``__signature__``).  Binding here keeps unknown and positional
        # arguments failing at Python's boundary, before any kernel state
        # exists, while one shared implementation serves every adapter.
        signature = getattr(type(self), "__signature__", None)
        if signature is not None:
            signature.bind(**kwargs)
        self._initialize_public(kwargs)

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

    def _validate_batch(self, batch: StructureBatch) -> None:
        """Enforce the registry-declared input capability before kernel entry."""

        capabilities = getattr(self, "_registry_input_capabilities", None)
        if capabilities is None:
            capabilities = self.input_capabilities
        if capabilities is None:
            # Compute-only descriptors do not have an input capability schema.
            return

        periodicity = tuple(
            str(value) for value in capabilities.get("periodicity", ())
        )
        for flags in batch.pbc:
            if all(bool(flag) for flag in flags):
                kind = "fully_periodic"
            elif not any(bool(flag) for flag in flags):
                kind = "isolated"
            else:  # StructureBatch normally rejects this during construction.
                kind = "mixed"
            if kind not in periodicity:
                raise _unsupported_periodicity(self.name, kind, periodicity)

        if batch.spins is not None and not bool(capabilities.get("spin", False)):
            raise _unsupported_input(self.name, "spins", True, None)
        if batch.charge_spin is not None and not bool(
            capabilities.get("charge_spin", False)
        ):
            raise _unsupported_input(self.name, "charge_spin", True, None)

    def _adapt_result(self, result: Any) -> DescriptorResult:
        adapted = adapt_result(result)
        adapted = self._apply_execution_metadata(adapted)
        adapted = self._add_model_identity(adapted)
        adapted = _apply_output(adapted, self._output_options)
        # Some CUDA layouts (notably batch-derived matrix padding) resolve
        # their width on the first batch and may widen on a later batch.
        self._feature_count = adapted.feature_count
        self._metadata_snapshot = dict(adapted.metadata)
        return adapted

    def _apply_execution_metadata(self, result: DescriptorResult) -> DescriptorResult:
        """Record the selected device independently of backend labels."""

        metadata = dict(result.metadata)
        metadata["execution"] = {
            "device": self._execution_options.device,
            "num_threads": self._execution_options.num_threads,
        }
        updated = copy(result)
        object.__setattr__(updated, "metadata", _json_safe(metadata))
        return updated

    def _add_model_identity(self, result: DescriptorResult) -> DescriptorResult:
        """Attach resolved model identity without changing the result schema."""

        resolved = getattr(self, "resolved_model", None)
        if resolved is None:
            return result
        model = result.metadata.get("model")
        if not isinstance(model, Mapping):
            resource = getattr(self, "model_resource", None)
            model = resource.to_dict() if resource is not None else {}
        model = dict(model)
        model["resolved"] = {
            "digest": resolved.digest,
            "source": resolved.source,
        }
        metadata = dict(result.metadata)
        metadata["model"] = model
        updated = copy(result)
        object.__setattr__(updated, "metadata", _json_safe(metadata))
        return updated

    def _snapshot_kernel_metadata(self) -> None:
        """Capture diagnostic metadata before a native runtime can disappear."""

        backend = getattr(self, "_backend", None)
        builder = getattr(backend, "metadata", None)
        if not callable(builder):
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
        and keeps backend-specific values under ``details``.  The envelope
        shape and validation stay with the shared result normalizer.
        """

        raw = dict(candidate)
        # The wrapper owns the representation fields; discard whatever the
        # backend reported so the resolved options below win.
        level = raw.pop("level", None)
        for name in (
            "schema_version",
            "output",
            "execution",
            "dtype",
            "sparse",
            "device",
            "num_threads",
        ):
            raw.pop(name, None)
        if level is None:
            try:
                from ..registry.builtins import builtin_registry

                level = builtin_registry.get(self.name).level
            except (ImportError, KeyError):  # pragma: no cover - direct private use
                level = "unknown"
        model = raw.pop("model", None)
        resource = getattr(self, "model_resource", None)
        if resource is not None:
            model = resource.to_dict()
        resolved = getattr(self, "resolved_model", None)
        if resolved is not None:
            if not isinstance(model, Mapping):
                model = {}
            model = dict(model)
            model["resolved"] = {
                "digest": resolved.digest,
                "source": resolved.source,
            }
        if model is not None:
            raw["model"] = model
        raw["output"] = {
            "dtype": self._output_options.dtype,
            "sparse": self._output_options.sparse,
        }
        raw["execution"] = {
            "device": self._execution_options.device,
            "num_threads": self._execution_options.num_threads,
        }
        return _metadata_v1(raw, level, self.feature_count)

    def _compute_batch(
        self,
        batch: StructureBatch,
        *,
        control: ComputeControl | None = None,
    ) -> DescriptorResult:
        try:
            raw_result = self._backend.compute(batch, control)
        except DescriptorInputError:
            raise
        except ValueError as exc:
            raise DescriptorInputError(
                str(exc), path=_input_error_path(str(exc))
            ) from exc
        return self._adapt_result(raw_result)

    def close(self) -> None:
        if self.closed:
            return
        self._snapshot_kernel_metadata()
        close = getattr(self._backend, "close", None)
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
    supported_devices: tuple[str, ...] | None = None,
    input_capabilities: Mapping[str, Any] | None = None,
) -> type[DescriptorAdapter]:
    """Create a named public adapter without repeating lifecycle boilerplate."""

    resolved_allowed = allowed_options
    if resolved_allowed is None:
        resolved_allowed = _builtin_parameter_names(name)
    if resolved_allowed is None:
        resolved_allowed = frozenset(
            parameter.name
            for parameter in _constructor_parameters(kernel_type).values()
            if parameter.kind
                not in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}
            and parameter.name not in _IMPLEMENTATION_ONLY_OPTIONS
        )
    resolved_allowed = frozenset(resolved_allowed) | _legacy_parameter_names(name)
    if hasattr(base, "model_keyword"):
        # Model-backed adapters expose one stable ``model=`` entry even if a
        # custom registry entry predates the model schema field.
        resolved_allowed = resolved_allowed | {"model"}

    signature = _public_signature(kernel_type, base, resolved_allowed)
    resolved_requires_species = (
        "species" in resolved_allowed
        if requires_species is None
        else bool(requires_species)
    )
    declared_execution = _builtin_info_field(name, "execution")
    declared_devices = (
        declared_execution.get("devices", ("cpu",))
        if declared_execution is not None
        else None
    )
    resolved_supported_devices = tuple(
        str(item)
        for item in (
            supported_devices
            if supported_devices is not None
            else (declared_devices if declared_devices is not None else ("cpu",))
        )
    )
    if not resolved_supported_devices:
        raise ValueError("supported_devices must contain at least one device")
    resolved_input_capabilities = (
        input_capabilities
        if input_capabilities is not None
        else _builtin_info_field(name, "input")
    )

    # ``__signature__`` plus the shared keyword-only ``DescriptorAdapter``
    # ``__init__`` make positional calls and unknown options fail at Python's
    # boundary before they can reach a private kernel.
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
            "supported_devices": resolved_supported_devices,
            "input_capabilities": resolved_input_capabilities,
            "__module__": module,
            "__doc__": kernel_type.__doc__ or f"{name} descriptor.",
            "__signature__": signature,
        },
    )
