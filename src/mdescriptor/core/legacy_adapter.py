"""Explicit boundary between the public descriptor contract and old kernels."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
import inspect
from typing import Any, ClassVar

import numpy as np

from .descriptor import Descriptor
from .errors import (
    CancelledError,
    DescriptorConfigError,
    DescriptorInputError,
    NativeCancelledError,
)
from .input import StructureBatch
from .options import ExecutionOptions, OutputOptions
from .result import DescriptorResult


# The legacy calculators use ``**kwargs`` for descriptor-specific config
# dictionaries. Keep the accepted public keys in one table so that this
# forwarding seam remains strict without duplicating constructor logic in
# every adapter module.
_PUBLIC_OPTION_NAMES: dict[str, frozenset[str]] = {
    "SOAP": frozenset({
        "species", "config", "rbf", "n_max", "l_max", "sigma", "average",
        "dtype", "sparse", "weighting", "r_cut", "compression", "num_threads",
    }),
    "ACSF": frozenset({
        "species", "config", "r_cut", "dtype", "sparse", "num_threads",
        "g2_params", "G2", "g2", "g3_params", "G3", "g3",
        "g4_params", "G4", "g4", "g5_params", "G5", "g5",
    }),
    "SOAPTurbo": frozenset({
        "species", "config", "alpha_max", "l_max", "rcut_hard", "rcut_soft",
        "nf", "radial_enhancement", "basis", "compression", "compress_mode",
        "dtype", "sparse", "num_threads", "atom_sigma_r", "atom_sigma_r_scaling",
        "atom_sigma_t", "atom_sigma_t_scaling", "amplitude_scaling",
        "central_weight", "central_species",
    }),
    "C00PSMLFF": frozenset({
        "species", "config", "r_cut", "cutoff", "n_radial", "n_max", "l_max",
        "cutoff_function", "include_radial", "include_angular", "normalize_radial",
        "normalize_angular", "super_vector", "radial_weight", "angular_weight",
        "exclude_self_interaction", "num_threads", "dtype", "sparse",
    }),
    "MTP": frozenset({
        "species", "config", "potential_path", "potential", "model", "min_dist",
        "max_dist", "r_cut", "cutoff", "radial_basis_size", "radial_funcs_count",
        "max_rank", "l_max", "max_level", "level", "radial_basis_type", "dtype",
        "sparse", "num_threads",
    }),
    "NEP": frozenset({
        "config", "dtype", "sparse", "num_threads",
    }),
    "DPA4": frozenset({
        "config", "calibrate", "device", "rcut", "channels", "lmax",
        "n_radial", "radial_modes", "n_blocks", "radial_hidden", "basis_type",
        "precision", "use_spin", "add_chg_spin_ebd", "default_chg_spin", "exclude_types",
    }),
    "DPA4C": frozenset({
        "config", "calibrate", "device", "use_amp", "rcut", "channels",
        "lmax", "basis_type", "n_radial", "radial_modes", "precision", "use_spin",
        "add_chg_spin_ebd", "default_chg_spin", "exclude_types",
    }),
    "SNAP": frozenset({"weights", "lmax", "rcut", "normalize_U"}),
    "LBispectrum": frozenset({
        "twojmax", "diagonal", "rfac0", "rmin0", "rcutfac", "element_profile",
        "element_radii", "weights", "rcut", "normalize_U",
    }),
    "LodeSphericalExpansion": frozenset({
        "species", "cutoff", "density_width", "max_radial", "max_angular", "num_threads",
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
    )


def _options_from_mapping(value: Mapping[str, Any], option_type: type[Any], name: str) -> Any:
    allowed = set(option_type.__dataclass_fields__)
    unknown = set(value) - allowed
    if unknown:
        names = ", ".join(sorted(str(item) for item in unknown))
        raise DescriptorConfigError(f"{name} contains unsupported option(s): {names}")
    try:
        return option_type(**dict(value))
    except (TypeError, ValueError) as exc:
        raise DescriptorConfigError(f"invalid {name} options: {exc}") from exc


def _coerce_options(value: Any, option_type: type[Any], name: str) -> Any:
    if isinstance(value, option_type):
        return value
    if isinstance(value, Mapping):
        return _options_from_mapping(value, option_type, name)
    raise DescriptorConfigError(f"{name} must be {option_type.__name__} or a mapping")


def _constructor_parameters(calculator: type[Any]) -> dict[str, inspect.Parameter]:
    try:
        signature = inspect.signature(calculator.__init__)
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

    dtype = np.dtype(options.dtype)
    values = result.values
    if hasattr(values, "astype"):
        values = values.astype(dtype, copy=False)
    else:
        values = np.asarray(values, dtype=dtype)

    if options.sparse:
        if not hasattr(values, "todense"):
            try:
                import sparse as sparse_module
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise ImportError(
                    "output.sparse=True requires the optional 'sparse' package"
                ) from exc
            values = sparse_module.COO.from_numpy(np.asarray(values))
    elif hasattr(values, "todense"):
        values = np.asarray(values.todense(), dtype=dtype)
    else:
        values = np.asarray(values, dtype=dtype)

    metadata = dict(result.metadata)
    metadata.update({"dtype": options.dtype, "sparse": options.sparse})
    return replace(result, values=values, metadata=metadata)


class LegacyDescriptorAdapter(Descriptor):
    """Centralize lifecycle, options, and result handling for one kernel.

    The wrapped implementation is deliberately private. Public callers get
    only :class:`Descriptor`'s lifecycle and ``compute`` contract plus the
    explicitly declared ``feature_count`` property.
    """

    legacy_type: ClassVar[type[Any]]
    name: ClassVar[str]
    wrap_value_errors_as_config: ClassVar[bool] = True
    allowed_options: ClassVar[frozenset[str] | None] = None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        output_value = kwargs.pop("output", None)
        execution_value = kwargs.pop("execution", None)
        self._output_options: OutputOptions | None = None
        self._execution_options: ExecutionOptions | None = None
        self._validate_public_call(args, kwargs)

        if output_value is not None:
            if "dtype" in kwargs or "sparse" in kwargs:
                raise DescriptorConfigError(
                    "pass dtype/sparse either through output or as direct descriptor options, not both"
                )
            self._output_options = _coerce_options(output_value, OutputOptions, "output")
        if execution_value is not None:
            if "device" in kwargs or "num_threads" in kwargs:
                raise DescriptorConfigError(
                    "pass device/num_threads either through execution or as direct descriptor options, not both"
                )
            self._execution_options = _coerce_options(
                execution_value, ExecutionOptions, "execution"
            )
            parameters = _constructor_parameters(self.legacy_type)
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

        try:
            self._legacy = self.legacy_type(*args, **kwargs)
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

    def _validate_public_call(self, args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> None:
        signature = getattr(type(self), "__signature__", None)
        if signature is not None:
            bind_kwargs = dict(kwargs)
            # ModelBackedAdapter injects the resolved path for its private
            # legacy calculator.  That implementation detail is deliberately
            # absent from the public signature and must not be treated as a
            # caller-supplied option.
            if getattr(self, "_resolved_model_path", False):
                bind_kwargs.pop(getattr(self, "model_keyword", "model_path"), None)
            try:
                signature.bind(*args, **bind_kwargs)
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
        config = kwargs.get("config")
        if config is not None:
            if not isinstance(config, Mapping):
                raise DescriptorConfigError(f"{self.name} config must be a mapping")
            nested_allowed_names = allowed_names - {internal_model_name}
            nested_unknown = set(config) - nested_allowed_names
            if nested_unknown:
                names = ", ".join(sorted(str(item) for item in nested_unknown))
                raise DescriptorConfigError(
                    f"{self.name} config contains unsupported option(s): {names}"
                )

    @property
    def feature_count(self) -> int | None:
        if self._feature_count is not None:
            return self._feature_count
        value = getattr(self._legacy, "feature_count", None)
        if value is None:
            return None
        value = int(value)
        if value > 0:
            self._feature_count = value
            return value
        return None

    def _adapt_result(self, result: Any) -> DescriptorResult:
        adapted = adapt_result(result)
        if self._output_options is not None:
            adapted = _apply_output(adapted, self._output_options)
        return adapted

    def _compute_batch(self, batch: StructureBatch, *, control: Any = None) -> DescriptorResult:
        try:
            raw_result = self._legacy.compute(batch, control)
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
        close = getattr(self._legacy, "close", None)
        if close is not None:
            close()
        super().close()


def _public_signature(
    legacy_type: type[Any],
    base: type[LegacyDescriptorAdapter],
    allowed_options: frozenset[str] | None = None,
) -> inspect.Signature:
    """Expose a useful constructor signature without forwarding old names."""

    parameters = list(_constructor_parameters(legacy_type).values())
    if any(parameter.kind is inspect.Parameter.VAR_POSITIONAL for parameter in parameters):
        # A few legacy subclasses used ``*args`` only to forward the parent
        # constructor. Reconstruct that parent signature so the public class
        # does not advertise an unbounded positional escape hatch.
        own_parameters = [
            parameter
            for parameter in parameters
            if parameter.kind
            not in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}
        ]
        for parent in legacy_type.__mro__[1:]:
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
                        for parameter in _constructor_parameters(legacy_type).values()
                        if parameter.kind is inspect.Parameter.VAR_KEYWORD
                    ],
                ]
                break
    if hasattr(base, "model_keyword"):
        parameters = [
            inspect.Parameter("model", inspect.Parameter.POSITIONAL_OR_KEYWORD, default=None),
            *[
                parameter
                for parameter in parameters
                if parameter.name not in {"model_path", "model_file", "model"}
            ],
        ]
    # The forwarding implementation uses ``**kwargs`` internally, but the
    # public boundary must enumerate every accepted option.  Add the explicit
    # descriptor options that are represented by the central allow-list and
    # omit legacy-only model path names.
    visible_options = set(allowed_options or ())
    if hasattr(base, "model_keyword"):
        visible_options.difference_update({"model_path", "model_file"})
    existing_names = {parameter.name for parameter in parameters}
    parameters = [
        parameter
        for parameter in parameters
        if parameter.kind is not inspect.Parameter.VAR_KEYWORD
    ]
    existing_names = {parameter.name for parameter in parameters}
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
    # legacy calculator arguments, but are part of every public constructor.
    for name in ("output", "execution"):
        if name not in {parameter.name for parameter in parameters}:
            parameters.append(
                inspect.Parameter(name, inspect.Parameter.KEYWORD_ONLY, default=None)
            )
    try:
        return inspect.Signature(parameters)
    except ValueError:  # pragma: no cover - defensive for unusual signatures
        return inspect.Signature()


def adapter_class(
    name: str,
    legacy_type: type[Any],
    module: str,
    *,
    base: type[LegacyDescriptorAdapter] = LegacyDescriptorAdapter,
    default_model: Any = None,
    allowed_options: frozenset[str] | None = None,
) -> type[LegacyDescriptorAdapter]:
    """Create a named public adapter without repeating lifecycle boilerplate."""

    resolved_allowed = allowed_options
    if resolved_allowed is None:
        resolved_allowed = _PUBLIC_OPTION_NAMES.get(name)
    if resolved_allowed is None:
        resolved_allowed = frozenset(
            parameter.name
            for parameter in _constructor_parameters(legacy_type).values()
            if parameter.kind
            not in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}
        )

    return type(
        name,
        (base,),
        {
            "legacy_type": legacy_type,
            "name": name,
            "default_model": default_model,
            "allowed_options": resolved_allowed,
            "__module__": module,
            "__doc__": legacy_type.__doc__ or f"{name} descriptor.",
            "__signature__": _public_signature(legacy_type, base, resolved_allowed),
        },
    )
