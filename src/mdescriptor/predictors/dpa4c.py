"""DPA4C energy and force predictor."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

import numpy as np

from ..core.errors import DescriptorInputError, ModelLoadError
from ..core.options import ExecutionOptions
from ..descriptors.model_backed.graph import _ATOMIC_SYMBOLS
from ..models import DPA4C_RESOURCE
from ._base import _Predictor, _unwrap_native_control

_DEFAULT_EXECUTION = ExecutionOptions()


class DPA4C(_Predictor):
    """Predict DPA4C total/per-atom energies and forces for structure batches."""

    name = "DPA4C"

    def __init__(
        self,
        model: Any = None,
        execution: ExecutionOptions = _DEFAULT_EXECUTION,
    ) -> None:
        from ..descriptors._kernels.dpa4c import Dpa4cKernel

        super().__init__(
            model,
            execution,
            default_model=DPA4C_RESOURCE,
            loader=Dpa4cKernel.load_model_artifact,
        )
        try:
            self._type_map, self._type_numbers, self._predictor_payload = (
                self._build_predictor_payload()
            )
            if execution.device == "cpu":
                from .. import _native

                if not hasattr(_native, "Dpa4cPredictor"):
                    raise ModelLoadError(
                        "native module does not provide Dpa4cPredictor; rebuild MDescriptor"
                    )
                self._backend = _native.Dpa4cPredictor(self._predictor_payload)
        except ModelLoadError:
            self.close()
            raise
        except Exception as exc:
            self.close()
            raise ModelLoadError("failed to load DPA4C model for prediction") from exc

    def _build_predictor_payload(
        self,
    ) -> tuple[tuple[str, ...], np.ndarray, dict[str, Any]]:
        from ..descriptors._kernels.dpa4c import _native_payload
        from ..descriptors._kernels.dpa_common import new_runtime

        loaded_model = self.loaded_model
        if loaded_model is None:
            raise ModelLoadError("DPA4C model session was closed during initialization")
        checkpoint = loaded_model.materialize_weights()
        runtime = new_runtime(self.resolved_model.path, checkpoint)
        descriptor = runtime.descriptor
        payload = _native_payload(
            descriptor,
            calibrate=True,
            num_threads=self.execution.num_threads or 1,
        )
        if payload is None:
            raise ModelLoadError(
                "DPA4C prediction requires an uncompressed, spin-free checkpoint "
                "supported by the native inference backend"
            )
        payload["feature_count"] = int(runtime.dim_out)
        _add_energy_fitting_payload(payload, checkpoint, runtime.type_map)
        type_map = tuple(str(value) for value in runtime.type_map)
        type_numbers = np.asarray([_atomic_number(symbol) for symbol in type_map], dtype=np.int32)
        payload["type_numbers"] = type_numbers
        return type_map, type_numbers, payload

    def _create_backend(self) -> Any:
        from .._runtime import create_cuda_predictor

        cuda_payload = {
            "model": self._predictor_payload,
            "type_numbers": self._type_numbers,
            "feature_count": self._predictor_payload["feature_count"],
        }
        return create_cuda_predictor("DPA4C", {"_cuda_payload": cuda_payload})

    def _metadata_fields(self) -> Mapping[str, Any]:
        return {"type_map": self._type_map}

    def _validate_batch(self, batch: Any) -> None:
        if batch.spins is not None:
            raise DescriptorInputError(
                "DPA4C prediction model does not support spins",
                code="unsupported_input",
                path=["input", "spins"],
            )
        if batch.charge_spin is not None:
            raise DescriptorInputError(
                "DPA4C prediction model does not support charge_spin",
                code="unsupported_input",
                path=["input", "charge_spin"],
            )
        _map_type_indices(batch.numbers, self._type_map)

    def _predict_backend(self, backend: Any, batch: Any, control: Any) -> Any:
        if self.execution.device == "cuda":
            return backend.predict(batch, control)
        type_indices = _map_type_indices(batch.numbers, self._type_map)
        return backend.predict(
            batch,
            type_indices,
            _unwrap_native_control(control),
        )

    def _release_payload(self) -> None:
        if hasattr(self, "_predictor_payload"):
            self._predictor_payload.clear()
        if hasattr(self, "_type_numbers"):
            self._type_numbers = np.empty(0, dtype=np.int32)


def _add_energy_fitting_payload(
    payload: dict[str, Any],
    checkpoint: Mapping[str, Any],
    type_map: tuple[str, ...] | list[str],
) -> None:
    model = checkpoint.get("model")
    if not isinstance(model, Mapping):
        raise ModelLoadError("DPA4C checkpoint is missing its model weights")
    parameters = model.get("_extra_state", {}).get("model_params", {})
    fitting = parameters.get("fitting_net", {}) if isinstance(parameters, Mapping) else {}
    if not isinstance(fitting, Mapping) or fitting.get("type", "ener") != "ener":
        raise ModelLoadError("DPA4C prediction requires an energy fitting network")
    if fitting.get("activation_function", "silu") != "silu":
        raise ModelLoadError("DPA4C prediction currently supports only SiLU fitting networks")
    if any(int(fitting.get(key, 0) or 0) for key in ("numb_fparam", "numb_aparam")):
        raise ModelLoadError("DPA4C prediction does not support fitting parameters")
    if int(fitting.get("dim_case_embd", 0) or 0) != 0:
        raise ModelLoadError("DPA4C prediction does not support case embeddings")

    prefix = _one_prefix(model, ".fitting_net.nets._module_networks.")
    layer_pattern = re.compile(
        re.escape(prefix) + r"nets\._module_networks\.(\d+)\.layers\.(\d+)\.(w|b)$"
    )
    layers: dict[int, dict[str, np.ndarray]] = {}
    networks: set[int] = set()
    for key, value in model.items():
        match = layer_pattern.fullmatch(str(key))
        if match is None:
            continue
        network_index, layer_index, field = (
            int(match.group(1)),
            int(match.group(2)),
            match.group(3),
        )
        networks.add(network_index)
        layers.setdefault(layer_index, {})[field] = np.asarray(value)
    if networks != {0} or not layers or sorted(layers) != list(range(len(layers))):
        raise ModelLoadError("DPA4C prediction requires one contiguous fitting network")

    widths = [int(value) for value in fitting.get("neuron", ())]
    weights: list[np.ndarray] = []
    biases: list[np.ndarray] = []
    previous_width = int(payload["feature_count"])
    for index in range(len(layers)):
        layer = layers[index]
        weight = np.asarray(layer.get("w"), dtype=np.float32)
        bias = np.asarray(layer.get("b"), dtype=np.float32)
        if (
            weight.ndim != 2
            or weight.shape[0] != previous_width
            or bias.shape != (weight.shape[1],)
        ):
            raise ModelLoadError(f"DPA4C fitting layer {index} has invalid shapes")
        if index < len(widths) and weight.shape[1] != widths[index]:
            raise ModelLoadError("DPA4C fitting widths do not match checkpoint weights")
        if index == len(layers) - 1 and weight.shape[1] != 1:
            raise ModelLoadError("DPA4C fitting network must produce one energy per atom")
        weights.append(np.ascontiguousarray(weight).reshape(-1))
        biases.append(np.ascontiguousarray(bias).reshape(-1))
        previous_width = int(weight.shape[1])
    if len(layers) != len(widths) + 1:
        raise ModelLoadError("DPA4C fitting network layer count is unsupported")

    atom_bias = _model_weight(model, ".fitting_net.bias_atom_e")
    output_bias = _model_weight(model, ".out_bias")
    atom_bias = np.asarray(atom_bias, dtype=np.float64).reshape(-1)
    output_bias = np.asarray(output_bias, dtype=np.float64).reshape(-1)
    if atom_bias.shape != (len(type_map),) or output_bias.shape != (len(type_map),):
        raise ModelLoadError("DPA4C atomic energy biases do not match its type map")

    payload.update(
        {
            "fitting_neurons": np.asarray(widths, dtype=np.int32),
            "fitting_weights": np.concatenate(weights).astype(np.float32, copy=False),
            "fitting_biases": np.concatenate(biases).astype(np.float32, copy=False),
            "fitting_activation": "silu",
            "fitting_atom_bias": np.ascontiguousarray(atom_bias),
            "output_bias": np.ascontiguousarray(output_bias),
        }
    )


def _one_prefix(model: Mapping[str, Any], marker: str) -> str:
    prefixes = {
        str(key)[: str(key).index(marker) + len(marker)] for key in model if marker in str(key)
    }
    # The matching layer regex needs the prefix through "fitting_net."; the
    # marker above locates the state entries while this derives that prefix.
    if len(prefixes) != 1:
        raise ModelLoadError("DPA4C prediction requires one fitting model branch")
    prefix = next(iter(prefixes))
    return prefix[: -len("nets._module_networks.")]


def _model_weight(model: Mapping[str, Any], suffix: str) -> np.ndarray:
    matches = [value for key, value in model.items() if str(key).endswith(suffix)]
    if len(matches) != 1:
        raise ModelLoadError(f"DPA4C checkpoint must contain exactly one {suffix} tensor")
    return np.asarray(matches[0])


def _atomic_number(symbol: str) -> int:
    for number, candidate in _ATOMIC_SYMBOLS.items():
        if candidate == symbol:
            return number
    raise ModelLoadError(f"DPA4C type map contains unsupported element {symbol!r}")


def _map_type_indices(numbers: np.ndarray, type_map: tuple[str, ...]) -> np.ndarray:
    by_symbol = {symbol: index for index, symbol in enumerate(type_map)}
    atomic_symbols = {number: symbol for number, symbol in _ATOMIC_SYMBOLS.items()}
    indices: list[int] = []
    for number in numbers:
        atomic_number = int(number)
        symbol = atomic_symbols.get(atomic_number)
        if symbol is None:
            raise DescriptorInputError(
                f"atomic number {atomic_number} has no known element symbol",
                code="unsupported_species",
                path=["input", "numbers"],
            )
        if symbol not in by_symbol:
            raise DescriptorInputError(
                f"element {symbol!r} is absent from the DPA4C type map",
                code="unsupported_species",
                path=["input", "numbers"],
            )
        indices.append(by_symbol[symbol])
    return np.asarray(indices, dtype=np.int32)


__all__ = ["DPA4C"]
