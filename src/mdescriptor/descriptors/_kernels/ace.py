"""Atomic Cluster Expansion (ACE1-compatible) descriptor adapter.

The numerical implementation lives in the C++17 extension.  This module owns
the small, JSON-safe option language exposed by the public ``ACE`` class and
converts it to the native ``AceOptions`` ABI.  The option names intentionally
follow the high-level ``ACE1.Utils.rpi_basis`` constructor so configurations
can be recorded and reconstructed without a Julia runtime.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Mapping
from numbers import Integral, Real
from typing import Any

import numpy as np

from ...core.adapter import DescriptorAdapter
from ...core.errors import DescriptorConfigError
from ...core.result import DescriptorResult
from ...core.species import validate_batch_species
from .core import StructureBatch, _as_batch, _cpp

_PERIODIC_SYMBOLS = (
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn "
    "Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La "
    "Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po "
    "At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg "
    "Cn Nh Fl Mc Lv Ts Og"
).split()
_SYMBOL_TO_NUMBER = {symbol: index for index, symbol in enumerate(_PERIODIC_SYMBOLS, 1)}


def _ace_path(name: str) -> list[str] | None:
    label = name.removeprefix("ACE ").strip()
    if not label:
        return None
    parts: list[str] = []
    for component in label.split("."):
        if "[" in component and component.endswith("]"):
            field, index = component[:-1].split("[", 1)
            parts.extend((field, index))
        else:
            parts.append(component)
    return parts


def _number(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise DescriptorConfigError(f"{name} must be an integer", path=_ace_path(name))
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value) or float(value) != int(value):
            raise DescriptorConfigError(f"{name} must be an integer", path=_ace_path(name))
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise DescriptorConfigError(f"{name} must be an integer", path=_ace_path(name)) from exc
    return result


def _positive_float(value: Any, name: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise DescriptorConfigError(f"{name} must be a finite number", path=_ace_path(name))
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise DescriptorConfigError(f"{name} must be a finite number", path=_ace_path(name)) from exc
    if not np.isfinite(result) or (result < 0.0 if nonnegative else result <= 0.0):
        qualifier = "non-negative" if nonnegative else "positive"
        raise DescriptorConfigError(
            f"{name} must be finite and {qualifier}", path=_ace_path(name)
        )
    return result


def normalize_ace_species(species: Iterable[Any] | Any | None) -> tuple[int, ...] | None:
    """Normalize atomic numbers and chemical symbols without requiring ASE."""

    if species is None:
        return None
    if isinstance(species, (str, bytes)) or isinstance(species, (Integral, Real)):
        raw_values = (species,)
    else:
        try:
            raw_values = tuple(species)
        except TypeError as exc:
            raise DescriptorConfigError(
                "ACE species must be a sequence of atomic numbers or symbols",
                path=["species"],
            ) from exc
    normalized: list[int] = []
    for value in raw_values:
        if isinstance(value, (bool, np.bool_)):
            raise DescriptorConfigError(
                "ACE species must contain atomic numbers or symbols",
                path=["species"],
            )
        if isinstance(value, str):
            symbol = value.strip()
            try:
                number = _SYMBOL_TO_NUMBER[symbol]
            except KeyError as exc:
                raise DescriptorConfigError(
                    f"unknown ACE chemical symbol: {value!r}",
                    path=["species"],
                ) from exc
        else:
            number = _number(value, "ACE species")
        if number <= 0:
            raise DescriptorConfigError(
                "ACE species must contain positive atomic numbers",
                path=["species"],
            )
        normalized.append(number)
    if not normalized:
        raise DescriptorConfigError(
            "ACE species must be a non-empty sequence", path=["species"]
        )
    if len(set(normalized)) != len(normalized):
        raise DescriptorConfigError(
            "ACE species must contain unique entries", path=["species"]
        )
    return tuple(normalized)


def _scalar_or_vector(value: Any, name: str, length: int) -> float | list[float]:
    if isinstance(value, (str, bytes, Mapping)):
        raw_values: tuple[Any, ...] | None = None
    else:
        try:
            raw_values = tuple(value)
        except TypeError:
            raw_values = None
    if raw_values is None:
        return _positive_float(value, name)
    if len(raw_values) != length:
        raise DescriptorConfigError(
            f"{name} must have exactly N={length} entries", path=_ace_path(name)
        )
    return [_positive_float(item, f"{name}[{index}]") for index, item in enumerate(raw_values)]


def _normalize_transform(value: Any, r0: float) -> dict[str, float | str]:
    if value is None:
        return {"type": "PolyTransform", "p": 2.0, "r0": r0, "a": 1.0}
    if not isinstance(value, Mapping):
        raise DescriptorConfigError("ACE trans must be a JSON object", path=["trans"])
    unknown = set(value) - {"type", "p", "r0", "a"}
    if unknown:
        names = ", ".join(sorted(str(item) for item in unknown))
        first = sorted(str(item) for item in unknown)[0]
        raise DescriptorConfigError(
            f"ACE trans has unsupported field(s): {names}",
            path=["trans", first],
        )
    transform_type = value.get("type", "PolyTransform")
    if transform_type != "PolyTransform":
        raise DescriptorConfigError(
            "ACE trans.type must be 'PolyTransform'", path=["trans", "type"]
        )
    transform_r0 = _positive_float(value.get("r0", r0), "ACE trans.r0")
    power = _positive_float(value.get("p", 2.0), "ACE trans.p")
    shift = _positive_float(value.get("a", 1.0), "ACE trans.a", nonnegative=True)
    if shift + transform_r0 <= 0.0:  # defensive; both validated above
        raise DescriptorConfigError(
            "ACE trans requires trans.a + trans.r0 > 0", path=["trans"]
        )
    return {"type": "PolyTransform", "p": power, "r0": transform_r0, "a": shift}


def _normalize_degree_mapping(value: Any) -> dict[str, float | str] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise DescriptorConfigError("ACE D must be a JSON object", path=["D"])
    unknown = set(value) - {"type", "wL", "csp", "chc", "ahc", "bhc"}
    if unknown:
        names = ", ".join(sorted(str(item) for item in unknown))
        first = sorted(str(item) for item in unknown)[0]
        raise DescriptorConfigError(
            f"ACE D has unsupported field(s): {names}", path=["D", first]
        )
    degree_type = value.get("type", "SparsePSHDegree")
    if degree_type != "SparsePSHDegree":
        raise DescriptorConfigError(
            "ACE D.type must be 'SparsePSHDegree'", path=["D", "type"]
        )
    return {
        "type": "SparsePSHDegree",
        "wL": _positive_float(value.get("wL", 1.5), "ACE D.wL"),
        "csp": _positive_float(value.get("csp", 1.0), "ACE D.csp", nonnegative=True),
        "chc": _positive_float(value.get("chc", 0.0), "ACE D.chc", nonnegative=True),
        "ahc": _positive_float(value.get("ahc", 0.0), "ACE D.ahc", nonnegative=True),
        "bhc": _positive_float(value.get("bhc", 0.0), "ACE D.bhc", nonnegative=True),
    }


def normalize_ace_options(options: Mapping[str, Any]) -> dict[str, Any]:
    """Return a canonical, JSON-safe ACE option mapping.

    This function is used before the common adapter snapshots configuration, so
    symbol species, default transforms, and the ``rin`` relationship are stable
    across direct construction and configuration round-trips.
    """

    result = dict(options)
    result["species"] = normalize_ace_species(result.get("species"))
    if result["species"] is None:
        raise DescriptorConfigError(
            "ACE requires an explicit species declaration at construction",
            path=["species"],
        )

    N = _number(result.get("N", 3), "ACE N")
    if N < 1:
        raise DescriptorConfigError("ACE N must be at least one")
    r0 = _positive_float(result.get("r0", 2.5), "ACE r0")
    result["N"] = N
    result["r0"] = r0
    result["trans"] = _normalize_transform(result.get("trans"), r0)

    result["rcut"] = _positive_float(result.get("rcut", 5.0), "ACE rcut")
    rin_value = result.get("rin")
    result["rin"] = (
        0.5 * r0 if rin_value is None else _positive_float(rin_value, "ACE rin", nonnegative=True)
    )
    if result["rin"] >= result["rcut"]:
        raise DescriptorConfigError(
            "ACE requires 0 <= rin < rcut", path=["rin"]
        )
    result["pcut"] = _number(result.get("pcut", 2), "ACE pcut")
    result["pin"] = _number(result.get("pin", 2), "ACE pin")
    if result["pcut"] < 0 or result["pin"] < 0:
        path = ["pcut"] if result["pcut"] < 0 else ["pin"]
        raise DescriptorConfigError(
            "ACE pcut and pin must be non-negative", path=path
        )
    constants = result.get("constants", False)
    if not isinstance(constants, (bool, np.bool_)):
        raise DescriptorConfigError("ACE constants must be a boolean", path=["constants"])
    result["constants"] = bool(constants)

    maxdeg = _scalar_or_vector(result.get("maxdeg", 8.0), "ACE maxdeg", N)
    wL = _scalar_or_vector(result.get("wL", 1.5), "ACE wL", N)
    degree = _normalize_degree_mapping(result.get("D"))
    if degree is not None:
        if isinstance(maxdeg, list) or isinstance(wL, list):
            raise DescriptorConfigError(
                "ACE explicit D cannot be combined with vector maxdeg or wL",
                path=["D"],
            )
        result["maxdeg"] = float(maxdeg)
        result["wL"] = float(degree["wL"])
        result["D"] = degree
    else:
        # ACE1.Utils._auto_degrees accepts a scalar wL alongside a vector
        # maxdeg and broadcasts it to every correlation order.  Preserve that
        # convenient high-level form while keeping the serialized form fully
        # explicit and JSON-safe.
        if isinstance(maxdeg, list) and not isinstance(wL, list):
            wL = [float(wL)] * N
        elif not isinstance(maxdeg, list) and isinstance(wL, list):
            raise DescriptorConfigError(
                "ACE vector wL requires vector maxdeg", path=["wL"]
            )
        result["maxdeg"] = maxdeg
        result["wL"] = wL
        result["D"] = None

    # Keep the canonical mapping limited to the public ACE options plus the
    # common adapter controls.  This also prevents accidental metadata leakage
    # if a private caller passes an unrelated keyword.
    allowed = {
        "species", "N", "r0", "trans", "wL", "maxdeg", "D", "rcut", "rin",
        "pcut", "pin", "constants", "output", "execution",
    }
    return {key: value for key, value in result.items() if key in allowed}


class AceKernel:
    """Stateful atom-level ACE invariant calculator backed by C++17."""

    name = "ACE"

    def __init__(
        self,
        species: Iterable[Any] | Any | None = None,
        N: int = 3,
        r0: float = 2.5,
        trans: Mapping[str, Any] | None = None,
        wL: float | Iterable[float] = 1.5,
        maxdeg: float | Iterable[float] = 8.0,
        D: Mapping[str, Any] | None = None,
        rcut: float = 5.0,
        rin: float | None = None,
        pcut: int = 2,
        pin: int = 2,
        constants: bool = False,
        num_threads: int | None = None,
    ) -> None:
        canonical = normalize_ace_options({
            "species": species, "N": N, "r0": r0, "trans": trans,
            "wL": wL, "maxdeg": maxdeg, "D": D, "rcut": rcut,
            "rin": rin, "pcut": pcut, "pin": pin, "constants": constants,
        })
        self.species = tuple(canonical["species"])
        self.N = int(canonical["N"])
        self.r0 = float(canonical["r0"])
        self.trans = dict(canonical["trans"])
        self.wL = canonical["wL"]
        self.maxdeg = canonical["maxdeg"]
        self.D = canonical["D"]
        self.rcut = float(canonical["rcut"])
        self.rin = float(canonical["rin"])
        self.pcut = int(canonical["pcut"])
        self.pin = int(canonical["pin"])
        self.constants = bool(canonical["constants"])
        self.num_threads = 0 if num_threads is None else _number(num_threads, "ACE num_threads")
        if self.num_threads < 0:
            raise DescriptorConfigError(
                "ACE num_threads must be non-negative", path=["num_threads"]
            )
        if self.pcut < 2:
            warnings.warn(
                "ACE pcut < 2 may reduce radial smoothness relative to ACE1 defaults",
                UserWarning,
                stacklevel=3,
            )
        if self.pin < 2 and self.pin != 0:
            warnings.warn(
                "ACE pin < 2 may reduce radial smoothness relative to ACE1 defaults",
                UserWarning,
                stacklevel=3,
            )

        options = _cpp.AceOptions()
        options.species = list(self.species)
        options.max_order = self.N
        options.r0 = self.r0
        options.transform_p = float(self.trans["p"])
        options.transform_a = float(self.trans["a"])
        options.r_cut = self.rcut
        options.r_in = self.rin
        options.p_cut = self.pcut
        options.p_in = self.pin
        options.constants = self.constants
        options.num_threads = self.num_threads
        if isinstance(self.D, Mapping):
            options.w_l = float(self.D["wL"])
            options.max_degree = float(self.maxdeg)
            options.degree_csp = float(self.D["csp"])
            options.degree_chc = float(self.D["chc"])
            options.degree_ahc = float(self.D["ahc"])
            options.degree_bhc = float(self.D["bhc"])
        elif isinstance(self.maxdeg, list):
            options.degree_by_order = list(map(float, self.maxdeg))
            options.angular_weight_by_order = list(map(float, self.wL))
        else:
            options.w_l = float(self.wL)
            options.max_degree = float(self.maxdeg)
        self._native = _cpp.AceCalculator(options)
        self._closed = False

    @property
    def feature_count(self) -> int:
        return int(self._native.feature_count)

    def compute(
        self,
        value: StructureBatch | Any,
        control: Any = None,
    ) -> DescriptorResult:
        batch = _as_batch(value)
        validate_batch_species(batch, self.species, descriptor=self.name)
        values = np.asarray(
            self._native.compute(
                batch.numbers,
                batch.positions,
                batch.cells,
                batch.pbc,
                batch.offsets,
                control,
            ),
            dtype=np.float64,
        )
        return DescriptorResult(
            values,
            "atom",
            batch.ids,
            batch.offsets.copy(),
            self._labels(values.shape[1]),
            self._metadata(),
        )

    def close(self) -> None:
        self._closed = True
        self._native.close()

    def _labels(self, width: int | None = None) -> tuple[str, ...]:
        count = self.feature_count if width is None else int(width)
        return tuple(f"ace1:feature={index}" for index in range(count))

    def _metadata(self) -> dict[str, Any]:
        return {
            "backend": "mdescriptor-cpp",
            "descriptor": self.name,
            "reference": {"implementation": "ACE1.jl", "version": "0.12.5", "path": "Utils.rpi_basis"},
            "species": list(self.species),
            "N": self.N,
            "r0": self.r0,
            "trans": dict(self.trans),
            "wL": self.wL,
            "maxdeg": self.maxdeg,
            "D": self.D,
            "rcut": self.rcut,
            "rin": self.rin,
            "pcut": self.pcut,
            "pin": self.pin,
            "constants": self.constants,
            "feature_counts": [int(value) for value in self._native.feature_counts],
            "max_angular": int(self._native.max_angular),
            "max_radial": int(self._native.max_radial),
            "num_threads": None if self.num_threads == 0 else self.num_threads,
        }


class _AceAdapterMixin(DescriptorAdapter):
    """Canonicalize ACE symbols/options before ``DescriptorAdapter`` snapshots."""

    def _initialize_public(self, options: Mapping[str, Any]) -> None:
        DescriptorAdapter._initialize(self, normalize_ace_options(options))


__all__ = ["AceKernel", "normalize_ace_options", "normalize_ace_species"]
