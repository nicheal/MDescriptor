"""The single built-in descriptor specification list."""

from __future__ import annotations

from typing import Any

from .info import DescriptorInfo
from .model_defaults import bundled_model_default
from .registry import DescriptorRegistry
from .spec import AssetPolicy, DescriptorSpec

_STANDALONE = "mdescriptor.descriptors.standalone."
_LOCAL = _STANDALONE + "local:"
_MATRICES = _STANDALONE + "matrices:"
_MANY_BODY = _STANDALONE + "many_body:"
_ROTATIONAL = _STANDALONE + "rotational:"
_MODEL = "mdescriptor.descriptors.model_backed."
_MISSING = object()
_ALL_PERIODICITY = ("isolated", "fully_periodic")
_PERIODIC_ONLY = ("fully_periodic",)


def _parameter(
    kind: str,
    *,
    description: str = "",
    required: bool = False,
    default: Any = _MISSING,
    **constraints: Any,
) -> dict[str, Any]:
    value: dict[str, Any] = {"type": kind, "required": required}
    if default is not _MISSING:
        value["default"] = default
    if description:
        value["description"] = description
    value.update(constraints)
    return value


def _species() -> dict[str, Any]:
    return _parameter(
        "species",
        required=True,
        description="Chemical species included in the descriptor.",
    )


def _model(*, default: str | None = None) -> dict[str, Any]:
    return _parameter(
        "model",
        description=(
            "Optional model resource. A JSON string is an explicit local path; "
            "a serialized ModelResource selects a named or checked resource."
        ),
        default=_MISSING if default is None else bundled_model_default(default),
    )


def _array(
    item_type: str,
    *,
    description: str = "",
    default: Any = _MISSING,
) -> dict[str, Any]:
    return _parameter(
        "array",
        description=description,
        default=default,
        items={"type": item_type},
    )


def _object(
    *,
    description: str = "",
    default: Any = _MISSING,
    properties: dict[str, Any] | None = None,
) -> dict[str, Any]:
    constraints: dict[str, Any] = {}
    if properties is not None:
        constraints["properties"] = properties
    return _parameter("object", description=description, default=default, **constraints)


def _enum(
    values: tuple[Any, ...],
    *,
    description: str = "",
    default: Any = _MISSING,
) -> dict[str, Any]:
    return _parameter(
        "enum",
        description=description,
        default=default,
        enum=list(values),
    )


def _asset(
    policy: AssetPolicy,
    *,
    bundled: tuple[str, ...] = (),
    extensions: tuple[str, ...] = (),
    allow_external: bool | None = None,
) -> dict[str, Any]:
    return {
        "policy": policy.value,
        "parameter": "model" if policy is not AssetPolicy.NONE else None,
        "allow_external": policy is not AssetPolicy.NONE
        if allow_external is None
        else allow_external,
        "bundled_resources": list(bundled),
        "file_extensions": list(extensions),
    }


def _info(
    display_name: str,
    description: str,
    category: str,
    parameters: dict[str, Any],
    *,
    periodicity: tuple[str, ...] = _ALL_PERIODICITY,
    spin: bool = False,
    charge_spin: bool = False,
    cooperative_cancel: bool = True,
    asset: dict[str, Any] | None = None,
) -> DescriptorInfo:
    return DescriptorInfo(
        display_name,
        description,
        category,
        parameters,
        {
            "devices": ["cpu"],
            "num_threads": True,
            "cooperative_cancel": cooperative_cancel,
        },
        {
            "periodicity": list(periodicity),
            "mixed_periodicity": False,
            "spin": spin,
            "charge_spin": charge_spin,
        },
        {"dtypes": ["float32", "float64"], "sparse": True},
        asset or _asset(AssetPolicy.NONE),
    )


_DESCRIPTOR_INFO = {
    "SOAP": _info(
        "SOAP",
        "Smooth overlap of atomic positions descriptor.",
        "local",
        {
            "species": _species(),
            "rbf": _enum(("gto", "polynomial"), default="gto"),
            "n_max": _parameter("integer", default=8, minimum=1),
            "l_max": _parameter("integer", default=6, minimum=0, maximum=20),
            "sigma": _parameter("number", default=1.0, exclusiveMinimum=0.0, unit="Å"),
            "average": _enum(("off", "inner", "outer"), default="inner"),
            "weighting": _object(default={}),
            "r_cut": _parameter("number", exclusiveMinimum=0.0, unit="Å"),
            "compression": _object(
                default={"mode": "off", "species_weighting": None}
            ),
        },
    ),
    "SOAPTurbo": _info(
        "SOAP Turbo",
        "Species-resolved compressed SOAP power spectrum descriptor.",
        "local",
        {
            "species": _species(),
            # These values are per-species. Their defaults cannot be expressed
            # statically because the species declaration determines the length.
            # The kernel keeps scalar broadcast defaults for direct Python use.
            "alpha_max": _array("integer"),
            "l_max": _parameter("integer", default=6, minimum=0, maximum=20),
            "rcut_hard": _parameter("number", default=5.0, exclusiveMinimum=0.0, unit="Å"),
            "rcut_soft": _parameter("number", exclusiveMinimum=0.0, unit="Å"),
            "nf": _parameter("number", default=1.0, exclusiveMinimum=0.0),
            "radial_enhancement": _parameter("integer", default=0, enum=[0, 1, 2]),
            "basis": _enum(("poly3", "poly3gauss"), default="poly3"),
            "compression": _enum(
                ("off", "trivial", "0_0", "0_1", "0_2", "1_0", "1_1", "1_2", "2_0", "2_1", "2_2"),
                default="off",
            ),
            "atom_sigma_r": _array("number"),
            "atom_sigma_r_scaling": _array("number"),
            "atom_sigma_t": _array("number"),
            "atom_sigma_t_scaling": _array("number"),
            "amplitude_scaling": _array("number"),
            "central_weight": _array("number"),
            "central_species": _array("integer"),
        },
    ),
    "ACSF": _info(
        "ACSF",
        "Atom-centred symmetry functions descriptor.",
        "local",
        {
            "species": _species(),
            "r_cut": _parameter("number", default=6.0, exclusiveMinimum=0.0, unit="Å"),
            "g2_params": _object(default={}),
            "g3_params": _array("number", default=[]),
            "g4_params": _object(default={}),
            "g5_params": _object(default={}),
        },
    ),
    "ACE": _info(
        "ACE",
        "Atomic cluster expansion descriptor.",
        "local",
        {
            "species": _species(),
            "N": _parameter("integer", default=3, minimum=1),
            "r0": _parameter("number", default=2.5, exclusiveMinimum=0.0, unit="Å"),
            "trans": _object(
                default=None,
                properties={
                    "type": _enum(("PolyTransform",), default="PolyTransform"),
                    "p": _parameter("number", default=2.0, exclusiveMinimum=0.0),
                    "r0": _parameter("number", exclusiveMinimum=0.0),
                    "a": _parameter("number", default=1.0, minimum=0.0),
                },
            ),
            # ACE accepts either one scalar or one value per correlation
            # order. The schema uses the canonical array form; the config
            # validator retains scalar broadcast compatibility for old
            # Python configurations.
            # These arrays depend on N, so they intentionally have no static
            # defaults. The constructor's scalar defaults remain the valid
            # broadcast form for a newly created descriptor.
            "wL": _array("number"),
            "maxdeg": _array("number"),
            "D": _object(
                default=None,
                properties={
                    "type": _enum(("SparsePSHDegree",), default="SparsePSHDegree"),
                    "wL": _parameter("number", default=1.5, exclusiveMinimum=0.0),
                    "csp": _parameter("number", default=1.0, minimum=0.0),
                    "chc": _parameter("number", default=0.0, minimum=0.0),
                    "ahc": _parameter("number", default=0.0, minimum=0.0),
                    "bhc": _parameter("number", default=0.0, minimum=0.0),
                },
            ),
            "rcut": _parameter("number", default=5.0, exclusiveMinimum=0.0, unit="Å"),
            "rin": _parameter("number", minimum=0.0, unit="Å"),
            "pcut": _parameter("integer", default=2, minimum=0),
            "pin": _parameter("integer", default=2, minimum=0),
            "constants": _parameter("boolean", default=False),
        },
    ),
    "CoulombMatrix": _info(
        "Coulomb Matrix",
        "Coulomb interaction matrix descriptor.",
        "matrix",
        {
            "n_atoms_max": _parameter("integer", minimum=1),
            "permutation": _enum(("none", "sorted_l2", "eigenspectrum"), default="sorted_l2"),
            "exponent": _parameter("number", default=2.4),
        },
    ),
    "SineMatrix": _info(
        "Sine Matrix",
        "Periodic sine interaction matrix descriptor.",
        "matrix",
        {
            "n_atoms_max": _parameter("integer", minimum=1),
            "permutation": _enum(("none", "sorted_l2", "eigenspectrum"), default="sorted_l2"),
            "exponent": _parameter("number", default=2.4),
        },
        periodicity=_PERIODIC_ONLY,
    ),
    "EwaldSumMatrix": _info(
        "Ewald Sum Matrix",
        "Ewald-summed periodic interaction matrix descriptor.",
        "matrix",
        {
            "n_atoms_max": _parameter("integer", minimum=1),
            "permutation": _enum(("none", "sorted_l2", "eigenspectrum"), default="sorted_l2"),
            "accuracy": _parameter("number", default=1e-5, exclusiveMinimum=0.0, maximum=1.0),
            "w": _parameter("number", default=1.0, exclusiveMinimum=0.0),
            "r_cut": _parameter("number", exclusiveMinimum=0.0, unit="Å"),
            "g_cut": _parameter("number", exclusiveMinimum=0.0, unit="Å"),
            "a": _parameter("number", exclusiveMinimum=0.0),
        },
        periodicity=_PERIODIC_ONLY,
    ),
    "MBTR": _info(
        "MBTR",
        "Many-body tensor representation descriptor.",
        "many_body",
        {
            "species": _species(),
            "geometry": _object(default={"function": "distance"}),
            "grid": _object(default={"min": 0.0, "max": 6.0, "n": 50, "sigma": 0.1}),
            "weighting": _object(default={"function": "exp", "scale": 0.5, "threshold": 1e-3}),
            "periodic": _parameter("boolean", default=True, enum=[True]),
            "normalize_gaussians": _parameter("boolean", default=True),
            "normalization": _enum(("none", "l2", "n_atoms", "valle_oganov"), default="none"),
        },
        periodicity=_PERIODIC_ONLY,
    ),
    "LMBTR": _info(
        "Local MBTR",
        "Local many-body tensor representation descriptor.",
        "many_body",
        {
            "species": _species(),
            "geometry": _object(default={"function": "distance"}),
            "grid": _object(default={"min": 0.0, "max": 6.0, "n": 50, "sigma": 0.1}),
            "weighting": _object(default={"function": "exp", "scale": 0.5, "threshold": 1e-3}),
            "periodic": _parameter("boolean", default=True, enum=[True]),
            "normalize_gaussians": _parameter("boolean", default=True),
            "normalization": _enum(("none", "l2", "n_atoms", "valle_oganov"), default="none"),
        },
        periodicity=_PERIODIC_ONLY,
    ),
    "ValleOganov": _info(
        "Valle–Oganov",
        "Valle-Oganov structural fingerprint descriptor.",
        "many_body",
        {
            "species": _species(),
            "function": _enum(("distance", "angle"), default="distance"),
            "n": _parameter("integer", default=50, minimum=2),
            "sigma": _parameter("number", default=0.1, exclusiveMinimum=0.0),
            "r_cut": _parameter("number", default=6.0, exclusiveMinimum=0.0, unit="Å"),
            "geometry": _object(),
            "grid": _object(),
            "weighting": _object(),
            "periodic": _parameter("boolean", default=True, enum=[True]),
            "normalize_gaussians": _parameter("boolean", default=True),
            "normalization": _enum(("none", "l2", "n_atoms", "valle_oganov"), default="valle_oganov"),
        },
        periodicity=_PERIODIC_ONLY,
    ),
    "AtomicComposition": _info(
        "Atomic Composition",
        "Species composition vector descriptor.",
        "local",
        {
            "species": _species(),
            "per_system": _parameter("boolean", default=True),
        },
    ),
    "NeighborList": _info(
        "Neighbor List",
        "Neighbor pair displacement and distance descriptor.",
        "local",
        {
            "cutoff": _parameter("number", default=6.0, exclusiveMinimum=0.0, unit="Å"),
            "full_neighbor_list": _parameter("boolean", default=True),
            "self_pairs": _parameter("boolean", default=False),
        },
    ),
    "SortedDistances": _info(
        "Sorted Distances",
        "Sorted neighbor distance descriptor.",
        "local",
        {
            "species": _species(),
            "cutoff": _parameter("number", default=6.0, exclusiveMinimum=0.0, unit="Å"),
            "max_neighbors": _parameter("integer", default=8, minimum=1),
            "separate_neighbor_types": _parameter("boolean", default=True),
        },
    ),
    "SphericalExpansion": _info(
        "Spherical Expansion",
        "Species-resolved spherical density expansion descriptor.",
        "local",
        {
            "species": _species(),
            "cutoff": _parameter("number", default=6.0, exclusiveMinimum=0.0, unit="Å"),
            "density_width": _parameter("number", default=0.3, exclusiveMinimum=0.0, unit="Å"),
            "max_radial": _parameter("integer", default=6, minimum=0),
            "max_angular": _parameter("integer", default=4, minimum=0),
        },
    ),
    "SphericalExpansionByPair": _info(
        "Spherical Expansion by Pair",
        "Pair-level spherical density expansion descriptor.",
        "local",
        {
            "species": _species(),
            "cutoff": _parameter("number", default=6.0, exclusiveMinimum=0.0, unit="Å"),
            "density_width": _parameter("number", default=0.3, exclusiveMinimum=0.0, unit="Å"),
            "max_radial": _parameter("integer", default=6, minimum=0),
            "max_angular": _parameter("integer", default=4, minimum=0),
        },
    ),
    "SoapRadialSpectrum": _info(
        "SOAP Radial Spectrum",
        "Radial spectrum derived from a spherical density expansion.",
        "local",
        {
            "species": _species(),
            "cutoff": _parameter("number", default=6.0, exclusiveMinimum=0.0, unit="Å"),
            "density_width": _parameter("number", default=0.3, exclusiveMinimum=0.0, unit="Å"),
            "max_radial": _parameter("integer", default=6, minimum=0),
            "max_angular": _parameter("integer", default=4, minimum=0),
        },
    ),
    "SoapPowerSpectrum": _info(
        "SOAP Power Spectrum",
        "Power spectrum derived from a spherical density expansion.",
        "local",
        {
            "species": _species(),
            "cutoff": _parameter("number", default=6.0, exclusiveMinimum=0.0, unit="Å"),
            "density_width": _parameter("number", default=0.3, exclusiveMinimum=0.0, unit="Å"),
            "max_radial": _parameter("integer", default=6, minimum=0),
            "max_angular": _parameter("integer", default=4, minimum=0),
        },
    ),
    "LodeSphericalExpansion": _info(
        "LODE Spherical Expansion",
        "Long-distance equivariant spherical density expansion descriptor.",
        "local",
        {
            "species": _species(),
            "cutoff": _parameter("number", default=6.0, exclusiveMinimum=0.0, unit="Å"),
            "density_width": _parameter("number", default=0.3, exclusiveMinimum=0.0, unit="Å"),
            "max_radial": _parameter("integer", default=6, minimum=0),
            "max_angular": _parameter("integer", default=4, minimum=0),
            "k_cutoff": _parameter("number", default=2.5, exclusiveMinimum=0.0),
            "exponent": _parameter("integer", default=1, minimum=0),
            "radial_radius": _parameter("number", exclusiveMinimum=0.0, unit="Å"),
        },
    ),
    "EAD": _info(
        "EAD",
        "Equivariant angular descriptor.",
        "rotational",
        {
            "parameters": _object(default={"L": 3, "eta": [0.05, 0.1, 0.5], "Rs": [0.0]}),
            "Rc": _parameter("number", default=6.0, exclusiveMinimum=0.0, unit="Å"),
            "cutoff": _enum(("cosine",), default="cosine"),
        },
    ),
    "SO3": _info(
        "SO3",
        "SO(3) rotationally invariant descriptor.",
        "rotational",
        {
            "nmax": _parameter("integer", default=3, minimum=1),
            "lmax": _parameter("integer", default=3, minimum=0),
            "rcut": _parameter("number", default=3.5, exclusiveMinimum=0.0, unit="Å"),
            "alpha": _parameter("number", default=2.0, exclusiveMinimum=0.0),
            "weight_on": _parameter("boolean", default=False),
        },
    ),
    "SO4": _info(
        "SO4",
        "SO(4) rotationally invariant descriptor.",
        "rotational",
        {
            "lmax": _parameter("integer", default=3, minimum=0),
            "rcut": _parameter("number", default=3.5, exclusiveMinimum=0.0, unit="Å"),
            "normalize_U": _parameter("boolean", default=False),
        },
    ),
    "SNAP": _info(
        "SNAP",
        "Spectral neighbor analysis potential bispectrum descriptor.",
        "rotational",
        {
            "weights": _object(default={}),
            "lmax": _parameter("integer", default=3, minimum=0),
            "rcut": _parameter("number", default=3.5, exclusiveMinimum=0.0, unit="Å"),
            "normalize_U": _parameter("boolean", default=False),
        },
    ),
    "LBispectrum": _info(
        "L-Bispectrum",
        "Low-rank bispectrum descriptor with optional element profiles.",
        "rotational",
        {
            "twojmax": _parameter("integer", default=3, minimum=0),
            "diagonal": _parameter("integer", default=3, minimum=0, maximum=3),
            "rfac0": _parameter("number", default=0.99363, exclusiveMinimum=0.0),
            "rmin0": _parameter("number", default=0.0, minimum=0.0, unit="Å"),
            "rcutfac": _parameter("number", default=1.0, exclusiveMinimum=0.0),
            "element_profile": _object(),
            "element_radii": _object(),
            "weights": _object(),
            "rcut": _parameter("number", default=3.5, exclusiveMinimum=0.0, unit="Å"),
            "normalize_U": _parameter("boolean", default=False),
        },
    ),
    "MTP": _info(
        "MTP",
        "Moment tensor potential basis descriptor.",
        "local",
        {
            "species": _species(),
            "model": _model(),
            "min_dist": _parameter("number", default=0.0, minimum=0.0, unit="Å"),
            "max_dist": _parameter("number", default=5.0, exclusiveMinimum=0.0, unit="Å"),
            "r_cut": _parameter("number", exclusiveMinimum=0.0, unit="Å"),
            "radial_basis_size": _parameter("integer", default=4, minimum=1),
            "radial_funcs_count": _parameter("integer", default=1, minimum=1),
            "max_rank": _parameter("integer", default=2, minimum=0, maximum=5),
            "radial_basis_type": _enum(
                ("RBChebyshev", "Chebyshev", "polynomial"), default="RBChebyshev"
            ),
        },
        asset=_asset(AssetPolicy.OPTIONAL, extensions=(".json", ".mtp")),
    ),
    "C00PSMLFF": _info(
        "C00PS-MLFF",
        "C00 plus power-spectrum machine-learning force-field descriptor.",
        "local",
        {
            "species": _species(),
            "r_cut": _parameter("number", default=6.0, exclusiveMinimum=0.0, unit="Å"),
            "n_radial": _parameter("integer", default=8, minimum=1),
            "l_max": _parameter("integer", default=4, minimum=0),
            "cutoff_function": _enum(("bp", "mo", "rj", "wmc"), default="bp"),
            "radial_sigma": _parameter("number", default=0.5, minimum=0.0, unit="Å"),
            "include_radial": _parameter("boolean", default=True),
            "include_angular": _parameter("boolean", default=True),
            "normalize_radial": _parameter("boolean", default=False),
            "normalize_angular": _parameter("boolean", default=False),
            "super_vector": _parameter("boolean", default=False),
            "radial_weight": _parameter("number", default=1.0, minimum=0.0),
            "angular_weight": _parameter("number", default=1.0, minimum=0.0),
            "exclude_self_interaction": _parameter("boolean", default=True),
        },
    ),
    "NEP": _info(
        "NEP",
        "Neuroevolution potential descriptor backed by a local model.",
        "model_backed",
        {"model": _model(default="NEP")},
        asset=_asset(
            AssetPolicy.REQUIRED,
            bundled=("nep89_20250409.txt",),
            extensions=(".txt",),
        ),
    ),
    "DPA4": _info(
        "DPA4",
        "Deep potential atom descriptor backed by a local checkpoint.",
        "model_backed",
        {"model": _model(default="DPA4")},
        spin=True,
        charge_spin=True,
        cooperative_cancel=False,
        asset=_asset(
            AssetPolicy.REQUIRED,
            bundled=("DPA4-Air-OMat24-v20260704.pt",),
            extensions=(".pt",),
        ),
    ),
    "DPA4C": _info(
        "DPA4C",
        "Calibrated deep potential atom descriptor backed by a local checkpoint.",
        "model_backed",
        {
            "model": _model(default="DPA4C"),
            "calibrate": _parameter("boolean", default=True),
        },
        spin=True,
        charge_spin=True,
        cooperative_cancel=False,
        asset=_asset(
            AssetPolicy.REQUIRED,
            bundled=("DPA4C-Air-OMat24-v20260819.pt",),
            extensions=(".pt",),
        ),
    ),
}


def _capabilities(info: DescriptorInfo) -> frozenset[str]:
    """Derive runtime capabilities from the GUI-facing metadata record."""

    capabilities: set[str] = set()
    if info.output.get("sparse", False):
        capabilities.add("sparse")
    if info.execution.get("num_threads", False):
        capabilities.add("num_threads")
    if info.execution.get("cooperative_cancel", False):
        capabilities.add("cooperative_cancel")
    if info.input.get("spin", False):
        capabilities.add("spin")
    if info.input.get("charge_spin", False):
        capabilities.add("charge_spin")
    if "model" in info.parameters or info.asset.get("parameter") == "model":
        capabilities.add("model")
    return frozenset(capabilities)


def _spec(
    name: str,
    import_path: str,
    backend: str,
    level: str,
    *,
    optional_extra: str | None = None,
) -> DescriptorSpec:
    info = _DESCRIPTOR_INFO[name]
    try:
        asset_policy = AssetPolicy(info.asset.get("policy", AssetPolicy.NONE.value))
    except ValueError as exc:  # pragma: no cover - guarded by DescriptorInfo
        raise ValueError(f"unknown asset policy for {name!r}") from exc
    return DescriptorSpec(
        name,
        import_path,
        asset_policy,
        backend,
        level,
        capabilities=_capabilities(info),
        optional_extra=optional_extra,
        info=info,
    )


_BUILTIN_SPECS = (
    _spec("SOAP", _STANDALONE + "soap:SOAP", "cpp", "structure"),
    _spec("SOAPTurbo", _STANDALONE + "soap_turbo:SOAPTurbo", "cpp", "atom"),
    _spec("ACSF", _STANDALONE + "acsf:ACSF", "cpp", "atom"),
    _spec("ACE", _STANDALONE + "ace:ACE", "cpp", "atom"),
    _spec("CoulombMatrix", _MATRICES + "CoulombMatrix", "cpp", "structure"),
    _spec("SineMatrix", _MATRICES + "SineMatrix", "cpp", "structure"),
    _spec("EwaldSumMatrix", _MATRICES + "EwaldSumMatrix", "cpp", "structure"),
    _spec("MBTR", _MANY_BODY + "MBTR", "cpp", "structure"),
    _spec("LMBTR", _MANY_BODY + "LMBTR", "cpp", "atom"),
    _spec("ValleOganov", _MANY_BODY + "ValleOganov", "cpp", "structure"),
    _spec("AtomicComposition", _LOCAL + "AtomicComposition", "cpp", "structure"),
    _spec("NeighborList", _LOCAL + "NeighborList", "cpp", "pair"),
    _spec("SortedDistances", _LOCAL + "SortedDistances", "cpp", "atom"),
    _spec("SphericalExpansion", _LOCAL + "SphericalExpansion", "cpp", "atom"),
    _spec("SphericalExpansionByPair", _LOCAL + "SphericalExpansionByPair", "cpp", "pair"),
    _spec("SoapRadialSpectrum", _LOCAL + "SoapRadialSpectrum", "cpp", "atom"),
    _spec("SoapPowerSpectrum", _LOCAL + "SoapPowerSpectrum", "cpp", "atom"),
    _spec("LodeSphericalExpansion", _LOCAL + "LodeSphericalExpansion", "cpp", "atom"),
    _spec("EAD", _ROTATIONAL + "EAD", "cpp", "atom"),
    _spec("SO3", _ROTATIONAL + "SO3", "cpp", "atom"),
    _spec("SO4", _ROTATIONAL + "SO4", "cpp", "atom"),
    _spec("SNAP", _ROTATIONAL + "SNAP", "cpp", "atom"),
    _spec("LBispectrum", _ROTATIONAL + "LBispectrum", "cpp", "atom"),
    _spec("MTP", _STANDALONE + "mtp:MTP", "cpp", "atom"),
    _spec("C00PSMLFF", _STANDALONE + "c00ps_mlff:C00PSMLFF", "cpp", "atom"),
    _spec("NEP", _MODEL + "nep.descriptor:NEP", "cpp", "atom"),
    _spec("DPA4", _MODEL + "dpa4.descriptor:DPA4", "numpy", "atom"),
    _spec("DPA4C", _MODEL + "dpa4c.descriptor:DPA4C", "numpy", "atom"),
)

builtin_registry = DescriptorRegistry(_BUILTIN_SPECS, frozen=True)
