"""The single built-in descriptor specification list."""

from __future__ import annotations

from typing import Any

from .._runtime import native_extension_available
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
_DPA_EXECUTION_ENGINE = "cpp" if native_extension_available() else "numpy"


def _parameter(
    kind: str,
    *,
    display_name: str = "",
    description: str = "",
    required: bool = False,
    default: Any = _MISSING,
    **constraints: Any,
) -> dict[str, Any]:
    value: dict[str, Any] = {"type": kind, "required": required}
    if display_name:
        value["display_name"] = display_name
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
        display_name="Chemical species",
        description="Chemical species included in the descriptor.",
    )


def _model(*, default: str | None = None) -> dict[str, Any]:
    return _parameter(
        "model",
        display_name="Model resource",
        description=(
            "Optional model resource. A JSON string is an explicit local path; "
            "a serialized ModelResource selects a named or checked resource."
        ),
        default=_MISSING if default is None else bundled_model_default(default),
    )


def _array(
    item_type: str,
    *,
    display_name: str = "",
    description: str = "",
    default: Any = _MISSING,
) -> dict[str, Any]:
    return _parameter(
        "array",
        display_name=display_name,
        description=description,
        default=default,
        items={"type": item_type},
    )


def _object(
    *,
    display_name: str = "",
    description: str = "",
    default: Any = _MISSING,
    properties: dict[str, Any] | None = None,
) -> dict[str, Any]:
    constraints: dict[str, Any] = {}
    if properties is not None:
        constraints["properties"] = properties
    return _parameter(
        "object",
        display_name=display_name,
        description=description,
        default=default,
        **constraints,
    )


def _enum(
    values: tuple[Any, ...],
    *,
    display_name: str = "",
    description: str = "",
    default: Any = _MISSING,
) -> dict[str, Any]:
    return _parameter(
        "enum",
        display_name=display_name,
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
    mixed_periodicity: bool | None = None,
    devices: tuple[str, ...] = ("cpu",),
    asset: dict[str, Any] | None = None,
) -> DescriptorInfo:
    if mixed_periodicity is None:
        mixed_periodicity = set(periodicity) == set(_ALL_PERIODICITY)
    return DescriptorInfo(
        display_name,
        description,
        category,
        parameters,
        {
            "devices": list(devices),
            "num_threads": True,
            "cooperative_cancel": cooperative_cancel,
        },
        {
            "periodicity": list(periodicity),
            "mixed_periodicity": set(periodicity) == set(_ALL_PERIODICITY),
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
            "rbf": _enum(
                ("gto", "polynomial"),
                display_name="Radial basis",
                description="Radial basis family used to expand the neighbor density.",
                default="gto",
            ),
            "n_max": _parameter(
                "integer",
                display_name="Maximum radial order",
                description="Number of radial basis functions used for each species.",
                default=8,
                minimum=1,
            ),
            "l_max": _parameter(
                "integer",
                display_name="Maximum angular order",
                description="Largest spherical-harmonic degree included in the expansion.",
                default=6,
                minimum=0,
                maximum=20,
            ),
            "sigma": _parameter(
                "number",
                display_name="Density width",
                description="Gaussian width used to smear each atomic neighbor density.",
                default=1.0,
                exclusiveMinimum=0.0,
                unit="Å",
            ),
            "average": _enum(
                ("off", "inner", "outer"),
                display_name="Environment averaging",
                description="Controls whether atom-centered environments are averaged and at which stage.",
                default="inner",
            ),
            "weighting": _object(
                display_name="Distance weighting",
                description="Optional distance-dependent weighting applied to neighbor density.",
                default={},
            ),
            "r_cut": _parameter(
                "number",
                display_name="Cutoff radius",
                description="Maximum distance from the center included in the descriptor.",
                exclusiveMinimum=0.0,
                unit="Å",
            ),
            "compression": _object(
                display_name="Feature compression",
                description="Optional compression of the species-resolved SOAP power spectrum.",
                default={"mode": "off", "species_weighting": None},
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
            "alpha_max": _array(
                "integer",
                display_name="Radial channels per species",
                description="Number of radial channels retained for each chemical species.",
            ),
            "l_max": _parameter(
                "integer",
                display_name="Maximum angular order",
                description="Largest spherical-harmonic degree included in the power spectrum.",
                default=6,
                minimum=0,
                maximum=20,
            ),
            "rcut_hard": _parameter(
                "number",
                display_name="Hard cutoff radius",
                description="Outer neighbor distance beyond which contributions are discarded.",
                default=5.0,
                exclusiveMinimum=0.0,
                unit="Å",
            ),
            "rcut_soft": _parameter(
                "number",
                display_name="Soft cutoff radius",
                description="Radius at which the smooth cutoff region begins; it cannot exceed the hard cutoff.",
                exclusiveMinimum=0.0,
                unit="Å",
            ),
            "nf": _parameter(
                "number",
                display_name="Cutoff smoothing factor",
                description="Positive factor controlling the radial cutoff smoothing profile.",
                default=1.0,
                exclusiveMinimum=0.0,
            ),
            "radial_enhancement": _parameter(
                "integer",
                display_name="Radial enhancement mode",
                description="Selects the radial enhancement variant used by SOAP Turbo.",
                default=0,
                enum=[0, 1, 2],
            ),
            "basis": _enum(
                ("poly3", "poly3gauss"),
                display_name="Radial basis",
                description="Radial basis family used for the compressed SOAP representation.",
                default="poly3",
            ),
            "compression": _enum(
                ("off", "trivial", "0_0", "0_1", "0_2", "1_0", "1_1", "1_2", "2_0", "2_1", "2_2"),
                display_name="Feature compression",
                description="Compression recipe for reducing SOAP Turbo power-spectrum channels.",
                default="off",
            ),
            "atom_sigma_r": _array(
                "number",
                display_name="Radial atomic width",
                description="Per-species width of the radial atomic density contribution.",
            ),
            "atom_sigma_r_scaling": _array(
                "number",
                display_name="Radial width scaling",
                description="Per-species scaling applied to the radial atomic width.",
            ),
            "atom_sigma_t": _array(
                "number",
                display_name="Tangential atomic width",
                description="Per-species width of the tangential/angular atomic density contribution.",
            ),
            "atom_sigma_t_scaling": _array(
                "number",
                display_name="Tangential width scaling",
                description="Per-species scaling applied to the tangential atomic width.",
            ),
            "amplitude_scaling": _array(
                "number",
                display_name="Amplitude scaling",
                description="Per-species scaling of neighbor density amplitudes.",
            ),
            "central_weight": _array(
                "number",
                display_name="Central-atom weight",
                description="Per-species weight assigned to the central atom contribution.",
            ),
            "central_species": _array(
                "integer",
                display_name="Central species filter",
                description="Optional atomic numbers for which central-atom environments are generated.",
            ),
        },
    ),
    "ACSF": _info(
        "ACSF",
        "Atom-centred symmetry functions descriptor.",
        "local",
        {
            "species": _species(),
            "r_cut": _parameter(
                "number",
                display_name="Cutoff radius",
                description="Maximum neighbor distance used by the symmetry functions.",
                default=6.0,
                exclusiveMinimum=0.0,
                unit="Å",
            ),
            "g2_params": _object(
                display_name="G2 radial parameters",
                description="Parameter groups for radial G2 symmetry functions, typically eta and Rs.",
                default={},
            ),
            "g3_params": _array(
                "number",
                display_name="G3 radial parameters",
                description="Kappa values controlling the radial G3 symmetry functions.",
                default=[],
            ),
            "g4_params": _object(
                display_name="G4 angular parameters",
                description="Parameter groups for angular G4 symmetry functions: eta, zeta, and lambda.",
                default={},
            ),
            "g5_params": _object(
                display_name="G5 angular parameters",
                description="Parameter groups for angular G5 symmetry functions: eta, zeta, and lambda.",
                default={},
            ),
        },
    ),
    "ACE": _info(
        "ACE",
        "Atomic cluster expansion descriptor.",
        "local",
        {
            "species": _species(),
            "N": _parameter(
                "integer",
                display_name="Maximum correlation order",
                description="Highest body/correlation order included in the ACE basis.",
                default=3,
                minimum=1,
            ),
            "r0": _parameter(
                "number",
                display_name="Reference radial scale",
                description="Reference radius used by the ACE radial transform.",
                default=2.5,
                exclusiveMinimum=0.0,
                unit="Å",
            ),
            "trans": _object(
                display_name="Radial transform",
                description="Configuration of the radial coordinate transform used by ACE.",
                default=None,
                properties={
                    "type": _enum(
                        ("PolyTransform",),
                        display_name="Transform type",
                        description="Radial transform family.",
                        default="PolyTransform",
                    ),
                    "p": _parameter(
                        "number",
                        display_name="Transform power",
                        description="Power used by the polynomial radial transform.",
                        default=2.0,
                        exclusiveMinimum=0.0,
                    ),
                    "r0": _parameter(
                        "number",
                        display_name="Transform reference radius",
                        description="Reference radius used inside the radial transform.",
                        exclusiveMinimum=0.0,
                    ),
                    "a": _parameter(
                        "number",
                        display_name="Transform shift",
                        description="Non-negative shift applied by the radial transform.",
                        default=1.0,
                        minimum=0.0,
                    ),
                },
            ),
            # ACE accepts either one scalar or one value per correlation
            # order. The schema uses the canonical array form; the config
            # validator retains scalar broadcast compatibility for old
            # Python configurations.
            # These arrays depend on N, so they intentionally have no static
            # defaults. The constructor's scalar defaults remain the valid
            # broadcast form for a newly created descriptor.
            "wL": _array(
                "number",
                display_name="Angular-degree weights",
                description="Weights controlling the relative contribution of ACE angular degrees.",
            ),
            "maxdeg": _array(
                "number",
                display_name="Maximum polynomial degrees",
                description="Maximum polynomial degree allowed for each ACE correlation order.",
            ),
            "D": _object(
                display_name="Sparse degree mapping",
                description="Optional explicit sparse mapping for ACE degree selection.",
                default=None,
                properties={
                    "type": _enum(
                        ("SparsePSHDegree",),
                        display_name="Degree mapping type",
                        description="Sparse degree-mapping family.",
                        default="SparsePSHDegree",
                    ),
                    "wL": _parameter(
                        "number",
                        display_name="Degree-map angular weight",
                        description="Angular-degree weight used by the sparse degree mapper.",
                        default=1.5,
                        exclusiveMinimum=0.0,
                    ),
                    "csp": _parameter(
                        "number",
                        display_name="Species degree coefficient",
                        description="Coefficient controlling species-dependent degree selection.",
                        default=1.0,
                        minimum=0.0,
                    ),
                    "chc": _parameter(
                        "number",
                        display_name="Correlation degree coefficient",
                        description="Coefficient controlling correlation-order degree selection.",
                        default=0.0,
                        minimum=0.0,
                    ),
                    "ahc": _parameter(
                        "number",
                        display_name="Angular degree coefficient",
                        description="Coefficient controlling angular-degree selection.",
                        default=0.0,
                        minimum=0.0,
                    ),
                    "bhc": _parameter(
                        "number",
                        display_name="Body-order coefficient",
                        description="Coefficient controlling body-order degree selection.",
                        default=0.0,
                        minimum=0.0,
                    ),
                },
            ),
            "rcut": _parameter(
                "number",
                display_name="Outer cutoff radius",
                description="Outer radius of the ACE neighbor environment.",
                default=5.0,
                exclusiveMinimum=0.0,
                unit="Å",
            ),
            "rin": _parameter(
                "number",
                display_name="Inner cutoff radius",
                description="Inner radius at which the ACE radial cutoff begins.",
                minimum=0.0,
                unit="Å",
            ),
            "pcut": _parameter(
                "integer",
                display_name="Outer cutoff polynomial degree",
                description="Polynomial degree used for the outer radial cutoff.",
                default=2,
                minimum=0,
            ),
            "pin": _parameter(
                "integer",
                display_name="Inner cutoff polynomial degree",
                description="Polynomial degree used for the inner radial cutoff.",
                default=2,
                minimum=0,
            ),
            "constants": _parameter(
                "boolean",
                display_name="Include constant terms",
                description="Whether to include constant basis terms in the ACE output.",
                default=False,
            ),
        },
    ),
    "CoulombMatrix": _info(
        "Coulomb Matrix",
        "Coulomb interaction matrix descriptor.",
        "matrix",
        {
            "n_atoms_max": _parameter(
                "integer",
                display_name="Maximum atom count",
                description="Maximum number of atoms represented in the padded matrix.",
                minimum=1,
            ),
            "permutation": _enum(
                ("none", "sorted_l2", "eigenspectrum"),
                display_name="Permutation handling",
                description="How atom ordering is handled to obtain a permutation-robust representation.",
                default="sorted_l2",
            ),
            "exponent": _parameter(
                "number",
                display_name="Atomic-number exponent",
                description="Exponent applied to atomic-number terms in the interaction matrix.",
                default=2.4,
            ),
        },
    ),
    "SineMatrix": _info(
        "Sine Matrix",
        "Periodic sine interaction matrix descriptor.",
        "matrix",
        {
            "n_atoms_max": _parameter(
                "integer",
                display_name="Maximum atom count",
                description="Maximum number of atoms represented in the padded matrix.",
                minimum=1,
            ),
            "permutation": _enum(
                ("none", "sorted_l2", "eigenspectrum"),
                display_name="Permutation handling",
                description="How atom ordering is handled to obtain a permutation-robust representation.",
                default="sorted_l2",
            ),
            "exponent": _parameter(
                "number",
                display_name="Atomic-number exponent",
                description="Exponent applied to atomic-number terms in the periodic sine matrix.",
                default=2.4,
            ),
        },
        periodicity=_PERIODIC_ONLY,
    ),
    "EwaldSumMatrix": _info(
        "Ewald Sum Matrix",
        "Ewald-summed periodic interaction matrix descriptor.",
        "matrix",
        {
            "n_atoms_max": _parameter(
                "integer",
                display_name="Maximum atom count",
                description="Maximum number of atoms represented in the padded matrix.",
                minimum=1,
            ),
            "permutation": _enum(
                ("none", "sorted_l2", "eigenspectrum"),
                display_name="Permutation handling",
                description="How atom ordering is handled to obtain a permutation-robust representation.",
                default="sorted_l2",
            ),
            "accuracy": _parameter(
                "number",
                display_name="Ewald accuracy",
                description="Target numerical accuracy for the Ewald summation.",
                default=1e-5,
                exclusiveMinimum=0.0,
                maximum=1.0,
            ),
            "w": _parameter(
                "number",
                display_name="Ewald weighting",
                description="Positive weighting parameter used in the Ewald interaction construction.",
                default=1.0,
                exclusiveMinimum=0.0,
            ),
            "r_cut": _parameter(
                "number",
                display_name="Real-space cutoff",
                description="Real-space cutoff radius used by the Ewald sum.",
                exclusiveMinimum=0.0,
                unit="Å",
            ),
            "g_cut": _parameter(
                "number",
                display_name="Reciprocal-space cutoff",
                description="Reciprocal-space cutoff used by the Ewald sum.",
                exclusiveMinimum=0.0,
                unit="Å",
            ),
            "a": _parameter(
                "number",
                display_name="Ewald splitting parameter",
                description="Positive parameter controlling the real-space/reciprocal-space Ewald split.",
                exclusiveMinimum=0.0,
            ),
        },
        periodicity=_PERIODIC_ONLY,
    ),
    "MBTR": _info(
        "MBTR",
        "Many-body tensor representation descriptor.",
        "many_body",
        {
            "species": _species(),
            "geometry": _object(
                display_name="Geometry mapping",
                description="Defines the geometric quantity, such as distance or angle, encoded by MBTR.",
                default={"function": "distance"},
            ),
            "grid": _object(
                display_name="Discretization grid",
                description="Defines the feature range, number of bins, and Gaussian broadening.",
                default={"min": 0.0, "max": 6.0, "n": 50, "sigma": 0.1},
            ),
            "weighting": _object(
                display_name="Geometry weighting",
                description="Defines how contributions are weighted by distance or other geometric criteria.",
                default={"function": "exp", "scale": 0.5, "threshold": 1e-3},
            ),
            "periodic": _parameter(
                "boolean",
                display_name="Periodic calculation",
                description="Enables periodic boundary handling; the current implementation supports only true.",
                default=True,
                enum=[True],
            ),
            "normalize_gaussians": _parameter(
                "boolean",
                display_name="Normalize Gaussian basis",
                description="Normalizes each Gaussian used to broaden the discretized representation.",
                default=True,
            ),
            "normalization": _enum(
                ("none", "l2", "n_atoms", "valle_oganov"),
                display_name="Feature normalization",
                description="Normalization applied to the final MBTR feature vector.",
                default="none",
            ),
        },
        periodicity=_PERIODIC_ONLY,
    ),
    "LMBTR": _info(
        "Local MBTR",
        "Local many-body tensor representation descriptor.",
        "many_body",
        {
            "species": _species(),
            "geometry": _object(
                display_name="Geometry mapping",
                description="Defines the local geometric quantity, such as distance or angle, encoded by LMBTR.",
                default={"function": "distance"},
            ),
            "grid": _object(
                display_name="Discretization grid",
                description="Defines the feature range, number of bins, and Gaussian broadening.",
                default={"min": 0.0, "max": 6.0, "n": 50, "sigma": 0.1},
            ),
            "weighting": _object(
                display_name="Geometry weighting",
                description="Defines how local contributions are weighted by distance or other geometric criteria.",
                default={"function": "exp", "scale": 0.5, "threshold": 1e-3},
            ),
            "periodic": _parameter(
                "boolean",
                display_name="Periodic calculation",
                description="Enables periodic boundary handling; the current implementation supports only true.",
                default=True,
                enum=[True],
            ),
            "normalize_gaussians": _parameter(
                "boolean",
                display_name="Normalize Gaussian basis",
                description="Normalizes each Gaussian used to broaden the discretized representation.",
                default=True,
            ),
            "normalization": _enum(
                ("none", "l2", "n_atoms", "valle_oganov"),
                display_name="Feature normalization",
                description="Normalization applied to the final LMBTR feature vector.",
                default="none",
            ),
        },
        periodicity=_PERIODIC_ONLY,
    ),
    "ValleOganov": _info(
        "Valle–Oganov",
        "Valle-Oganov structural fingerprint descriptor.",
        "many_body",
        {
            "species": _species(),
            "function": _enum(
                ("distance", "angle"),
                display_name="Fingerprint geometry",
                description="Selects whether the fingerprint is built from distances or angles.",
                default="distance",
            ),
            "n": _parameter(
                "integer",
                display_name="Grid point count",
                description="Number of grid points used to discretize the fingerprint.",
                default=50,
                minimum=2,
            ),
            "sigma": _parameter(
                "number",
                display_name="Gaussian width",
                description="Width of the Gaussian broadening applied to each fingerprint sample.",
                default=0.1,
                exclusiveMinimum=0.0,
            ),
            "r_cut": _parameter(
                "number",
                display_name="Cutoff radius",
                description="Maximum neighbor distance used to construct the fingerprint.",
                default=6.0,
                exclusiveMinimum=0.0,
                unit="Å",
            ),
            "geometry": _object(
                display_name="Geometry mapping",
                description="Optional explicit geometry configuration; normally derived from function.",
            ),
            "grid": _object(
                display_name="Discretization grid",
                description="Optional explicit grid configuration; normally derived from n, sigma, and r_cut.",
            ),
            "weighting": _object(
                display_name="Geometry weighting",
                description="Optional explicit distance/angle weighting configuration.",
            ),
            "periodic": _parameter(
                "boolean",
                display_name="Periodic calculation",
                description="Enables periodic boundary handling; the current implementation supports only true.",
                default=True,
                enum=[True],
            ),
            "normalize_gaussians": _parameter(
                "boolean",
                display_name="Normalize Gaussian basis",
                description="Normalizes each Gaussian used to broaden the fingerprint.",
                default=True,
            ),
            "normalization": _enum(
                ("none", "l2", "n_atoms", "valle_oganov"),
                display_name="Feature normalization",
                description="Normalization applied to the final Valle–Oganov feature vector.",
                default="valle_oganov",
            ),
        },
        periodicity=_PERIODIC_ONLY,
    ),
    "AtomicComposition": _info(
        "Atomic Composition",
        "Species composition vector descriptor.",
        "local",
        {
            "species": _species(),
            "per_system": _parameter(
                "boolean",
                display_name="Aggregate per structure",
                description="Returns one composition vector per structure instead of one vector per atom.",
                default=True,
            ),
        },
    ),
    "NeighborList": _info(
        "Neighbor List",
        "Neighbor pair displacement and distance descriptor.",
        "local",
        {
            "cutoff": _parameter(
                "number",
                display_name="Cutoff radius",
                description="Maximum center-neighbor distance included in the pair list.",
                default=6.0,
                exclusiveMinimum=0.0,
                unit="Å",
            ),
            "full_neighbor_list": _parameter(
                "boolean",
                display_name="Full neighbor list",
                description="Includes both ordered directions of each neighbor pair when enabled.",
                default=True,
            ),
            "self_pairs": _parameter(
                "boolean",
                display_name="Include self pairs",
                description="Includes zero-shift center-to-itself pairs when enabled.",
                default=False,
            ),
        },
        devices=("cpu", "cuda"),
    ),
    "SortedDistances": _info(
        "Sorted Distances",
        "Sorted neighbor distance descriptor.",
        "local",
        {
            "species": _species(),
            "cutoff": _parameter(
                "number",
                display_name="Cutoff radius",
                description="Maximum center-neighbor distance considered for sorting.",
                default=6.0,
                exclusiveMinimum=0.0,
                unit="Å",
            ),
            "max_neighbors": _parameter(
                "integer",
                display_name="Maximum neighbors",
                description="Maximum number of neighbor distances retained for each atom.",
                default=8,
                minimum=1,
            ),
            "separate_neighbor_types": _parameter(
                "boolean",
                display_name="Separate neighbor species",
                description="Keeps distance slots separated by neighbor chemical species when enabled.",
                default=True,
            ),
        },
    ),
    "SphericalExpansion": _info(
        "Spherical Expansion",
        "Species-resolved spherical density expansion descriptor.",
        "local",
        {
            "species": _species(),
            "cutoff": _parameter(
                "number",
                display_name="Cutoff radius",
                description="Maximum neighbor distance included in the density expansion.",
                default=6.0,
                exclusiveMinimum=0.0,
                unit="Å",
            ),
            "density_width": _parameter(
                "number",
                display_name="Density width",
                description="Gaussian width used to represent each neighbor in the smooth density.",
                default=0.3,
                exclusiveMinimum=0.0,
                unit="Å",
            ),
            "max_radial": _parameter(
                "integer",
                display_name="Maximum radial order",
                description="Number of radial basis functions retained in the expansion.",
                default=6,
                minimum=0,
            ),
            "max_angular": _parameter(
                "integer",
                display_name="Maximum angular order",
                description="Largest spherical-harmonic degree retained in the expansion.",
                default=4,
                minimum=0,
            ),
        },
        devices=("cpu", "cuda"),
    ),
    "SphericalExpansionByPair": _info(
        "Spherical Expansion by Pair",
        "Pair-level spherical density expansion descriptor.",
        "local",
        {
            "species": _species(),
            "cutoff": _parameter(
                "number",
                display_name="Cutoff radius",
                description="Maximum neighbor distance included in the pair density expansion.",
                default=6.0,
                exclusiveMinimum=0.0,
                unit="Å",
            ),
            "density_width": _parameter(
                "number",
                display_name="Density width",
                description="Gaussian width used to represent each neighbor in the smooth pair density.",
                default=0.3,
                exclusiveMinimum=0.0,
                unit="Å",
            ),
            "max_radial": _parameter(
                "integer",
                display_name="Maximum radial order",
                description="Number of radial basis functions retained in the pair expansion.",
                default=6,
                minimum=0,
            ),
            "max_angular": _parameter(
                "integer",
                display_name="Maximum angular order",
                description="Largest spherical-harmonic degree retained in the pair expansion.",
                default=4,
                minimum=0,
            ),
        },
    ),
    "SoapRadialSpectrum": _info(
        "SOAP Radial Spectrum",
        "Radial spectrum derived from a spherical density expansion.",
        "local",
        {
            "species": _species(),
            "cutoff": _parameter(
                "number",
                display_name="Cutoff radius",
                description="Maximum neighbor distance included in the spherical density.",
                default=6.0,
                exclusiveMinimum=0.0,
                unit="Å",
            ),
            "density_width": _parameter(
                "number",
                display_name="Density width",
                description="Gaussian width used to smooth each neighbor density contribution.",
                default=0.3,
                exclusiveMinimum=0.0,
                unit="Å",
            ),
            "max_radial": _parameter(
                "integer",
                display_name="Maximum radial order",
                description="Number of radial basis functions retained in the spectrum.",
                default=6,
                minimum=0,
            ),
            "max_angular": _parameter(
                "integer",
                display_name="Maximum angular order",
                description="Largest spherical-harmonic degree used to form the spectrum.",
                default=4,
                minimum=0,
            ),
        },
        devices=("cpu", "cuda"),
    ),
    "SoapPowerSpectrum": _info(
        "SOAP Power Spectrum",
        "Power spectrum derived from a spherical density expansion.",
        "local",
        {
            "species": _species(),
            "cutoff": _parameter(
                "number",
                display_name="Cutoff radius",
                description="Maximum neighbor distance included in the spherical density.",
                default=6.0,
                exclusiveMinimum=0.0,
                unit="Å",
            ),
            "density_width": _parameter(
                "number",
                display_name="Density width",
                description="Gaussian width used to smooth each neighbor density contribution.",
                default=0.3,
                exclusiveMinimum=0.0,
                unit="Å",
            ),
            "max_radial": _parameter(
                "integer",
                display_name="Maximum radial order",
                description="Number of radial basis functions retained in the power spectrum.",
                default=6,
                minimum=0,
            ),
            "max_angular": _parameter(
                "integer",
                display_name="Maximum angular order",
                description="Largest spherical-harmonic degree used to form the power spectrum.",
                default=4,
                minimum=0,
            ),
        },
        devices=("cpu", "cuda"),
    ),
    "LodeSphericalExpansion": _info(
        "LODE Spherical Expansion",
        "Long-distance equivariant spherical density expansion descriptor.",
        "local",
        {
            "species": _species(),
            "cutoff": _parameter(
                "number",
                display_name="Neighbor cutoff radius",
                description="Maximum short-range neighbor distance included in the density expansion.",
                default=6.0,
                exclusiveMinimum=0.0,
                unit="Å",
            ),
            "density_width": _parameter(
                "number",
                display_name="Density width",
                description="Gaussian width used to smooth each neighbor density contribution.",
                default=0.3,
                exclusiveMinimum=0.0,
                unit="Å",
            ),
            "max_radial": _parameter(
                "integer",
                display_name="Maximum radial order",
                description="Number of radial basis functions retained in the long-distance expansion.",
                default=6,
                minimum=0,
            ),
            "max_angular": _parameter(
                "integer",
                display_name="Maximum angular order",
                description="Largest spherical-harmonic degree retained in the long-distance expansion.",
                default=4,
                minimum=0,
            ),
            "k_cutoff": _parameter(
                "number",
                display_name="Reciprocal-space cutoff",
                description="Cutoff controlling the reciprocal-space part of the long-distance representation.",
                default=2.5,
                exclusiveMinimum=0.0,
            ),
            "exponent": _parameter(
                "integer",
                display_name="Long-distance exponent",
                description="Power-law exponent used for the long-distance interaction term.",
                default=1,
                minimum=1,
                maximum=9,
            ),
            "radial_radius": _parameter(
                "number",
                display_name="Radial basis radius",
                description="Radius used to construct the radial basis for the long-distance channel.",
                exclusiveMinimum=0.0,
                unit="Å",
            ),
        },
        periodicity=_PERIODIC_ONLY,
    ),
    "EAD": _info(
        "EAD",
        "Equivariant angular descriptor.",
        "rotational",
        {
            "parameters": _object(
                display_name="Angular parameters",
                description="EAD angular settings containing the maximum angular order L, widths eta, and centers Rs.",
                default={"L": 3, "eta": [0.05, 0.1, 0.5], "Rs": [0.0]},
            ),
            "Rc": _parameter(
                "number",
                display_name="Cutoff radius",
                description="Maximum neighbor distance included by the EAD descriptor.",
                default=6.0,
                exclusiveMinimum=0.0,
                unit="Å",
            ),
            "cutoff": _enum(
                ("cosine",),
                display_name="Cutoff function",
                description="Smooth radial cutoff function used by EAD.",
                default="cosine",
            ),
        },
    ),
    "SO3": _info(
        "SO3",
        "SO(3) rotationally invariant descriptor.",
        "rotational",
        {
            "nmax": _parameter(
                "integer",
                display_name="Maximum radial order",
                description="Number of radial basis channels used by the SO(3) descriptor.",
                default=3,
                minimum=1,
            ),
            "lmax": _parameter(
                "integer",
                display_name="Maximum angular order",
                description="Largest angular degree included in the SO(3) expansion.",
                default=3,
                minimum=0,
            ),
            "rcut": _parameter(
                "number",
                display_name="Cutoff radius",
                description="Maximum neighbor distance included in the SO(3) descriptor.",
                default=3.5,
                exclusiveMinimum=0.0,
                unit="Å",
            ),
            "alpha": _parameter(
                "number",
                display_name="Radial decay parameter",
                description="Positive radial decay parameter controlling the SO(3) basis.",
                default=2.0,
                exclusiveMinimum=0.0,
            ),
            "weight_on": _parameter(
                "boolean",
                display_name="Enable neighbor weighting",
                description="Enables the radial neighbor weighting used by the SO(3) construction.",
                default=False,
            ),
        },
    ),
    "SO4": _info(
        "SO4",
        "SO(4) rotationally invariant descriptor.",
        "rotational",
        {
            "lmax": _parameter(
                "integer",
                display_name="Maximum angular order",
                description="Largest hyperspherical angular degree included in the SO(4) expansion.",
                default=3,
                minimum=0,
            ),
            "rcut": _parameter(
                "number",
                display_name="Cutoff radius",
                description="Maximum neighbor distance included in the SO(4) descriptor.",
                default=3.5,
                exclusiveMinimum=0.0,
                unit="Å",
            ),
            "normalize_U": _parameter(
                "boolean",
                display_name="Normalize expansion basis",
                description="Normalizes the hyperspherical expansion coefficients before forming features.",
                default=False,
            ),
        },
    ),
    "SNAP": _info(
        "SNAP",
        "Spectral neighbor analysis potential bispectrum descriptor.",
        "rotational",
        {
            "weights": _object(
                display_name="Element weights",
                description="Optional per-element weights applied to neighbor contributions.",
                default={},
            ),
            "lmax": _parameter(
                "integer",
                display_name="Maximum angular order",
                description="Largest angular degree included in the SNAP bispectrum.",
                default=3,
                minimum=0,
            ),
            "rcut": _parameter(
                "number",
                display_name="Cutoff radius",
                description="Maximum neighbor distance included in the SNAP descriptor.",
                default=3.5,
                exclusiveMinimum=0.0,
                unit="Å",
            ),
            "normalize_U": _parameter(
                "boolean",
                display_name="Normalize expansion basis",
                description="Normalizes the hyperspherical expansion coefficients before forming features.",
                default=False,
            ),
        },
    ),
    "LBispectrum": _info(
        "L-Bispectrum",
        "Low-rank bispectrum descriptor with optional element profiles.",
        "rotational",
        {
            "twojmax": _parameter(
                "integer",
                display_name="Maximum bispectrum order",
                description="Angular resolution parameter controlling the bispectrum basis size.",
                default=3,
                minimum=0,
            ),
            "diagonal": _parameter(
                "integer",
                display_name="Diagonal truncation",
                description="Level of diagonal truncation applied to the bispectrum terms.",
                default=3,
                minimum=0,
                maximum=3,
            ),
            "rfac0": _parameter(
                "number",
                display_name="Radial mapping factor",
                description="Positive factor controlling the radial mapping used by the bispectrum.",
                default=0.99363,
                exclusiveMinimum=0.0,
            ),
            "rmin0": _parameter(
                "number",
                display_name="Minimum radial distance",
                description="Lower radial bound used by the bispectrum radial mapping.",
                default=0.0,
                minimum=0.0,
                unit="Å",
            ),
            "rcutfac": _parameter(
                "number",
                display_name="Radial cutoff factor",
                description="Positive factor scaling the effective radial cutoff.",
                default=1.0,
                exclusiveMinimum=0.0,
            ),
            "element_profile": _object(
                display_name="Element profiles",
                description="Optional combined per-element radius and weight profiles.",
            ),
            "element_radii": _object(
                display_name="Element radii",
                description="Optional per-element radial values used in the neighbor profile.",
            ),
            "weights": _object(
                display_name="Element weights",
                description="Optional per-element weights applied to neighbor contributions.",
            ),
            "rcut": _parameter(
                "number",
                display_name="Cutoff radius",
                description="Maximum neighbor distance included in the bispectrum descriptor.",
                default=3.5,
                exclusiveMinimum=0.0,
                unit="Å",
            ),
            "normalize_U": _parameter(
                "boolean",
                display_name="Normalize expansion basis",
                description="Normalizes the hyperspherical expansion coefficients before forming features.",
                default=False,
            ),
        },
    ),
    "MTP": _info(
        "MTP",
        "Moment tensor potential basis descriptor.",
        "local",
        {
            "species": _species(),
            "model": _model(),
            "min_dist": _parameter(
                "number",
                display_name="Minimum radial distance",
                description="Inner radial boundary of the MTP environment.",
                default=0.0,
                minimum=0.0,
                unit="Å",
            ),
            "max_dist": _parameter(
                "number",
                display_name="Maximum radial distance",
                description="Outer radial boundary of the MTP environment.",
                default=5.0,
                exclusiveMinimum=0.0,
                unit="Å",
            ),
            "r_cut": _parameter(
                "number",
                display_name="Interaction cutoff",
                description="Compatibility cutoff value used to derive the MTP maximum radial distance.",
                exclusiveMinimum=0.0,
                unit="Å",
            ),
            "radial_basis_size": _parameter(
                "integer",
                display_name="Radial basis size",
                description="Number of basis functions in each MTP radial channel.",
                default=4,
                minimum=1,
            ),
            "radial_funcs_count": _parameter(
                "integer",
                display_name="Radial function count",
                description="Number of independent radial function families used by the MTP basis.",
                default=1,
                minimum=1,
            ),
            "max_rank": _parameter(
                "integer",
                display_name="Maximum tensor rank",
                description="Largest moment-tensor rank included in the standalone MTP basis.",
                default=2,
                minimum=0,
                maximum=5,
            ),
            "radial_basis_type": _enum(
                ("RBChebyshev", "Chebyshev", "polynomial"),
                display_name="Radial basis family",
                description="Radial basis family used by the standalone MTP implementation.",
                default="RBChebyshev",
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
            "r_cut": _parameter(
                "number",
                display_name="Cutoff radius",
                description="Maximum neighbor distance included in the C00 and power-spectrum channels.",
                default=6.0,
                exclusiveMinimum=0.0,
                unit="Å",
            ),
            "n_radial": _parameter(
                "integer",
                display_name="Radial basis size",
                description="Number of radial basis functions used by the C00/PS representation.",
                default=8,
                minimum=1,
            ),
            "l_max": _parameter(
                "integer",
                display_name="Maximum angular order",
                description="Largest angular degree included in the power-spectrum channel.",
                default=4,
                minimum=0,
            ),
            "cutoff_function": _enum(
                ("bp", "mo", "rj", "wmc"),
                display_name="Cutoff function",
                description="Radial cutoff function used to smoothly limit neighbor contributions.",
                default="bp",
            ),
            "radial_sigma": _parameter(
                "number",
                display_name="Radial density width",
                description="Width used to smooth the radial density basis.",
                default=0.5,
                minimum=0.0,
                unit="Å",
            ),
            "include_radial": _parameter(
                "boolean",
                display_name="Include radial channel",
                description="Includes the C00 radial features in the output.",
                default=True,
            ),
            "include_angular": _parameter(
                "boolean",
                display_name="Include angular channel",
                description="Includes the power-spectrum angular features in the output.",
                default=True,
            ),
            "normalize_radial": _parameter(
                "boolean",
                display_name="Normalize radial channel",
                description="Normalizes the radial C00 features before combining channels.",
                default=False,
            ),
            "normalize_angular": _parameter(
                "boolean",
                display_name="Normalize angular channel",
                description="Normalizes the angular power-spectrum features before combining channels.",
                default=False,
            ),
            "super_vector": _parameter(
                "boolean",
                display_name="Use super vector",
                description="Uses the super-vector layout for combining species-resolved features.",
                default=False,
            ),
            "radial_weight": _parameter(
                "number",
                display_name="Radial channel weight",
                description="Non-negative weight applied to the radial feature channel.",
                default=1.0,
                minimum=0.0,
            ),
            "angular_weight": _parameter(
                "number",
                display_name="Angular channel weight",
                description="Non-negative weight applied to the angular feature channel.",
                default=1.0,
                minimum=0.0,
            ),
            "exclude_self_interaction": _parameter(
                "boolean",
                display_name="Exclude self interaction",
                description="Excludes the central atom's zero-distance self contribution when enabled.",
                default=True,
            ),
        },
    ),
    "NEP": _info(
        "NEP",
        "Neuroevolution potential descriptor backed by a local model.",
        "model_backed",
        {"model": _model(default="NEP")},
        devices=("cpu", "cuda"),
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
        devices=("cpu", "cuda"),
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
            "calibrate": _parameter(
                "boolean",
                display_name="Enable calibration",
                description="Applies the DPA4C calibration layer to the model-backed descriptor output.",
                default=True,
            ),
        },
        spin=True,
        charge_spin=True,
        devices=("cpu", "cuda"),
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
    descriptor_version: str = "1",
    execution_engine: str | None = None,
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
        descriptor_version=descriptor_version,
        execution_engine=execution_engine,
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
    _spec(
        "DPA4",
        _MODEL + "dpa4.descriptor:DPA4",
        "numpy",
        "atom",
        execution_engine=_DPA_EXECUTION_ENGINE,
    ),
    _spec(
        "DPA4C",
        _MODEL + "dpa4c.descriptor:DPA4C",
        "numpy",
        "atom",
        execution_engine=_DPA_EXECUTION_ENGINE,
    ),
)

builtin_registry = DescriptorRegistry(_BUILTIN_SPECS, frozen=True)
