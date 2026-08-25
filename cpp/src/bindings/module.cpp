#include "mdescriptor/descriptor.hpp"
#include "mdescriptor/extra.hpp"
#include "mdescriptor/local_descriptors.hpp"
#include "mdescriptor/nep.hpp"
#include "mdescriptor/neighbor.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <limits>
#include <string>
#include <vector>

namespace py = pybind11;
using mdescriptor::AcsfCalculator;
using mdescriptor::AcsfOptions;
using mdescriptor::ComputeControl;
using mdescriptor::MtpCalculator;
using mdescriptor::MtpOptions;
using mdescriptor::NepCalculator;
using mdescriptor::NepOptions;
using mdescriptor::SoapCalculator;
using mdescriptor::SoapOptions;
using mdescriptor::SoapTurboCalculator;
using mdescriptor::SoapTurboOptions;
using mdescriptor::C00PSMlffCalculator;
using mdescriptor::C00PSMlffOptions;
using mdescriptor::StructureBatchView;

namespace {

using I32Array = py::array_t<std::int32_t, py::array::c_style | py::array::forcecast>;
using I64Array = py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>;
using F64Array = py::array_t<double, py::array::c_style | py::array::forcecast>;

StructureBatchView view_batch(
    const I32Array& numbers,
    const F64Array& positions,
    const F64Array& cells,
    const I32Array& pbc,
    const I64Array& offsets
) {
    if (numbers.ndim() != 1 || positions.ndim() != 2 || positions.shape(1) != 3
        || cells.ndim() != 3 || cells.shape(1) != 3 || cells.shape(2) != 3
        || pbc.ndim() != 2 || pbc.shape(1) != 3 || offsets.ndim() != 1
        || offsets.shape(0) < 1) {
        throw std::invalid_argument("invalid StructureBatch array shapes");
    }
    const auto structures = static_cast<std::int64_t>(offsets.shape(0) - 1);
    if (positions.shape(0) != numbers.shape(0) || cells.shape(0) != structures || pbc.shape(0) != structures) {
        throw std::invalid_argument("StructureBatch arrays have inconsistent lengths");
    }
    auto numbers_info = numbers.request();
    auto positions_info = positions.request();
    auto cells_info = cells.request();
    auto pbc_info = pbc.request();
    auto offsets_info = offsets.request();
    return {
        static_cast<const std::int32_t*>(numbers_info.ptr),
        static_cast<const double*>(positions_info.ptr),
        static_cast<const double*>(cells_info.ptr),
        static_cast<const std::int32_t*>(pbc_info.ptr),
        static_cast<const std::int64_t*>(offsets_info.ptr),
        structures,
        static_cast<std::int64_t>(numbers.shape(0)),
    };
}

std::shared_ptr<ComputeControl> control_or_default(const std::shared_ptr<ComputeControl>& control) {
    return control ? control : std::make_shared<ComputeControl>();
}

py::array compute_soap_array(
    const SoapCalculator& calculator,
    const I32Array& numbers,
    const F64Array& positions,
    const F64Array& cells,
    const I32Array& pbc,
    const I64Array& offsets,
    const std::shared_ptr<ComputeControl>& control,
    std::int32_t num_threads,
    bool inner_average,
    bool outer_average
) {
    const auto batch = view_batch(numbers, positions, cells, pbc, offsets);
    const auto rows = (inner_average || outer_average) ? batch.structures : batch.atoms;
    py::array_t<double> output({rows, calculator.feature_count()});
    auto output_info = output.request();
    auto ctrl = control_or_default(control);
    {
        py::gil_scoped_release release;
        (void)num_threads;
        calculator.compute(batch, static_cast<double*>(output_info.ptr), ctrl);
    }
    return output;
}

py::array compute_soap_turbo_array(
    const SoapTurboCalculator& calculator,
    const I32Array& numbers,
    const F64Array& positions,
    const F64Array& cells,
    const I32Array& pbc,
    const I64Array& offsets,
    const std::shared_ptr<ComputeControl>& control
) {
    const auto batch = view_batch(numbers, positions, cells, pbc, offsets);
    py::array_t<double> output({batch.atoms, calculator.feature_count()});
    auto output_info = output.request();
    auto ctrl = control_or_default(control);
    {
        py::gil_scoped_release release;
        calculator.compute(batch, static_cast<double*>(output_info.ptr), ctrl);
    }
    return output;
}

py::array compute_acsf_array(
    const AcsfCalculator& calculator,
    const I32Array& numbers,
    const F64Array& positions,
    const F64Array& cells,
    const I32Array& pbc,
    const I64Array& offsets,
    const std::shared_ptr<ComputeControl>& control
) {
    const auto batch = view_batch(numbers, positions, cells, pbc, offsets);
    py::array_t<double> output({batch.atoms, calculator.feature_count()});
    auto output_info = output.request();
    auto ctrl = control_or_default(control);
    {
        py::gil_scoped_release release;
        calculator.compute(batch, static_cast<double*>(output_info.ptr), ctrl);
    }
    return output;
}

py::array compute_c00ps_mlff_array(
    const C00PSMlffCalculator& calculator,
    const I32Array& numbers,
    const F64Array& positions,
    const F64Array& cells,
    const I32Array& pbc,
    const I64Array& offsets,
    const std::shared_ptr<ComputeControl>& control
) {
    const auto batch = view_batch(numbers, positions, cells, pbc, offsets);
    py::array_t<double> output({batch.atoms, calculator.feature_count()});
    auto ctrl = control_or_default(control);
    {
        py::gil_scoped_release release;
        calculator.compute(batch, output.mutable_data(), ctrl);
    }
    return output;
}

py::array compute_mtp_array(
    const MtpCalculator& calculator,
    const I32Array& numbers,
    const F64Array& positions,
    const F64Array& cells,
    const I32Array& pbc,
    const I64Array& offsets,
    const std::shared_ptr<ComputeControl>& control
) {
    const auto batch = view_batch(numbers, positions, cells, pbc, offsets);
    py::array_t<double> output({batch.atoms, calculator.feature_count()});
    auto ctrl = control_or_default(control);
    {
        py::gil_scoped_release release;
        calculator.compute(batch, output.mutable_data(), ctrl);
    }
    return output;
}

py::array compute_nep_array(
    const NepCalculator& calculator,
    const I32Array& numbers,
    const F64Array& positions,
    const F64Array& cells,
    const I32Array& pbc,
    const I64Array& offsets,
    const std::shared_ptr<ComputeControl>& control
) {
    const auto batch = view_batch(numbers, positions, cells, pbc, offsets);
    py::array_t<double> output({batch.atoms, calculator.feature_count()});
    auto output_info = output.request();
    auto ctrl = control_or_default(control);
    {
        py::gil_scoped_release release;
        calculator.compute(batch, static_cast<double*>(output_info.ptr), ctrl);
    }
    return output;
}

py::array compute_coulomb_matrix_array(
    const I32Array& numbers,
    const F64Array& positions,
    const F64Array& cells,
    const I32Array& pbc,
    const I64Array& offsets,
    std::int64_t n_atoms_max,
    const std::string& permutation,
    double exponent,
    const std::shared_ptr<ComputeControl>& control
) {
    const auto batch = view_batch(numbers, positions, cells, pbc, offsets);
    if (n_atoms_max <= 0 || n_atoms_max > std::numeric_limits<std::int64_t>::max() / n_atoms_max) {
        throw std::invalid_argument("n_atoms_max must be a positive value");
    }
    const auto features = permutation == "eigenspectrum" ? n_atoms_max : n_atoms_max * n_atoms_max;
    py::array_t<double> output({batch.structures, features});
    auto output_info = output.request();
    std::fill(
        static_cast<double*>(output_info.ptr),
        static_cast<double*>(output_info.ptr) + batch.structures * features,
        0.0);
    auto ctrl = control_or_default(control);
    {
        py::gil_scoped_release release;
        mdescriptor::compute_coulomb_matrix(
            batch, n_atoms_max, permutation, exponent, static_cast<double*>(output_info.ptr), ctrl);
    }
    return output;
}

py::array compute_matrix_array(
    const I32Array& numbers,
    const F64Array& positions,
    const F64Array& cells,
    const I32Array& pbc,
    const I64Array& offsets,
    std::int64_t n_atoms_max,
    const std::string& permutation,
    double exponent,
    std::int32_t kind,
    double accuracy,
    double w,
    double r_cut,
    double g_cut,
    double a,
    const std::shared_ptr<ComputeControl>& control
) {
    const auto batch = view_batch(numbers, positions, cells, pbc, offsets);
    const auto columns = permutation == "eigenspectrum" ? n_atoms_max : n_atoms_max * n_atoms_max;
    py::array_t<double> output({batch.structures, columns});
    auto ctrl = control_or_default(control);
    {
        py::gil_scoped_release release;
        mdescriptor::compute_matrix(
            batch, n_atoms_max, permutation, exponent,
            static_cast<mdescriptor::MatrixKind>(kind), accuracy, w, r_cut, g_cut, a,
            output.mutable_data(), ctrl);
    }
    return output;
}

py::array compute_mbtr_array(
    const I32Array& numbers,
    const F64Array& positions,
    const F64Array& cells,
    const I32Array& pbc,
    const I64Array& offsets,
    const std::vector<std::int32_t>& species,
    std::int32_t geometry,
    std::int32_t weighting,
    std::int32_t normalization,
    double grid_min,
    double grid_max,
    double grid_sigma,
    int grid_n,
    bool normalize_gaussians,
    double scale,
    double threshold,
    double r_cut,
    double sharpness,
    bool local,
    const std::shared_ptr<ComputeControl>& control
) {
    const auto batch = view_batch(numbers, positions, cells, pbc, offsets);
    mdescriptor::MBTROptions options;
    options.species = species;
    options.geometry = static_cast<mdescriptor::MBTRGeometry>(geometry);
    options.weighting = static_cast<mdescriptor::MBTRWeighting>(weighting);
    options.normalization = static_cast<mdescriptor::MBTRNormalization>(normalization);
    options.grid_min = grid_min;
    options.grid_max = grid_max;
    options.grid_sigma = grid_sigma;
    options.grid_n = grid_n;
    options.normalize_gaussians = normalize_gaussians;
    options.scale = scale;
    options.threshold = threshold;
    options.r_cut = r_cut;
    options.sharpness = sharpness;
    options.local = local;
    const auto rows = local ? batch.atoms : batch.structures;
    const auto columns = mdescriptor::mbtr_feature_count(options);
    py::array_t<double> output({rows, columns});
    auto ctrl = control_or_default(control);
    {
        py::gil_scoped_release release;
        mdescriptor::compute_mbtr(batch, options, output.mutable_data(), ctrl);
    }
    return output;
}

py::array compute_ead_array(
    const I32Array& numbers,
    const F64Array& positions,
    const F64Array& cells,
    const I32Array& pbc,
    const I64Array& offsets,
    int max_degree,
    double cutoff,
    const std::vector<double>& eta,
    const std::vector<double>& rs,
    const std::shared_ptr<ComputeControl>& control
) {
    const auto batch = view_batch(numbers, positions, cells, pbc, offsets);
    mdescriptor::EadOptions options;
    options.max_degree = max_degree;
    options.cutoff = cutoff;
    options.eta = eta;
    options.rs = rs;
    const auto columns = mdescriptor::ead_feature_count(options);
    py::array_t<double> output({batch.atoms, columns});
    auto ctrl = control_or_default(control);
    {
        py::gil_scoped_release release;
        mdescriptor::compute_ead(batch, options, output.mutable_data(), ctrl);
    }
    return output;
}

py::array compute_rotational_descriptors_array(
    const I32Array& numbers,
    const F64Array& positions,
    const F64Array& cells,
    const I32Array& pbc,
    const I64Array& offsets,
    std::int32_t kind,
    int n_max,
    int l_max,
    double cutoff,
    double alpha,
    bool weight_on,
    bool normalize_u,
    double weight_scale,
    int twojmax,
    int diagonal,
    const std::shared_ptr<ComputeControl>& control,
    double rfac0,
    const std::vector<double>& neighbor_weights,
    double rmin0,
    double rcutfac,
    const std::vector<double>& neighbor_radii
) {
    const auto batch = view_batch(numbers, positions, cells, pbc, offsets);
    mdescriptor::RotationalDescriptorOptions options;
    options.kind = static_cast<mdescriptor::RotationalDescriptorKind>(kind);
    options.n_max = n_max;
    options.l_max = l_max;
    options.cutoff = cutoff;
    options.alpha = alpha;
    options.weight_on = weight_on;
    options.normalize_u = normalize_u;
    options.weight_scale = weight_scale;
    options.rfac0 = rfac0;
    options.rmin0 = rmin0;
    options.rcutfac = rcutfac;
    options.neighbor_weights = neighbor_weights;
    options.neighbor_radii = neighbor_radii;
    options.twojmax = twojmax;
    options.diagonal = diagonal;
    const auto columns = mdescriptor::rotational_feature_count(options);
    py::array_t<double> output({batch.atoms, columns});
    auto ctrl = control_or_default(control);
    {
        py::gil_scoped_release release;
        mdescriptor::compute_rotational_descriptors(batch, options, output.mutable_data(), ctrl);
    }
    return output;
}

py::array compute_atomic_composition_array(
    const I32Array& numbers,
    const F64Array& positions,
    const F64Array& cells,
    const I32Array& pbc,
    const I64Array& offsets,
    const std::vector<std::int32_t>& species,
    bool per_system,
    const std::shared_ptr<ComputeControl>& control
) {
    const auto batch = view_batch(numbers, positions, cells, pbc, offsets);
    const auto rows = per_system ? batch.structures : batch.atoms;
    py::array_t<double> output({rows, static_cast<std::int64_t>(species.size())});
    auto output_info = output.request();
    auto ctrl = control_or_default(control);
    {
        py::gil_scoped_release release;
        mdescriptor::compute_atomic_composition(
            batch, species, per_system, static_cast<double*>(output_info.ptr), ctrl);
    }
    return output;
}

py::array compute_sorted_distances_array(
    const I32Array& numbers,
    const F64Array& positions,
    const F64Array& cells,
    const I32Array& pbc,
    const I64Array& offsets,
    const std::vector<std::int32_t>& species,
    double cutoff,
    int max_neighbors,
    bool separate_neighbor_types,
    std::int32_t num_threads,
    const std::shared_ptr<ComputeControl>& control
) {
    const auto batch = view_batch(numbers, positions, cells, pbc, offsets);
    const auto columns = separate_neighbor_types
        ? static_cast<std::int64_t>(species.size()) * max_neighbors
        : max_neighbors;
    py::array_t<double> output({batch.atoms, columns});
    auto output_info = output.request();
    mdescriptor::LocalDescriptorOptions options;
    options.species = species;
    options.cutoff = cutoff;
    options.num_threads = num_threads;
    auto ctrl = control_or_default(control);
    {
        py::gil_scoped_release release;
        mdescriptor::compute_sorted_distances(
            batch, options, max_neighbors, separate_neighbor_types,
            static_cast<double*>(output_info.ptr), ctrl);
    }
    return output;
}

py::tuple compute_neighbor_list_array(
    const I32Array& numbers,
    const F64Array& positions,
    const F64Array& cells,
    const I32Array& pbc,
    const I64Array& offsets,
    double cutoff,
    bool full_neighbor_list,
    bool self_pairs,
    const std::shared_ptr<ComputeControl>& control
) {
    const auto batch = view_batch(numbers, positions, cells, pbc, offsets);
    auto ctrl = control_or_default(control);
    mdescriptor::DescriptorPairTable pairs;
    {
        py::gil_scoped_release release;
        pairs = mdescriptor::compute_neighbor_list(
            batch, cutoff, full_neighbor_list, self_pairs, ctrl);
    }
    py::array_t<double> values(std::vector<py::ssize_t>{
        static_cast<py::ssize_t>(pairs.values.size() / 9), 9});
    py::array_t<std::int64_t> pair_offsets(pairs.offsets.size());
    std::copy(pairs.values.begin(), pairs.values.end(), values.mutable_data());
    std::copy(pairs.offsets.begin(), pairs.offsets.end(), pair_offsets.mutable_data());
    return py::make_tuple(values, pair_offsets);
}

py::array compute_spherical_expansion_array(
    const I32Array& numbers,
    const F64Array& positions,
    const F64Array& cells,
    const I32Array& pbc,
    const I64Array& offsets,
    const std::vector<std::int32_t>& species,
    double cutoff,
    double density_width,
    int max_radial,
    int max_angular,
    std::int32_t kind,
    double k_cutoff,
    int exponent,
    double radial_radius,
    std::int32_t num_threads,
    const std::shared_ptr<ComputeControl>& control
) {
    const auto batch = view_batch(numbers, positions, cells, pbc, offsets);
    mdescriptor::LocalDescriptorOptions options;
    options.species = species;
    options.cutoff = cutoff;
    options.density_width = density_width;
    options.max_radial = max_radial;
    options.max_angular = max_angular;
    options.k_cutoff = k_cutoff;
    options.exponent = exponent;
    options.radial_radius = radial_radius > 0.0 ? radial_radius : cutoff;
    options.num_threads = num_threads;
    const auto descriptor_kind = static_cast<mdescriptor::LocalDescriptorKind>(kind);
    const auto features = mdescriptor::local_descriptor_feature_count(options, descriptor_kind);
    py::array_t<double> output({batch.atoms, features});
    auto output_info = output.request();
    auto ctrl = control_or_default(control);
    {
        py::gil_scoped_release release;
        mdescriptor::compute_spherical_expansion(
            batch, options, descriptor_kind, static_cast<double*>(output_info.ptr), ctrl);
    }
    return output;
}

py::tuple compute_spherical_expansion_by_pair_array(
    const I32Array& numbers,
    const F64Array& positions,
    const F64Array& cells,
    const I32Array& pbc,
    const I64Array& offsets,
    const std::vector<std::int32_t>& species,
    double cutoff,
    double density_width,
    int max_radial,
    int max_angular,
    std::int32_t num_threads,
    const std::shared_ptr<ComputeControl>& control
) {
    const auto batch = view_batch(numbers, positions, cells, pbc, offsets);
    mdescriptor::LocalDescriptorOptions options;
    options.species = species;
    options.cutoff = cutoff;
    options.density_width = density_width;
    options.max_radial = max_radial;
    options.max_angular = max_angular;
    options.num_threads = num_threads;
    const auto features = static_cast<std::int64_t>(
        (max_angular + 1) * (max_angular + 1) * (max_radial + 1));
    auto ctrl = control_or_default(control);
    mdescriptor::DescriptorPairTable pairs;
    {
        py::gil_scoped_release release;
        pairs = mdescriptor::compute_spherical_expansion_by_pair(batch, options, ctrl);
    }
    const auto rows = static_cast<py::ssize_t>(pairs.values.size() / static_cast<std::size_t>(9 + features));
    py::array_t<double> output(std::vector<py::ssize_t>{rows, static_cast<py::ssize_t>(features)});
    py::array_t<std::int64_t> pair_offsets(pairs.offsets.size());
    py::array_t<double> identifiers(std::vector<py::ssize_t>{rows, 9});
    auto output_ptr = output.mutable_data();
    auto identifiers_ptr = identifiers.mutable_data();
    for (py::ssize_t row = 0; row < rows; ++row) {
        const double* source = pairs.values.data() + static_cast<std::size_t>(row) * static_cast<std::size_t>(9 + features);
        std::copy(source, source + 9, identifiers_ptr + row * 9);
        std::copy(source + 9, source + 9 + features, output_ptr + row * features);
    }
    std::copy(pairs.offsets.begin(), pairs.offsets.end(), pair_offsets.mutable_data());
    return py::make_tuple(output, pair_offsets, identifiers);
}

py::tuple build_neighbor_graph_arrays(
    const I32Array& numbers,
    const F64Array& positions,
    const F64Array& cell,
    const I32Array& pbc,
    double cutoff
) {
    if (cell.ndim() != 2 || cell.shape(0) != 3 || cell.shape(1) != 3
        || pbc.ndim() != 1 || pbc.shape(0) != 3) {
        throw std::invalid_argument("invalid single-structure neighbor graph shapes");
    }
    if (positions.ndim() != 2 || positions.shape(1) != 3
        || positions.shape(0) != numbers.shape(0)) {
        throw std::invalid_argument("invalid single-structure neighbor positions");
    }
    const std::int64_t offsets_data[2] = {0, static_cast<std::int64_t>(numbers.shape(0))};
    StructureBatchView batch{
        static_cast<const std::int32_t*>(numbers.request().ptr),
        static_cast<const double*>(positions.request().ptr),
        static_cast<const double*>(cell.request().ptr),
        static_cast<const std::int32_t*>(pbc.request().ptr),
        offsets_data,
        1,
        static_cast<std::int64_t>(numbers.shape(0)),
    };
    mdescriptor::NeighborGraph graph;
    {
        py::gil_scoped_release release;
        graph = mdescriptor::build_neighbor_graph(batch, cutoff);
    }

    const auto& graph_offsets = graph.offsets();
    const auto& graph_atoms = graph.atoms_data();
    const auto& graph_shifts = graph.shifts();
    const auto& graph_displacements = graph.displacements();
    const auto& graph_distance2 = graph.distance2();
    py::array_t<std::int64_t> output_offsets(graph_offsets.size());
    py::array_t<std::int32_t> output_atoms(graph_atoms.size());
    py::array_t<std::int32_t> output_shifts({graph_atoms.size(), std::size_t(3)});
    py::array_t<double> output_displacements({graph_atoms.size(), std::size_t(3)});
    py::array_t<double> output_distance2(graph_distance2.size());
    std::copy(graph_offsets.begin(), graph_offsets.end(), output_offsets.mutable_data());
    std::copy(graph_atoms.begin(), graph_atoms.end(), output_atoms.mutable_data());
    std::copy(graph_shifts.begin(), graph_shifts.end(), output_shifts.mutable_data());
    std::copy(graph_displacements.begin(), graph_displacements.end(), output_displacements.mutable_data());
    std::copy(graph_distance2.begin(), graph_distance2.end(), output_distance2.mutable_data());
    return py::make_tuple(output_offsets, output_atoms, output_shifts, output_displacements, output_distance2);
}

} // namespace

PYBIND11_MODULE(_native, module) {
    module.doc() = "MDescriptor periodic descriptor kernels";
    py::register_exception<mdescriptor::CancelledError>(module, "CancelledError");

    py::class_<ComputeControl, std::shared_ptr<ComputeControl>>(module, "ComputeControl")
        .def(py::init<>())
        .def("cancel", &ComputeControl::cancel)
        .def("cancelled", &ComputeControl::cancelled)
        .def("completed", &ComputeControl::completed)
        .def("total", &ComputeControl::total)
        .def("mark_completed", &ComputeControl::mark_completed)
        .def("reset", &ComputeControl::reset);

    module.def("build_neighbor_graph", &build_neighbor_graph_arrays,
               py::arg("numbers"), py::arg("positions"), py::arg("cell"),
               py::arg("pbc"), py::arg("cutoff"));

    module.def("compute_coulomb_matrix", &compute_coulomb_matrix_array,
               py::arg("numbers"), py::arg("positions"), py::arg("cells"), py::arg("pbc"),
               py::arg("offsets"), py::arg("n_atoms_max"), py::arg("permutation"),
               py::arg("exponent") = 2.4, py::arg("control") = nullptr);

    module.def("compute_matrix", &compute_matrix_array,
               py::arg("numbers"), py::arg("positions"), py::arg("cells"), py::arg("pbc"),
               py::arg("offsets"), py::arg("n_atoms_max"), py::arg("permutation"),
               py::arg("exponent") = 2.4, py::arg("kind") = 0,
               py::arg("accuracy") = 1e-5, py::arg("w") = 1.0,
               py::arg("r_cut") = 0.0, py::arg("g_cut") = 0.0, py::arg("a") = 0.0,
               py::arg("control") = nullptr);

    module.def("compute_mbtr", &compute_mbtr_array,
               py::arg("numbers"), py::arg("positions"), py::arg("cells"), py::arg("pbc"),
               py::arg("offsets"), py::arg("species"), py::arg("geometry") = 1,
               py::arg("weighting") = 1, py::arg("normalization") = 0,
               py::arg("grid_min") = 0.0, py::arg("grid_max") = 6.0,
               py::arg("grid_sigma") = 0.1, py::arg("grid_n") = 50,
               py::arg("normalize_gaussians") = true, py::arg("scale") = 0.5,
               py::arg("threshold") = 1e-3, py::arg("r_cut") = 6.0,
               py::arg("sharpness") = 2.0, py::arg("local") = false,
               py::arg("control") = nullptr);

    module.def("compute_ead", &compute_ead_array,
               py::arg("numbers"), py::arg("positions"), py::arg("cells"), py::arg("pbc"),
               py::arg("offsets"), py::arg("max_degree") = 3, py::arg("cutoff") = 6.0,
               py::arg("eta"), py::arg("rs"), py::arg("control") = nullptr);

    module.def("compute_rotational_descriptors", &compute_rotational_descriptors_array,
               py::arg("numbers"), py::arg("positions"), py::arg("cells"), py::arg("pbc"),
               py::arg("offsets"), py::arg("kind") = 0, py::arg("n_max") = 3,
               py::arg("l_max") = 3, py::arg("cutoff") = 3.5, py::arg("alpha") = 2.0,
               py::arg("weight_on") = false, py::arg("normalize_u") = false,
               py::arg("weight_scale") = 1.0, py::arg("twojmax") = 3,
               py::arg("diagonal") = 3, py::arg("control") = nullptr,
               py::arg("rfac0") = 1.0, py::arg("neighbor_weights") = std::vector<double>{},
               py::arg("rmin0") = 0.0, py::arg("rcutfac") = 1.0,
               py::arg("neighbor_radii") = std::vector<double>{});

    module.def("compute_atomic_composition", &compute_atomic_composition_array,
               py::arg("numbers"), py::arg("positions"), py::arg("cells"), py::arg("pbc"),
               py::arg("offsets"), py::arg("species"), py::arg("per_system") = true,
               py::arg("control") = nullptr);

    module.def("compute_sorted_distances", &compute_sorted_distances_array,
               py::arg("numbers"), py::arg("positions"), py::arg("cells"), py::arg("pbc"),
               py::arg("offsets"), py::arg("species"), py::arg("cutoff"),
               py::arg("max_neighbors"), py::arg("separate_neighbor_types") = true,
               py::arg("num_threads") = 0, py::arg("control") = nullptr);

    module.def("compute_neighbor_list", &compute_neighbor_list_array,
               py::arg("numbers"), py::arg("positions"), py::arg("cells"), py::arg("pbc"),
               py::arg("offsets"), py::arg("cutoff"), py::arg("full_neighbor_list") = true,
               py::arg("self_pairs") = false, py::arg("control") = nullptr);

    module.def("compute_spherical_expansion", &compute_spherical_expansion_array,
               py::arg("numbers"), py::arg("positions"), py::arg("cells"), py::arg("pbc"),
               py::arg("offsets"), py::arg("species"), py::arg("cutoff") = 6.0,
               py::arg("density_width") = 0.3, py::arg("max_radial") = 6,
               py::arg("max_angular") = 4, py::arg("kind") = 0,
               py::arg("k_cutoff") = 2.5, py::arg("exponent") = 1,
               py::arg("radial_radius") = 0.0, py::arg("num_threads") = 0,
               py::arg("control") = nullptr);

    module.def("compute_spherical_expansion_by_pair", &compute_spherical_expansion_by_pair_array,
               py::arg("numbers"), py::arg("positions"), py::arg("cells"), py::arg("pbc"),
               py::arg("offsets"), py::arg("species"), py::arg("cutoff") = 6.0,
               py::arg("density_width") = 0.3, py::arg("max_radial") = 6,
               py::arg("max_angular") = 4, py::arg("num_threads") = 0,
               py::arg("control") = nullptr);

    py::class_<SoapCalculator>(module, "SoapCalculator")
        .def(py::init<SoapOptions>())
        .def_property_readonly("feature_count", &SoapCalculator::feature_count)
        .def_property_readonly("species", &SoapCalculator::species)
        .def("close", &SoapCalculator::close)
        .def("closed", &SoapCalculator::closed)
        .def("compute", &compute_soap_array,
             py::arg("numbers"), py::arg("positions"), py::arg("cells"), py::arg("pbc"),
             py::arg("offsets"), py::arg("control") = nullptr, py::arg("num_threads") = 0,
             py::arg("inner_average") = true, py::arg("outer_average") = false);

    py::class_<SoapTurboCalculator>(module, "SoapTurboCalculator")
        .def(py::init<SoapTurboOptions>())
        .def_property_readonly("feature_count", &SoapTurboCalculator::feature_count)
        .def_property_readonly("species", &SoapTurboCalculator::species)
        .def("close", &SoapTurboCalculator::close)
        .def("closed", &SoapTurboCalculator::closed)
        .def("compute", &compute_soap_turbo_array,
             py::arg("numbers"), py::arg("positions"), py::arg("cells"), py::arg("pbc"),
             py::arg("offsets"), py::arg("control") = nullptr);

    py::class_<AcsfCalculator>(module, "AcsfCalculator")
        .def(py::init<AcsfOptions>())
        .def_property_readonly("feature_count", &AcsfCalculator::feature_count)
        .def_property_readonly("species", &AcsfCalculator::species)
        .def("close", &AcsfCalculator::close)
        .def("closed", &AcsfCalculator::closed)
        .def("compute", &compute_acsf_array,
             py::arg("numbers"), py::arg("positions"), py::arg("cells"), py::arg("pbc"),
             py::arg("offsets"), py::arg("control") = nullptr);

    py::class_<C00PSMlffCalculator>(module, "C00PSMlffCalculator")
        .def(py::init<C00PSMlffOptions>())
        .def_property_readonly("feature_count", &C00PSMlffCalculator::feature_count)
        .def_property_readonly("species", &C00PSMlffCalculator::species)
        .def_property_readonly("radial_counts", &C00PSMlffCalculator::radial_counts)
        .def("close", &C00PSMlffCalculator::close)
        .def("closed", &C00PSMlffCalculator::closed)
        .def("compute", &compute_c00ps_mlff_array,
             py::arg("numbers"), py::arg("positions"), py::arg("cells"), py::arg("pbc"),
             py::arg("offsets"), py::arg("control") = nullptr);

    py::class_<MtpCalculator>(module, "MtpCalculator")
        .def(py::init<MtpOptions>())
        .def_property_readonly("feature_count", &MtpCalculator::feature_count)
        .def_property_readonly("species", &MtpCalculator::species)
        .def_property_readonly("official_model", &MtpCalculator::official_model)
        .def_property_readonly("official_mlip4", &MtpCalculator::official_mlip4)
        .def_property_readonly("official_format", &MtpCalculator::official_format)
        .def_property_readonly("official_alpha_moment_mapping", &MtpCalculator::official_alpha_moment_mapping)
        .def_property_readonly("official_min_dist", &MtpCalculator::official_min_dist)
        .def_property_readonly("official_max_dist", &MtpCalculator::official_max_dist)
        .def_property_readonly("official_radial_basis_size", &MtpCalculator::official_radial_basis_size)
        .def_property_readonly("official_radial_funcs_count", &MtpCalculator::official_radial_funcs_count)
        .def_property_readonly("official_radial_basis_type", &MtpCalculator::official_radial_basis_type)
        .def("close", &MtpCalculator::close)
        .def("closed", &MtpCalculator::closed)
        .def("compute", &compute_mtp_array,
             py::arg("numbers"), py::arg("positions"), py::arg("cells"), py::arg("pbc"),
             py::arg("offsets"), py::arg("control") = nullptr);

    py::class_<NepCalculator>(module, "NepCalculator")
        .def(py::init<NepOptions>())
        .def_property_readonly("feature_count", &NepCalculator::feature_count)
        .def_property_readonly("species", &NepCalculator::species)
        .def_property_readonly("model_path", &NepCalculator::model_path)
        .def_property_readonly("radial_cutoff", &NepCalculator::radial_cutoff)
        .def_property_readonly("angular_cutoff", &NepCalculator::angular_cutoff)
        .def_property_readonly("n_max_radial", &NepCalculator::n_max_radial)
        .def_property_readonly("n_max_angular", &NepCalculator::n_max_angular)
        .def_property_readonly("l_max", &NepCalculator::l_max)
        .def("close", &NepCalculator::close)
        .def("closed", &NepCalculator::closed)
        .def("compute", &compute_nep_array,
             py::arg("numbers"), py::arg("positions"), py::arg("cells"), py::arg("pbc"),
             py::arg("offsets"), py::arg("control") = nullptr);

    py::class_<SoapOptions>(module, "SoapOptions")
        .def(py::init<>())
        .def_readwrite("species", &SoapOptions::species)
        .def_readwrite("r_cut", &SoapOptions::r_cut)
        .def_readwrite("n_max", &SoapOptions::n_max)
        .def_readwrite("l_max", &SoapOptions::l_max)
        .def_readwrite("sigma", &SoapOptions::sigma)
        .def_readwrite("radial_basis", &SoapOptions::radial_basis)
        .def_readwrite("alphas", &SoapOptions::alphas)
        .def_readwrite("betas", &SoapOptions::betas)
        .def_readwrite("radial_grid", &SoapOptions::radial_grid)
        .def_readwrite("radial_weights", &SoapOptions::radial_weights)
        .def_readwrite("radial_values", &SoapOptions::radial_values)
        .def_readwrite("weighting_function", &SoapOptions::weighting_function)
        .def_readwrite("weighting_has_w0", &SoapOptions::weighting_has_w0)
        .def_readwrite("weighting_has_function", &SoapOptions::weighting_has_function)
        .def_readwrite("weighting_r0", &SoapOptions::weighting_r0)
        .def_readwrite("weighting_c", &SoapOptions::weighting_c)
        .def_readwrite("weighting_d", &SoapOptions::weighting_d)
        .def_readwrite("weighting_m", &SoapOptions::weighting_m)
        .def_readwrite("weighting_threshold", &SoapOptions::weighting_threshold)
        .def_readwrite("weighting_w0", &SoapOptions::weighting_w0)
        .def_readwrite("species_weights", &SoapOptions::species_weights)
        .def_readwrite("compression", &SoapOptions::compression)
        .def_readwrite("inner_average", &SoapOptions::inner_average)
        .def_readwrite("outer_average", &SoapOptions::outer_average)
        .def_readwrite("num_threads", &SoapOptions::num_threads);

    py::class_<C00PSMlffOptions>(module, "C00PSMlffOptions")
        .def(py::init<>())
        .def_readwrite("species", &C00PSMlffOptions::species)
        .def_readwrite("r_cut", &C00PSMlffOptions::r_cut)
        .def_readwrite("n_radial", &C00PSMlffOptions::n_radial)
        .def_readwrite("l_max", &C00PSMlffOptions::l_max)
        .def_readwrite("cutoff_function", &C00PSMlffOptions::cutoff_function)
        .def_readwrite("include_radial", &C00PSMlffOptions::include_radial)
        .def_readwrite("include_angular", &C00PSMlffOptions::include_angular)
        .def_readwrite("normalize_radial", &C00PSMlffOptions::normalize_radial)
        .def_readwrite("normalize_angular", &C00PSMlffOptions::normalize_angular)
        .def_readwrite("super_vector", &C00PSMlffOptions::super_vector)
        .def_readwrite("radial_weight", &C00PSMlffOptions::radial_weight)
        .def_readwrite("angular_weight", &C00PSMlffOptions::angular_weight)
        .def_readwrite("exclude_self_interaction", &C00PSMlffOptions::exclude_self_interaction)
        .def_readwrite("num_threads", &C00PSMlffOptions::num_threads);

    py::class_<SoapTurboOptions>(module, "SoapTurboOptions")
        .def(py::init<>())
        .def_readwrite("species", &SoapTurboOptions::species)
        .def_readwrite("alpha_max", &SoapTurboOptions::alpha_max)
        .def_readwrite("central_species", &SoapTurboOptions::central_species)
        .def_readwrite("atom_sigma_r", &SoapTurboOptions::atom_sigma_r)
        .def_readwrite("atom_sigma_r_scaling", &SoapTurboOptions::atom_sigma_r_scaling)
        .def_readwrite("atom_sigma_t", &SoapTurboOptions::atom_sigma_t)
        .def_readwrite("atom_sigma_t_scaling", &SoapTurboOptions::atom_sigma_t_scaling)
        .def_readwrite("amplitude_scaling", &SoapTurboOptions::amplitude_scaling)
        .def_readwrite("central_weight", &SoapTurboOptions::central_weight)
        .def_readwrite("l_max", &SoapTurboOptions::l_max)
        .def_readwrite("rcut_hard", &SoapTurboOptions::rcut_hard)
        .def_readwrite("rcut_soft", &SoapTurboOptions::rcut_soft)
        .def_readwrite("nf", &SoapTurboOptions::nf)
        .def_readwrite("radial_enhancement", &SoapTurboOptions::radial_enhancement)
        .def_readwrite("basis", &SoapTurboOptions::basis)
        .def_readwrite("compression", &SoapTurboOptions::compression)
        .def_readwrite("num_threads", &SoapTurboOptions::num_threads);

    py::class_<AcsfOptions>(module, "AcsfOptions")
        .def(py::init<>())
        .def_readwrite("species", &AcsfOptions::species)
        .def_readwrite("r_cut", &AcsfOptions::r_cut)
        .def_readwrite("g2_params", &AcsfOptions::g2_params)
        .def_readwrite("g3_params", &AcsfOptions::g3_params)
        .def_readwrite("g4_params", &AcsfOptions::g4_params)
        .def_readwrite("g5_params", &AcsfOptions::g5_params)
        .def_readwrite("n_g2", &AcsfOptions::n_g2)
        .def_readwrite("n_g3", &AcsfOptions::n_g3)
        .def_readwrite("n_g4", &AcsfOptions::n_g4)
        .def_readwrite("n_g5", &AcsfOptions::n_g5)
        .def_readwrite("num_threads", &AcsfOptions::num_threads);

    py::class_<MtpOptions>(module, "MtpOptions")
        .def(py::init<>())
        .def_readwrite("species", &MtpOptions::species)
        .def_readwrite("potential_path", &MtpOptions::potential_path)
        .def_readwrite("min_dist", &MtpOptions::min_dist)
        .def_readwrite("max_dist", &MtpOptions::max_dist)
        .def_readwrite("radial_basis_size", &MtpOptions::radial_basis_size)
        .def_readwrite("radial_funcs_count", &MtpOptions::radial_funcs_count)
        .def_readwrite("max_rank", &MtpOptions::max_rank)
        .def_readwrite("num_threads", &MtpOptions::num_threads);

    py::class_<NepOptions>(module, "NepOptions")
        .def(py::init<>())
        .def_readwrite("model_path", &NepOptions::model_path)
        .def_readwrite("num_threads", &NepOptions::num_threads);
}
