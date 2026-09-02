#include "mdescriptor/cuda/backend.hpp"

#include "mdescriptor/cuda/batch.hpp"
#include "mdescriptor/cuda/local_descriptors.hpp"
#include "mdescriptor/cuda/neighbor_graph.hpp"
#include "mdescriptor/local_descriptors.hpp"
#include "mdescriptor/nep.hpp"
#include "mdescriptor/neighbor.hpp"
#include "mdescriptor/detail/batch.hpp"
#include "local_layout.hpp"

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace mdescriptor::cuda {
namespace {

using I32Array = py::array_t<std::int32_t, py::array::c_style | py::array::forcecast>;
using I64Array = py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>;
using F64Array = py::array_t<double, py::array::c_style | py::array::forcecast>;

struct BatchArrays {
    I32Array numbers;
    F64Array positions;
    F64Array cells;
    I32Array pbc;
    I64Array offsets;
    detail::StructureBatchView view;
};

BatchArrays arrays_from_batch(const py::object& value) {
    auto numbers = I32Array::ensure(value.attr("numbers"));
    auto positions = F64Array::ensure(value.attr("positions"));
    auto cells = F64Array::ensure(value.attr("cells"));
    auto pbc = I32Array::ensure(value.attr("pbc"));
    auto offsets = I64Array::ensure(value.attr("offsets"));
    if (!numbers || !positions || !cells || !pbc || !offsets
        || numbers.ndim() != 1 || positions.ndim() != 2
        || positions.shape(1) != 3 || cells.ndim() != 3
        || cells.shape(1) != 3 || cells.shape(2) != 3 || pbc.ndim() != 2
        || pbc.shape(1) != 3 || offsets.ndim() != 1 || offsets.shape(0) < 1) {
        throw std::invalid_argument("invalid StructureBatch array shapes");
    }
    const auto structures = static_cast<std::int64_t>(offsets.shape(0) - 1);
    const auto atoms = static_cast<std::int64_t>(numbers.shape(0));
    if (positions.shape(0) != numbers.shape(0) || cells.shape(0) != structures
        || pbc.shape(0) != structures) {
        throw std::invalid_argument("StructureBatch arrays have inconsistent lengths");
    }
    detail::StructureBatchView view{
        numbers.data(), positions.data(), cells.data(), pbc.data(), offsets.data(),
        structures, atoms,
    };
    detail::validate_batch(view);
    return {std::move(numbers), std::move(positions), std::move(cells),
            std::move(pbc), std::move(offsets), view};
}

template <typename Value>
Value option(const py::dict& options, const char* name, Value fallback) {
    const py::str key(name);
    if (!options.contains(key) || options[key].is_none()) {
        return fallback;
    }
    return py::cast<Value>(options[key]);
}

std::vector<std::int32_t> species_option(const py::dict& options) {
    const py::str key("species");
    if (!options.contains(key) || options[key].is_none()) {
        return {};
    }
    return py::cast<std::vector<std::int32_t>>(options[key]);
}

py::dict dpa_payload_option(const py::dict& options, const std::string& name) {
    const py::str key("_cuda_payload");
    if (!options.contains(key) || options[key].is_none()) {
        throw std::invalid_argument(
            name + " CUDA backend requires the validated private model payload");
    }
    try {
        return py::cast<py::dict>(options[key]);
    } catch (const py::cast_error&) {
        throw std::invalid_argument(name + " CUDA model payload is not a mapping");
    }
}

std::int64_t dpa_feature_count(const py::dict& options, const std::string& name) {
    const py::dict payload = dpa_payload_option(options, name);
    const py::str key("feature_count");
    if (!payload.contains(key)) {
        throw std::invalid_argument(name + " CUDA model payload is missing feature_count");
    }
    const auto count = py::cast<std::int64_t>(payload[key]);
    if (count <= 0) {
        throw std::invalid_argument(name + " CUDA model payload has an invalid feature count");
    }
    return count;
}

std::int64_t feature_count_for(
    const std::string& name,
    const py::dict& options) {
    if (name == "NeighborList") {
        return 4;
    }
    if (name == "DPA4" || name == "DPA4C") {
        return dpa_feature_count(options, name);
    }
    mdescriptor::LocalDescriptorOptions layout_options;
    layout_options.species = species_option(options);
    layout_options.max_radial = option(options, "max_radial", 6);
    layout_options.max_angular = option(options, "max_angular", 4);
    if (name == "SoapRadialSpectrum") {
        return detail::local_layout_feature_count(
            layout_options, mdescriptor::LocalDescriptorKind::SoapRadialSpectrum);
    }
    if (name == "SoapPowerSpectrum") {
        return detail::local_layout_feature_count(
            layout_options, mdescriptor::LocalDescriptorKind::SoapPowerSpectrum);
    }
    if (name == "SphericalExpansion") {
        return detail::local_layout_feature_count(
            layout_options, mdescriptor::LocalDescriptorKind::SphericalExpansion);
    }
    // The validation kernel supplies the canonical feature count for the
    // descriptors whose layout is owned by Python/native CPU code.  Keeping
    // that value private lets the CUDA plugin add a family without duplicating
    // every label/layout formula in this dispatch seam (matrix descriptors
    // may legitimately resolve their width only after seeing a batch).
    const py::str feature_key("_cuda_feature_count");
    if (options.contains(feature_key) && !options[feature_key].is_none()) {
        const auto value = py::cast<std::int64_t>(options[feature_key]);
        return value > 0 ? value : 0;
    }
    if (name == "AtomicComposition" || name == "SortedDistances"
        || name == "SphericalExpansionByPair" || name == "SOAP"
        || name == "SOAPTurbo" || name == "ACSF" || name == "ACE"
        || name == "LodeSphericalExpansion" || name == "CoulombMatrix"
        || name == "SineMatrix" || name == "EwaldSumMatrix" || name == "MBTR"
        || name == "LMBTR" || name == "ValleOganov" || name == "EAD"
        || name == "SO3" || name == "SO4" || name == "SNAP"
        || name == "LBispectrum" || name == "MTP" || name == "C00PSMLFF") {
        return 0;
    }
    throw std::invalid_argument("CUDA backend does not support this descriptor");
}

std::vector<std::int32_t> dpa_type_indices(
    const py::dict& options,
    const BatchArrays& arrays,
    const std::string& name) {
    const py::dict payload = dpa_payload_option(options, name);
    const py::str key("type_numbers");
    if (!payload.contains(key)) {
        throw std::invalid_argument(name + " CUDA model payload is missing type_numbers");
    }
    auto type_numbers = I32Array::ensure(payload[key]);
    if (!type_numbers || type_numbers.ndim() != 1 || type_numbers.shape(0) <= 0) {
        throw std::invalid_argument(name + " CUDA model payload has invalid type_numbers");
    }
    std::vector<std::int32_t> result(static_cast<std::size_t>(arrays.view.atoms), -1);
    for (std::int64_t atom = 0; atom < arrays.view.atoms; ++atom) {
        const auto number = arrays.view.numbers[atom];
        for (py::ssize_t type = 0; type < type_numbers.shape(0); ++type) {
            if (type_numbers.data()[type] == number) {
                result[static_cast<std::size_t>(atom)] = static_cast<std::int32_t>(type);
                break;
            }
        }
        if (result[static_cast<std::size_t>(atom)] < 0) {
            throw std::invalid_argument(
                name + " batch contains an element absent from the checkpoint type map: "
                + std::to_string(number));
        }
    }
    return result;
}

py::list generic_labels(const std::string& name, std::int64_t features);

py::list dpa_labels(const py::dict& options, const std::string& name, std::int64_t features) {
    const py::dict payload = dpa_payload_option(options, name);
    const py::str key("labels");
    if (payload.contains(key)) {
        try {
            return py::list(payload[key]);
        } catch (const py::error_already_set&) {
            throw std::invalid_argument(name + " CUDA model payload has invalid labels");
        }
    }
    return generic_labels(name, features);
}

bool cancelled(const py::object& control) {
    return !control.is_none() && control.attr("cancelled")().cast<bool>();
}

void reset_control(const py::object& control, std::int64_t total) {
    if (!control.is_none()) {
        control.attr("reset")(total);
    }
}

void mark_completed(const py::object& control) {
    if (!control.is_none()) {
        control.attr("mark_completed")();
    }
}

void check_cancelled(const py::object& control) {
    if (cancelled(control)) {
        throw std::runtime_error("descriptor computation cancelled");
    }
}

py::array double_array(const std::vector<double>& values, std::int64_t rows, std::int64_t columns) {
    py::array_t<double> result({
        static_cast<py::ssize_t>(rows), static_cast<py::ssize_t>(columns),
    });
    if (!values.empty()) {
        std::copy(values.begin(), values.end(), result.mutable_data());
    }
    return result;
}

py::list generic_labels(const std::string& name, std::int64_t features) {
    py::list labels;
    for (std::int64_t index = 0; index < features; ++index) {
        labels.append(name + ":" + std::to_string(index));
    }
    return labels;
}

py::list nep_labels(std::int64_t features) {
    py::list labels;
    for (std::int64_t index = 0; index < features; ++index) {
        labels.append("nep:q" + std::to_string(index + 1));
    }
    return labels;
}

py::array offsets_array(const std::vector<std::int64_t>& offsets) {
    py::array_t<std::int64_t> result(offsets.size());
    std::copy(offsets.begin(), offsets.end(), result.mutable_data());
    return result;
}

py::dict result_metadata(const std::string& name, const py::dict& options) {
    py::dict metadata;
    metadata["descriptor"] = name;
    metadata["backend"] = "mdescriptor-cuda";
    py::dict execution;
    execution["device"] = "cuda";
    const py::str execution_key("execution");
    if (options.contains(execution_key) && !options[execution_key].is_none()) {
        const py::dict options_execution = py::cast<py::dict>(options[execution_key]);
        execution["num_threads"] = options_execution.contains("num_threads")
            ? options_execution["num_threads"] : py::none();
    } else {
        execution["num_threads"] = py::none();
    }
    metadata["execution"] = execution;
    return metadata;
}

__global__ void write_neighbor_records(
    const std::int64_t* graph_offsets,
    const std::int32_t* graph_atoms,
    const std::int32_t* graph_shifts,
    const double* graph_displacements,
    const double* graph_distance2,
    const std::int64_t* output_offsets,
    std::int64_t atoms,
    bool full_neighbor_list,
    bool self_pairs,
    double* output) {
    const std::int64_t center = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (center >= atoms) {
        return;
    }
    std::int64_t row = output_offsets[center];
    const std::int64_t begin = graph_offsets[center];
    const std::int64_t end = graph_offsets[center + 1];
    for (std::int64_t index = begin; index < end; ++index) {
        const std::int32_t atom = graph_atoms[index];
        const std::int32_t shift_x = graph_shifts[index * 3 + 0];
        const std::int32_t shift_y = graph_shifts[index * 3 + 1];
        const std::int32_t shift_z = graph_shifts[index * 3 + 2];
        if (!self_pairs && atom == center && shift_x == 0 && shift_y == 0 && shift_z == 0) {
            continue;
        }
        if (!full_neighbor_list && atom < center) {
            continue;
        }
        double* record = output + row * 9;
        record[0] = static_cast<double>(center);
        record[1] = static_cast<double>(atom);
        record[2] = static_cast<double>(shift_x);
        record[3] = static_cast<double>(shift_y);
        record[4] = static_cast<double>(shift_z);
        record[5] = graph_displacements[index * 3 + 0];
        record[6] = graph_displacements[index * 3 + 1];
        record[7] = graph_displacements[index * 3 + 2];
        record[8] = sqrt(fmax(0.0, graph_distance2[index]));
        ++row;
    }
}

void check_cuda_backend(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) {
        return;
    }
    if (status == cudaErrorMemoryAllocation) {
        throw CudaOutOfMemory(operation);
    }
    if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver
        || status == cudaErrorSystemDriverMismatch) {
        throw CudaUnavailable(operation);
    }
    throw std::runtime_error(operation);
}

} // namespace

Backend::Backend(std::string name, py::dict options)
    : name_(std::move(name)), options_(std::move(options)),
      feature_count_(name_ == "NEP" ? 0 : feature_count_for(name_, options_)),
      context_(std::make_unique<CudaExecutionContext>(0)) {
    if (name_ == "NEP") {
        const auto model_path = option(options_, "model_path", std::string{});
        if (model_path.empty()) {
            throw std::invalid_argument("CUDA NEP backend requires model_path");
        }
        mdescriptor::NepOptions nep_options;
        nep_options.model_path = model_path;
        nep_options.model_digest = option(options_, "model_digest", std::string{});
        nep_options.num_threads = 0;
        mdescriptor::NepCalculator calculator(nep_options);
        const auto parameters = calculator.descriptor_parameters();
        feature_count_ = parameters.dimension;
        nep_model_ = std::make_unique<DeviceNepModel>(*context_, parameters);
    } else if (name_ == "DPA4C") {
        dpa4c_model_ = std::make_unique<DeviceDpa4cModel>(
            *context_, dpa_payload_option(options_, name_));
    } else if (name_ == "DPA4") {
        dpa4_model_ = std::make_unique<DeviceDpa4Model>(
            *context_, dpa_payload_option(options_, name_));
    }
}

Backend::~Backend() noexcept {
    close();
}

py::object Backend::compute(py::object batch_object, py::object control) {
    std::lock_guard<std::mutex> guard(compute_mutex_);
    if (closed_ || context_ == nullptr) {
        throw std::runtime_error("CUDA backend is closed");
    }
    BatchArrays arrays = arrays_from_batch(batch_object);
    reset_control(control, arrays.view.structures);
    check_cancelled(control);

    if (name_ != "NEP") {
        py::gil_scoped_release release;
        device_batch_.upload(*context_, arrays.view);
    }

    if (name_ == "NeighborList") {
        const double cutoff = option(options_, "cutoff", 6.0);
        const bool full_neighbor_list = option(options_, "full_neighbor_list", true);
        const bool self_pairs = option(options_, "self_pairs", false);
        std::vector<std::int64_t> row_offsets{0};
        std::vector<double> round_tripped;
        std::int64_t rows = 0;
        {
            py::gil_scoped_release release;
            const auto graph = mdescriptor::build_neighbor_graph(
                arrays.view, cutoff, nullptr, 0, true, false, true);
            device_graph_.upload(
                *context_, graph.offsets(), graph.atoms_data(), graph.shifts(),
                graph.displacements(), graph.distance2());
            std::vector<std::int64_t> atom_output_offsets(
                static_cast<std::size_t>(arrays.view.atoms) + 1, 0);
            for (std::int64_t center = 0; center < arrays.view.atoms; ++center) {
                const std::int64_t begin = graph.offsets()[static_cast<std::size_t>(center)];
                const std::int64_t end = graph.offsets()[static_cast<std::size_t>(center + 1)];
                std::int64_t count = 0;
                for (std::int64_t index = begin; index < end; ++index) {
                    const auto atom = graph.atoms_data()[static_cast<std::size_t>(index)];
                    const auto shift = graph.shifts().data() + static_cast<std::size_t>(index) * 3;
                    if (!self_pairs && atom == center && shift[0] == 0 && shift[1] == 0 && shift[2] == 0) {
                        continue;
                    }
                    if (!full_neighbor_list && atom < center) {
                        continue;
                    }
                    ++count;
                }
                atom_output_offsets[static_cast<std::size_t>(center + 1)]
                    = atom_output_offsets[static_cast<std::size_t>(center)] + count;
            }
            for (std::int64_t structure = 0; structure < arrays.view.structures; ++structure) {
                row_offsets.push_back(
                    atom_output_offsets[static_cast<std::size_t>(arrays.view.offsets[structure + 1])]);
            }
            rows = atom_output_offsets.back();
            std::int64_t* device_output_offsets = nullptr;
            try {
                if (arrays.view.atoms > 0) {
                    check_cuda_backend(
                        cudaMalloc(
                            reinterpret_cast<void**>(&device_output_offsets),
                            atom_output_offsets.size() * sizeof(std::int64_t)),
                        "could not allocate CUDA neighbor output offsets");
                    check_cuda_backend(
                        cudaMemcpyAsync(
                            device_output_offsets, atom_output_offsets.data(),
                            atom_output_offsets.size() * sizeof(std::int64_t),
                            cudaMemcpyHostToDevice, context_->stream()),
                        "could not upload CUDA neighbor output offsets");
                }
                if (rows > 0) {
                    double* output = context_->output_buffer(static_cast<std::size_t>(rows) * 9);
                    context_->synchronize();
                    constexpr unsigned int block_size = 256;
                    const auto blocks = static_cast<unsigned int>(
                        (arrays.view.atoms + block_size - 1) / block_size);
                    write_neighbor_records<<<blocks, block_size, 0, context_->stream()>>>(
                        device_graph_.offsets(), device_graph_.atoms(), device_graph_.shifts(),
                        device_graph_.displacements(), device_graph_.distance2(),
                        device_output_offsets, arrays.view.atoms, full_neighbor_list, self_pairs,
                        output);
                    check_cuda_backend(cudaGetLastError(), "CUDA neighbor kernel launch failed");
                    round_tripped = context_->download_output(static_cast<std::size_t>(rows) * 9);
                }
                if (device_output_offsets != nullptr) {
                    check_cuda_backend(cudaFree(device_output_offsets), "could not release CUDA neighbor output offsets");
                    device_output_offsets = nullptr;
                }
            } catch (...) {
                if (device_output_offsets != nullptr) {
                    (void)cudaFree(device_output_offsets);
                }
                throw;
            }
        }
        check_cancelled(control);
        for (std::int64_t structure = 0; structure < arrays.view.structures; ++structure) {
            mark_completed(control);
        }
        std::vector<double> pair_records;
        std::vector<double> values;
        pair_records.reserve(static_cast<std::size_t>(rows) * 5);
        values.reserve(static_cast<std::size_t>(rows) * 4);
        for (std::int64_t row = 0; row < rows; ++row) {
            const double* source = round_tripped.data() + static_cast<std::size_t>(row) * 9;
            for (int column = 0; column < 5; ++column) {
                pair_records.push_back(source[column]);
            }
            for (int column = 5; column < 9; ++column) {
                values.push_back(source[column]);
            }
        }
        py::dict result;
        result["values"] = double_array(values, rows, 4);
        result["level"] = "pair";
        result["row_offsets"] = offsets_array(row_offsets);
        result["pair_records"] = double_array(pair_records, rows, 5);
        py::list labels;
        labels.append("dx");
        labels.append("dy");
        labels.append("dz");
        labels.append("distance");
        result["labels"] = labels;
        result["metadata"] = result_metadata(name_, options_);
        return std::move(result);
    }

    if (name_ == "NEP") {
        if (arrays.view.atoms == 0) {
            for (std::int64_t structure = 0; structure < arrays.view.structures; ++structure) {
                mark_completed(control);
            }
            py::dict result;
            result["values"] = double_array({}, 0, feature_count_);
            result["level"] = "atom";
            result["row_offsets"] = arrays.offsets;
            result["labels"] = nep_labels(feature_count_);
            result["metadata"] = result_metadata(name_, options_);
            return std::move(result);
        }
        for (std::int64_t atom = 0; atom < arrays.view.atoms; ++atom) {
            if (!nep_model_->supports_atomic_number(arrays.view.numbers[atom])) {
                throw std::invalid_argument(
                    "structure contains an element not present in the NEP model: "
                    + std::to_string(arrays.view.numbers[atom]));
            }
        }
        std::vector<double> values;
        {
            py::gil_scoped_release release;
            const double cutoff = std::max(
                nep_model_->radial_cutoff_max(), nep_model_->angular_cutoff_max());
            // Keep the source batch resident on the device.  If a small
            // periodic cell needs replicas, the expanded atom array is also
            // generated by CUDA and later reduced there; otherwise the image
            // enumerator handles the original batch directly.  This leaves
            // mixed periodic/isolated batches on the same device graph path.
            device_batch_.upload(*context_, arrays.view);
            DeviceBatch* compute_batch = &device_batch_;
            detail::StructureBatchView compute_view = arrays.view;
            if (nep_expanded_batch_.expand_nep(
                    *context_, device_batch_, arrays.view, cutoff)) {
                compute_batch = &nep_expanded_batch_;
                compute_view = nep_expanded_batch_.metadata_view();
            }
            device_graph_.build_nep(
                *context_, *compute_batch, compute_view, cutoff);
            const auto computed = compute_nep(
                *context_, *compute_batch, device_graph_, *nep_model_, true);
            values = computed;
        }
        check_cancelled(control);
        for (std::int64_t structure = 0; structure < arrays.view.structures; ++structure) {
            mark_completed(control);
        }
        py::dict result;
        result["values"] = double_array(values, arrays.view.atoms, feature_count_);
        result["level"] = "atom";
        result["row_offsets"] = arrays.offsets;
        result["labels"] = nep_labels(feature_count_);
        result["metadata"] = result_metadata(name_, options_);
        return std::move(result);
    }

    if (name_ == "DPA4" || name_ == "DPA4C") {
        const auto type_indices = dpa_type_indices(options_, arrays, name_);
        std::vector<double> values;
        if (arrays.view.atoms > 0) {
            py::gil_scoped_release release;
            device_graph_.build_dpa(
                *context_, device_batch_, arrays.view,
                name_ == "DPA4C" ? dpa4c_model_->cutoff() : dpa4_model_->cutoff(),
                name_ == "DPA4",
                name_ == "DPA4");
            values = name_ == "DPA4C"
                ? dpa4c_model_->compute(*context_, device_batch_, device_graph_, type_indices)
                : dpa4_model_->compute(*context_, device_batch_, device_graph_, type_indices);
        }
        check_cancelled(control);
        for (std::int64_t structure = 0; structure < arrays.view.structures; ++structure) {
            mark_completed(control);
        }
        py::dict result;
        result["values"] = double_array(values, arrays.view.atoms, feature_count_);
        result["level"] = "atom";
        result["row_offsets"] = arrays.offsets;
        result["labels"] = dpa_labels(options_, name_, feature_count_);
        result["metadata"] = result_metadata(name_, options_);
        return std::move(result);
    }

    if (name_ == "AtomicComposition" || name_ == "SortedDistances"
        || name_ == "SphericalExpansionByPair" || name_ == "SOAP"
        || name_ == "SOAPTurbo" || name_ == "ACSF" || name_ == "ACE"
        || name_ == "LodeSphericalExpansion" || name_ == "CoulombMatrix"
        || name_ == "SineMatrix" || name_ == "EwaldSumMatrix" || name_ == "MBTR"
        || name_ == "LMBTR" || name_ == "ValleOganov" || name_ == "EAD"
        || name_ == "SO3" || name_ == "SO4" || name_ == "SNAP"
        || name_ == "LBispectrum" || name_ == "MTP" || name_ == "C00PSMLFF") {
        const auto result = compute_extended_descriptor(
            *context_, device_batch_, device_graph_, arrays.view,
            name_, options_, control);
        check_cancelled(control);
        for (std::int64_t structure = 0; structure < arrays.view.structures; ++structure) {
            mark_completed(control);
        }
        if (feature_count_ <= 0 && result.contains("values")) {
            const auto values = py::cast<py::array>(result["values"]);
            if (values.ndim() == 2) {
                feature_count_ = static_cast<std::int64_t>(values.shape(1));
            }
        }
        return result;
    }

    const auto species = species_option(options_);
    mdescriptor::LocalDescriptorOptions descriptor_options;
    descriptor_options.species = species;
    descriptor_options.cutoff = option(options_, "cutoff", 6.0);
    descriptor_options.density_width = option(options_, "density_width", 0.3);
    descriptor_options.max_radial = option(options_, "max_radial", 6);
    descriptor_options.max_angular = option(options_, "max_angular", 4);
    descriptor_options.num_threads = 0;
    const auto kind = name_ == "SoapRadialSpectrum"
        ? mdescriptor::LocalDescriptorKind::SoapRadialSpectrum
        : name_ == "SoapPowerSpectrum"
        ? mdescriptor::LocalDescriptorKind::SoapPowerSpectrum
        : mdescriptor::LocalDescriptorKind::SphericalExpansion;
    const auto features = mdescriptor::local_descriptor_feature_count(descriptor_options, kind);
    check_cancelled(control);
    std::vector<double> values;
    {
        py::gil_scoped_release release;
        const auto spherical_graph = mdescriptor::build_neighbor_graph(
            arrays.view, descriptor_options.cutoff, nullptr, 0, true, false, true);
        device_graph_.upload(
            *context_, spherical_graph.offsets(), spherical_graph.atoms_data(),
            spherical_graph.shifts(), spherical_graph.displacements(),
            spherical_graph.distance2());
        values = compute_local_descriptors(
            *context_, device_batch_, device_graph_, species,
            descriptor_options.cutoff, descriptor_options.density_width,
            descriptor_options.max_radial, descriptor_options.max_angular,
            static_cast<std::int32_t>(kind));
    }
    check_cancelled(control);
    for (std::int64_t structure = 0; structure < arrays.view.structures; ++structure) {
        mark_completed(control);
    }
    py::dict result;
    result["values"] = double_array(values, arrays.view.atoms, features);
    result["level"] = "atom";
    result["row_offsets"] = arrays.offsets;
    result["labels"] = generic_labels(name_, features);
    result["metadata"] = result_metadata(name_, options_);
    return std::move(result);
}

py::dict Backend::metadata() const {
    return result_metadata(name_, options_);
}

void Backend::close() noexcept {
    if (closed_) {
        return;
    }
    closed_ = true;
    device_graph_.clear();
    nep_expanded_batch_.clear();
    device_batch_.clear();
    dpa4c_model_.reset();
    dpa4_model_.reset();
    nep_model_.reset();
    if (context_ != nullptr) {
        context_->close();
        context_.reset();
    }
}

} // namespace mdescriptor::cuda
