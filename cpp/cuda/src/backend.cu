#include "mdescriptor/cuda/backend.hpp"
#include "mdescriptor/cuda/descriptor_dispatch.hpp"
#include "mdescriptor/cuda/error.hpp"

#include "mdescriptor/cuda/batch.hpp"
#include "mdescriptor/cuda/local_descriptors.hpp"
#include "mdescriptor/cuda/neighbor_graph.hpp"
#include "mdescriptor/local_descriptors.hpp"
#include "mdescriptor/nep.hpp"
#include "mdescriptor/detail/batch.hpp"
#include "mdescriptor/detail/neighbor_filter.hpp"
#include "local_layout.hpp"

// pytypes.h only declares object::cast<T>; the definitions live here and a
// missing include surfaces only as an undefined-symbol link error.
#include <pybind11/cast.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <thrust/execution_policy.h>
#include <thrust/scan.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace py = pybind11;

// Shared extended-descriptor glue.  This header leaves namespace
// mdescriptor::cuda open for its includers, so it must stay the last include:
// everything below is already inside that namespace and the file closes it.
#include "extended_descriptors_common.cuh"

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
    const std::int64_t configured = feature_count_option(options, 0);
    if (is_extended_descriptor(name) || configured > 0) {
        return configured;
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
    return labels_option(options, name, features);
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
        throw mdescriptor::cuda::CudaCancelledError();
    }
}

py::list nep_labels(std::int64_t features) {
    py::list labels;
    for (std::int64_t index = 0; index < features; ++index) {
        labels.append("nep:q" + std::to_string(index + 1));
    }
    return labels;
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
        if (!self_pairs && exact_self_edge(center, atom, shift_x, shift_y, shift_z)) {
            continue;
        }
        if (!full_neighbor_list && !detail::keep_half_neighbor(
            center, atom, shift_x, shift_y, shift_z)) {
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

__global__ void count_filtered_neighbor_records(
    const std::int64_t* graph_offsets,
    const std::int32_t* graph_atoms,
    const std::int32_t* graph_shifts,
    std::int64_t atoms,
    bool full_neighbor_list,
    bool self_pairs,
    std::int64_t* output_offsets) {
    const std::int64_t center = static_cast<std::int64_t>(blockIdx.x)
        * blockDim.x + threadIdx.x;
    if (center >= atoms) {
        return;
    }
    const std::int64_t begin = graph_offsets[center];
    const std::int64_t end = graph_offsets[center + 1];
    std::int64_t count = 0;
    for (std::int64_t index = begin; index < end; ++index) {
        const std::int32_t atom = graph_atoms[index];
        const std::int32_t* shift = graph_shifts + index * 3;
        if (!self_pairs && exact_self_edge(center, atom, graph_shifts, index)) {
            continue;
        }
        if (!full_neighbor_list && !detail::keep_half_neighbor(
            center, atom, shift[0], shift[1], shift[2])) {
            continue;
        }
        ++count;
    }
    output_offsets[center + 1] = count;
}

} // namespace

Backend::Backend(std::string name, py::dict options)
    : name_(std::move(name)), options_(std::move(options)),
      feature_count_(name_ == "NEP" ? 0 : feature_count_for(name_, options_)),
      context_(std::make_unique<CudaExecutionContext>(0)) {
    if (name_ == "SO4" || name_ == "SNAP" || name_ == "LBispectrum") {
        rotational_plan_ = std::make_unique<RotationalPlanCache>();
    }
    if (name_ == "NEP") {
        const auto model_path = option(options_, "model_path", std::string{});
        if (model_path.empty()) {
            throw std::invalid_argument("CUDA NEP backend requires model_path");
        }
        mdescriptor::NepOptions nep_options;
        nep_options.model_path = model_path;
        nep_options.model_digest = option(options_, "model_digest", std::string{});
        if (options_.contains("model_data") && !options_["model_data"].is_none()) {
            nep_options.model_data = py::cast<std::string>(options_["model_data"]);
        }
        nep_options.num_threads = 0;
        mdescriptor::NepCalculator calculator(nep_options);
        if (option(options_, "_prediction", false)) {
            const auto parameters = calculator.prediction_parameters();
            feature_count_ = parameters.dimension;
            nep_model_ = std::make_unique<DeviceNepModel>(*context_, parameters);
        } else {
            const auto parameters = calculator.descriptor_parameters();
            feature_count_ = parameters.dimension;
            nep_model_ = std::make_unique<DeviceNepModel>(*context_, parameters);
        }
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
    std::unique_lock<std::mutex> guard(compute_mutex_, std::defer_lock);
    // A second Python thread must not hold the GIL while waiting for the
    // instance lock: the active compute needs that GIL to finish unwinding.
    {
        py::gil_scoped_release release;
        guard.lock();
    }
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
        std::vector<std::int64_t> atom_output_offsets(
            static_cast<std::size_t>(arrays.view.atoms) + 1U, 0);
        std::int64_t rows = 0;
        {
            py::gil_scoped_release release;
            std::int64_t* device_output_offsets = nullptr;
            try {
                device_graph_.build_dpa(
                    *context_, device_batch_, arrays.view, cutoff, true, false,
                    true, false, true, NeighborGraphOrdering::Canonical);
                if (arrays.view.atoms > 0) {
                    check_cuda(
                        cudaMalloc(
                            reinterpret_cast<void**>(&device_output_offsets),
                            atom_output_offsets.size() * sizeof(std::int64_t)),
                        "could not allocate CUDA neighbor output offsets");
                    check_cuda(
                        cudaMemsetAsync(
                            device_output_offsets, 0,
                            atom_output_offsets.size() * sizeof(std::int64_t),
                            context_->stream()),
                        "could not clear CUDA neighbor output offsets");
                    constexpr unsigned int block_size = 256;
                    const auto blocks = static_cast<unsigned int>(
                        (arrays.view.atoms + block_size - 1) / block_size);
                    count_filtered_neighbor_records<<<blocks, block_size, 0, context_->stream()>>>(
                        device_graph_.offsets(), device_graph_.atoms(), device_graph_.shifts(),
                        arrays.view.atoms, full_neighbor_list, self_pairs,
                        device_output_offsets);
                    check_cuda(
                        cudaGetLastError(), "CUDA neighbor filtering kernel launch failed");
                    const auto execution_policy = thrust::cuda::par.on(context_->stream());
                    thrust::inclusive_scan(
                        execution_policy, device_output_offsets + 1,
                        device_output_offsets + atom_output_offsets.size(),
                        device_output_offsets + 1);
                    check_cuda(
                        cudaMemcpyAsync(
                            atom_output_offsets.data(), device_output_offsets,
                            atom_output_offsets.size() * sizeof(std::int64_t),
                            cudaMemcpyDeviceToHost, context_->stream()),
                        "could not download CUDA neighbor output offsets");
                    context_->synchronize();
                    rows = atom_output_offsets.back();
                }
                for (std::int64_t structure = 0; structure < arrays.view.structures; ++structure) {
                    row_offsets.push_back(
                        atom_output_offsets[static_cast<std::size_t>(
                            arrays.view.offsets[structure + 1])]);
                }
                if (rows > 0) {
                    double* output = context_->output_buffer(static_cast<std::size_t>(rows) * 9);
                    constexpr unsigned int block_size = 256;
                    const auto blocks = static_cast<unsigned int>(
                        (arrays.view.atoms + block_size - 1) / block_size);
                    write_neighbor_records<<<blocks, block_size, 0, context_->stream()>>>(
                        device_graph_.offsets(), device_graph_.atoms(), device_graph_.shifts(),
                        device_graph_.displacements(), device_graph_.distance2(),
                        device_output_offsets, arrays.view.atoms, full_neighbor_list, self_pairs,
                        output);
                    check_cuda(cudaGetLastError(), "CUDA neighbor kernel launch failed");
                    round_tripped = context_->download_output(static_cast<std::size_t>(rows) * 9);
                }
                if (device_output_offsets != nullptr) {
                    check_cuda(
                        cudaFree(device_output_offsets),
                        "could not release CUDA neighbor output offsets");
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
        result["values"] = values_array(values, rows, 4);
        result["_mdescriptor_owned_values"] = true;
        result["level"] = "pair";
        result["row_offsets"] = i64_array(row_offsets);
        result["pair_records"] = values_array(pair_records, rows, 5);
        py::list labels;
        labels.append("dx");
        labels.append("dy");
        labels.append("dz");
        labels.append("distance");
        result["labels"] = labels;
        result["metadata"] = mdescriptor::cuda::metadata(options_, name_);
        return std::move(result);
    }

    if (name_ == "NEP") {
        if (arrays.view.atoms == 0) {
            for (std::int64_t structure = 0; structure < arrays.view.structures; ++structure) {
                mark_completed(control);
            }
            py::dict result;
            result["values"] = values_array({}, 0, feature_count_);
            result["_mdescriptor_owned_values"] = true;
            result["level"] = "atom";
            result["row_offsets"] = arrays.offsets;
            result["labels"] = nep_labels(feature_count_);
            result["metadata"] = mdescriptor::cuda::metadata(options_, name_);
            return std::move(result);
        }
        for (std::int64_t atom = 0; atom < arrays.view.atoms; ++atom) {
            if (!nep_model_->supports_atomic_number(arrays.view.numbers[atom])) {
                throw std::invalid_argument(
                    "structure contains an element not present in the NEP model: "
                    + std::to_string(arrays.view.numbers[atom]));
            }
        }
        py::array_t<double> values({
            static_cast<py::ssize_t>(arrays.view.atoms),
            static_cast<py::ssize_t>(feature_count_)});
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
            compute_nep_into(
                *context_, *compute_batch, device_graph_, *nep_model_,
                values.mutable_data());
        }
        check_cancelled(control);
        for (std::int64_t structure = 0; structure < arrays.view.structures; ++structure) {
            mark_completed(control);
        }
        py::dict result;
        result["values"] = std::move(values);
        result["_mdescriptor_owned_values"] = true;
        result["level"] = "atom";
        result["row_offsets"] = arrays.offsets;
        result["labels"] = nep_labels(feature_count_);
        result["metadata"] = mdescriptor::cuda::metadata(options_, name_);
        return std::move(result);
    }

    if (name_ == "DPA4" || name_ == "DPA4C") {
        const auto type_indices = dpa_type_indices(options_, arrays, name_);
        py::array_t<double> values({
            static_cast<py::ssize_t>(arrays.view.atoms),
            static_cast<py::ssize_t>(feature_count_)});
        if (arrays.view.atoms > 0) {
            py::gil_scoped_release release;
            device_graph_.build_dpa(
                *context_, device_batch_, arrays.view,
                name_ == "DPA4C" ? dpa4c_model_->cutoff() : dpa4_model_->cutoff(),
                name_ == "DPA4",
                name_ == "DPA4");
            if (name_ == "DPA4C") {
                dpa4c_model_->compute_into(
                    *context_, device_batch_, device_graph_, type_indices,
                    values.mutable_data());
            } else {
                dpa4_model_->compute_into(
                    *context_, device_batch_, device_graph_, type_indices,
                    values.mutable_data());
            }
        }
        check_cancelled(control);
        for (std::int64_t structure = 0; structure < arrays.view.structures; ++structure) {
            mark_completed(control);
        }
        py::dict result;
        result["values"] = std::move(values);
        result["_mdescriptor_owned_values"] = true;
        result["level"] = "atom";
        result["row_offsets"] = arrays.offsets;
        result["labels"] = dpa_labels(options_, name_, feature_count_);
        result["metadata"] = mdescriptor::cuda::metadata(options_, name_);
        return std::move(result);
    }

    if (is_extended_descriptor(name_)) {
        auto result = compute_extended_descriptor(
            *context_, device_batch_, device_graph_, arrays.view,
            name_, options_, control, rotational_plan_.get());
        if (result.contains("values")) {
            result["_mdescriptor_owned_values"] = true;
        }
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
    py::array_t<double> values({
        static_cast<py::ssize_t>(arrays.view.atoms),
        static_cast<py::ssize_t>(features)});
    {
        py::gil_scoped_release release;
        device_graph_.build_dpa(
            *context_, device_batch_, arrays.view, descriptor_options.cutoff,
            true, false, true, false, true, NeighborGraphOrdering::Canonical);
        compute_local_descriptors_into(
            *context_, device_batch_, device_graph_, species,
            descriptor_options.cutoff, descriptor_options.density_width,
            descriptor_options.max_radial, descriptor_options.max_angular,
            static_cast<std::int32_t>(kind), values.mutable_data());
    }
    check_cancelled(control);
    for (std::int64_t structure = 0; structure < arrays.view.structures; ++structure) {
        mark_completed(control);
    }
    py::dict result;
    result["values"] = std::move(values);
    result["_mdescriptor_owned_values"] = true;
    result["level"] = "atom";
    result["row_offsets"] = arrays.offsets;
    result["labels"] = labels_option(options_, name_, features);
    result["metadata"] = mdescriptor::cuda::metadata(options_, name_);
    return std::move(result);
}

py::object Backend::predict(py::object batch_object, py::object control) {
    std::unique_lock<std::mutex> guard(compute_mutex_, std::defer_lock);
    {
        py::gil_scoped_release release;
        guard.lock();
    }
    if (closed_ || context_ == nullptr) {
        throw std::runtime_error("CUDA backend is closed");
    }
    if (!option(options_, "_prediction", false)
        || (name_ != "NEP" && name_ != "DPA4C")) {
        throw std::invalid_argument("CUDA backend is not a predictor");
    }
    BatchArrays arrays = arrays_from_batch(batch_object);
    reset_control(control, arrays.view.structures);
    check_cancelled(control);
    py::array_t<double> energy(arrays.view.structures);
    py::array_t<double> atom_energy(arrays.view.atoms);
    py::array_t<double> forces({arrays.view.atoms, static_cast<std::int64_t>(3)});
    std::fill_n(energy.mutable_data(), arrays.view.structures, 0.0);
    std::fill_n(atom_energy.mutable_data(), arrays.view.atoms, 0.0);
    std::fill_n(forces.mutable_data(), arrays.view.atoms * 3, 0.0);

    if (arrays.view.atoms > 0 && name_ == "NEP") {
        for (std::int64_t atom = 0; atom < arrays.view.atoms; ++atom) {
            if (!nep_model_->supports_atomic_number(arrays.view.numbers[atom])) {
                throw std::invalid_argument(
                    "structure contains an element not present in the NEP model: "
                    + std::to_string(arrays.view.numbers[atom]));
            }
        }
        py::gil_scoped_release release;
        const double cutoff = nep_model_->neighbor_cutoff();
        device_batch_.upload(*context_, arrays.view);
        bool use_direct_image_graph = arrays.view.atoms <= 4096;
        for (std::int64_t structure = 0;
             use_direct_image_graph && structure < arrays.view.structures;
             ++structure) {
            const std::int64_t structure_atoms =
                arrays.view.offsets[structure + 1] - arrays.view.offsets[structure];
            use_direct_image_graph = structure_atoms <= 256;
        }
        use_direct_image_graph = use_direct_image_graph
            && nep_expanded_batch_.requires_nep_expansion(arrays.view, cutoff);
        if (use_direct_image_graph) {
            // Keep structures that would require replicas as image edges.
            // Larger structures and cells that need no replicas use the
            // NEP cell-list path.
            device_graph_.build_dpa(
                *context_, device_batch_, arrays.view, cutoff,
                false, false, false, false, true, NeighborGraphOrdering::Unsorted);
            predict_nep_into(*context_, device_batch_, device_graph_, *nep_model_,
                arrays.view, energy.mutable_data(), atom_energy.mutable_data(),
                forces.mutable_data());
        } else {
            DeviceBatch* compute_batch = &device_batch_;
            detail::StructureBatchView compute_view = arrays.view;
            if (nep_expanded_batch_.expand_nep(*context_, device_batch_, arrays.view, cutoff)) {
                compute_batch = &nep_expanded_batch_;
                compute_view = nep_expanded_batch_.metadata_view();
            }
            device_graph_.build_nep(*context_, *compute_batch, compute_view, cutoff);
            predict_nep_into(*context_, *compute_batch, device_graph_, *nep_model_,
                arrays.view, energy.mutable_data(), atom_energy.mutable_data(),
                forces.mutable_data());
        }
    } else if (arrays.view.atoms > 0 && name_ == "DPA4C") {
        const auto type_indices = dpa_type_indices(options_, arrays, name_);
        py::gil_scoped_release release;
        device_batch_.upload(*context_, arrays.view);
        device_graph_.build_dpa(*context_, device_batch_, arrays.view,
            dpa4c_model_->cutoff(), false, false);
        dpa4c_model_->predict_into(*context_, device_batch_, device_graph_,
            type_indices, energy.mutable_data(), atom_energy.mutable_data(),
            forces.mutable_data());
    }
    check_cancelled(control);
    for (std::int64_t structure = 0; structure < arrays.view.structures; ++structure) {
        mark_completed(control);
    }
    return py::make_tuple(std::move(energy), std::move(atom_energy), std::move(forces));
}

py::dict Backend::metadata() const {
    return mdescriptor::cuda::metadata(options_, name_);
}

void Backend::close() noexcept {
    std::unique_lock<std::mutex> guard(compute_mutex_, std::defer_lock);
    // Explicit close() is bound with the GIL released. The conditional also
    // keeps destruction safe when a holder is released on a GIL-owning thread.
    if (PyGILState_Check()) {
        py::gil_scoped_release release;
        guard.lock();
    } else {
        guard.lock();
    }
    if (!closed_) {
        closed_ = true;
        device_graph_.clear();
        nep_expanded_batch_.clear();
        device_batch_.clear();
        rotational_plan_.reset();
        dpa4c_model_.reset();
        dpa4_model_.reset();
        nep_model_.reset();
        if (context_ != nullptr) {
            context_->close();
            context_.reset();
        }
    }
}

} // namespace mdescriptor::cuda
