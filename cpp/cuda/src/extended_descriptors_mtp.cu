#include "extended_dispatch.hpp"
#include "extended_descriptors_common.cuh"

__global__ void mtp4_cuda_kernel(
    const I32* numbers,
    const I64* graph_offsets,
    const I32* graph_atoms,
    const I32* graph_shifts,
    const double* graph_displacements,
    const double* graph_distance2,
    const I32* species,
    int species_count,
    int radial_kind,
    int radial_basis_size,
    int radial_funcs_count,
    double min_dist,
    double max_dist,
    double max_dist_sq,
    double max_dist_sq_minus_eps,
    double exp_ratio,
    double radial_zeroth,
    const double* radial_recursive,
    const double* radial_vdw_params,
    const double* model_parameters,
    double radial_scaling,
    const I32* moments,
    I64 moment_count,
    const I32* eval_kinds,
    const I32* eval_linear_ids,
    const double* eval_linear_coefficients,
    I64 eval_count,
    const I64* eval_product_offsets,
    const I32* eval_product_left,
    const I32* eval_product_right,
    const double* eval_product_coefficients,
    const I32* scalar_output_ids,
    I64 features,
    I64 atoms,
    double* eval_workspace,
    double* output) {
    const I64 center = static_cast<I64>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (center >= atoms || eval_count <= 0 || eval_count > 65536) return;
    const int center_type = species_index(numbers[center], species, species_count);
    if (center_type < 0) return;
    double* raw = eval_workspace + center * eval_count;
    for (I64 index = 0; index < eval_count; ++index) raw[index] = 0.0;
    const I64 radial_count = static_cast<I64>(species_count) * species_count
        * radial_funcs_count * radial_basis_size;
    double radial_basis[64]{};
    double radial_values[32]{};
    const I64 begin = graph_offsets[center];
    const I64 end = graph_offsets[center + 1];
    for (I64 edge = begin; edge < end; ++edge) {
        const I32 atom = graph_atoms[edge];
        if (atom == center && graph_shifts[edge * 3] == 0
            && graph_shifts[edge * 3 + 1] == 0 && graph_shifts[edge * 3 + 2] == 0) continue;
        const double r_sq = fmax(0.0, graph_distance2[edge]);
        const double distance = sqrt(r_sq);
        if (distance <= 0.0 || distance > max_dist) continue;
        const int neighbor_type = species_index(numbers[atom], species, species_count);
        if (neighbor_type < 0) continue;
        mtp4_radial_basis_device(
            r_sq, radial_kind, radial_basis_size, min_dist, max_dist, max_dist_sq,
            max_dist_sq_minus_eps, exp_ratio, radial_zeroth, radial_recursive,
            radial_vdw_params, center_type, neighbor_type, radial_basis);
        const I64 pair_offset = (static_cast<I64>(center_type) * species_count + neighbor_type)
            * radial_funcs_count * radial_basis_size;
        for (int function = 0; function < radial_funcs_count; ++function) {
            double value = 0.0;
            for (int radial = 0; radial < radial_basis_size; ++radial) {
                value += model_parameters[pair_offset + function * radial_basis_size + radial]
                    * radial_basis[radial];
            }
            radial_values[function] = value * radial_scaling;
        }
        const double distance_power_limit = distance;
        const double* displacement = graph_displacements + edge * 3;
        for (I64 id = 0; id < moment_count; ++id) {
            const I32* moment = moments + id * 4;
            if (moment[3] > 1 || moment[0] < 0 || moment[0] >= radial_funcs_count) continue;
            double coordinate_product = 1.0;
            double coordinate_x = 1.0;
            double coordinate_y = 1.0;
            double coordinate_z = 1.0;
            double distance_power = 1.0;
            const int degree = moment[1] + moment[2] + moment[3];
            for (int power = 0; power < moment[1]; ++power) coordinate_x *= displacement[0];
            for (int power = 0; power < moment[2]; ++power) coordinate_y *= displacement[1];
            for (int power = 0; power < moment[3]; ++power) coordinate_z *= displacement[2];
            for (int power = 0; power < degree; ++power) distance_power *= distance_power_limit;
            coordinate_product = coordinate_x * coordinate_y * coordinate_z;
            raw[id] += radial_values[moment[0]] * coordinate_product / distance_power;
        }
    }
    for (I64 id = 0; id < eval_count; ++id) {
        const int kind = eval_kinds[id];
        if (kind == 0) continue;
        if (kind == 1) {
            double value = 0.0;
            for (int term = 0; term < 3; ++term) {
                const int dependency = eval_linear_ids[id * 3 + term];
                if (dependency >= 0) value += eval_linear_coefficients[id * 3 + term] * raw[dependency];
            }
            raw[id] = value;
        } else {
            double value = 0.0;
            for (I64 term = eval_product_offsets[id]; term < eval_product_offsets[id + 1]; ++term) {
                value += eval_product_coefficients[term]
                    * raw[eval_product_left[term]] * raw[eval_product_right[term]];
            }
            raw[id] = value;
        }
    }
    double* target = output + center * features;
    for (I64 feature = 0; feature < features; ++feature) {
        target[feature] = raw[scalar_output_ids[feature]];
    }
    (void)radial_count;
}

py::dict compute_mtp4_descriptor(
    CudaExecutionContext& context,
    DeviceBatch& batch,
    DeviceNeighborGraph& graph,
    const detail::StructureBatchView& host_batch,
    const py::dict& options) {
    const py::str payload_key("_cuda_payload");
    if (!options.contains(payload_key) || options[payload_key].is_none()) {
        throw std::invalid_argument("MLIP-4 MTP CUDA backend requires its evaluator payload");
    }
    const py::dict payload = py::cast<py::dict>(options[payload_key]);
    const auto species = species_option(options);
    const auto model_species = py::cast<std::vector<I32>>(payload["model_species"]);
    const auto model_parameters = vector_child(payload, "model_parameters");
    const auto radial_recursive = vector_child(payload, "radial_recursive");
    const auto radial_vdw_params = vector_child(payload, "radial_vdw_params");
    const auto moments = py::cast<std::vector<I32>>(payload["moments"]);
    const auto eval_kinds = py::cast<std::vector<I32>>(payload["eval_kinds"]);
    const auto eval_linear_ids = py::cast<std::vector<I32>>(payload["eval_linear_ids"]);
    const auto eval_linear_coefficients = vector_child(payload, "eval_linear_coefficients");
    const auto eval_product_offsets = py::cast<std::vector<I64>>(payload["eval_product_offsets"]);
    const auto eval_product_left = py::cast<std::vector<I32>>(payload["eval_product_left"]);
    const auto eval_product_right = py::cast<std::vector<I32>>(payload["eval_product_right"]);
    const auto eval_product_coefficients = vector_child(payload, "eval_product_coefficients");
    const auto scalar_output_ids = py::cast<std::vector<I32>>(payload["scalar_output_ids"]);
    const I64 features = feature_count_option(options, 0);
    const int radial_kind = py::cast<int>(payload["radial_kind"]);
    const int radial_basis_size = py::cast<int>(payload["radial_basis_size"]);
    const int radial_funcs_count = py::cast<int>(payload["radial_funcs_count"]);
    const double min_dist = py::cast<double>(payload["radial_min_dist"]);
    const double max_dist = py::cast<double>(payload["radial_max_dist"]);
    const double radial_scaling = py::cast<double>(payload["radial_scaling"]);
    const double radial_zeroth = py::cast<double>(payload["radial_zeroth"]);
    const double radial_exp_ratio = py::cast<double>(payload["radial_exp_ratio"]);
    const double radial_maxdist_sq = py::cast<double>(payload["radial_maxdist_sq"]);
    const double radial_maxdist_sq_minus_eps = py::cast<double>(payload["radial_maxdist_sq_minus_eps"]);
    const I64 moment_count = static_cast<I64>(moments.size() / 4U);
    const I64 eval_count = static_cast<I64>(eval_kinds.size());
    const I64 radial_parameter_count = static_cast<I64>(species.size()) * species.size()
        * radial_funcs_count * radial_basis_size;
    if (model_species.size() != species.size() || species.empty()
        || radial_kind < 0 || radial_kind > 3 || radial_basis_size <= 0
        || radial_basis_size > 64 || radial_funcs_count <= 0 || radial_funcs_count > 32
        || !std::isfinite(min_dist) || !std::isfinite(max_dist)
        || min_dist < 0.0 || max_dist <= min_dist
        || model_parameters.size() < static_cast<std::size_t>(radial_parameter_count)
        || moments.size() % 4U != 0 || moment_count <= 0 || eval_count <= 0
        || eval_linear_ids.size() != static_cast<std::size_t>(eval_count * 3)
        || eval_linear_coefficients.size() != static_cast<std::size_t>(eval_count * 3)
        || eval_product_offsets.size() != static_cast<std::size_t>(eval_count + 1)
        || eval_product_left.size() != eval_product_right.size()
        || eval_product_left.size() != eval_product_coefficients.size()
        || eval_product_offsets.back() != static_cast<I64>(eval_product_left.size())
        || scalar_output_ids.size() != static_cast<std::size_t>(features)) {
        throw std::invalid_argument("invalid MLIP-4 MTP CUDA evaluator payload");
    }
    if (radial_kind == 0 && radial_recursive.size() < static_cast<std::size_t>(3 * (radial_basis_size - 1))) {
        throw std::invalid_argument("MLIP-4 Cinf radial evaluator payload is incomplete");
    }
    if (radial_kind == 3 && radial_vdw_params.size() < species.size() + 2) {
        throw std::invalid_argument("MLIP-4 damped radial evaluator payload is incomplete");
    }
    for (const I32 id : scalar_output_ids) {
        if (id < 0 || id >= eval_count) throw std::invalid_argument("MLIP-4 scalar output id is out of range");
    }
    graph.build_dpa(context, batch, host_batch, max_dist, true, false, false);

    DeviceBuffer<I32> d_species;
    DeviceBuffer<double> d_parameters;
    DeviceBuffer<double> d_recursive;
    DeviceBuffer<double> d_vdw_params;
    DeviceBuffer<I32> d_moments;
    DeviceBuffer<I32> d_eval_kinds;
    DeviceBuffer<I32> d_eval_linear_ids;
    DeviceBuffer<double> d_eval_linear_coefficients;
    DeviceBuffer<I64> d_eval_product_offsets;
    DeviceBuffer<I32> d_eval_product_left;
    DeviceBuffer<I32> d_eval_product_right;
    DeviceBuffer<double> d_eval_product_coefficients;
    DeviceBuffer<I32> d_scalar_output_ids;
    d_species.upload(species.data(), species.size(), context.stream(), "could not upload MLIP-4 species");
    d_parameters.upload(model_parameters.data(), model_parameters.size(), context.stream(), "could not upload MLIP-4 radial parameters");
    d_recursive.upload(radial_recursive.data(), radial_recursive.size(), context.stream(), "could not upload MLIP-4 radial recurrence");
    d_vdw_params.upload(radial_vdw_params.data(), radial_vdw_params.size(), context.stream(), "could not upload MLIP-4 damped radial parameters");
    d_moments.upload(moments.data(), moments.size(), context.stream(), "could not upload MLIP-4 moments");
    d_eval_kinds.upload(eval_kinds.data(), eval_kinds.size(), context.stream(), "could not upload MLIP-4 evaluator kinds");
    d_eval_linear_ids.upload(eval_linear_ids.data(), eval_linear_ids.size(), context.stream(), "could not upload MLIP-4 linear evaluator ids");
    d_eval_linear_coefficients.upload(eval_linear_coefficients.data(), eval_linear_coefficients.size(), context.stream(), "could not upload MLIP-4 linear evaluator coefficients");
    d_eval_product_offsets.upload(eval_product_offsets.data(), eval_product_offsets.size(), context.stream(), "could not upload MLIP-4 product evaluator offsets");
    d_eval_product_left.upload(eval_product_left.data(), eval_product_left.size(), context.stream(), "could not upload MLIP-4 product evaluator left ids");
    d_eval_product_right.upload(eval_product_right.data(), eval_product_right.size(), context.stream(), "could not upload MLIP-4 product evaluator right ids");
    d_eval_product_coefficients.upload(eval_product_coefficients.data(), eval_product_coefficients.size(), context.stream(), "could not upload MLIP-4 product evaluator coefficients");
    d_scalar_output_ids.upload(scalar_output_ids.data(), scalar_output_ids.size(), context.stream(), "could not upload MLIP-4 scalar output ids");
    const std::size_t size = static_cast<std::size_t>(batch.atoms()) * static_cast<std::size_t>(features);
    double* output = context.output_buffer(size);
    auto* workspace = static_cast<double*>(context.workspace_buffer(
        static_cast<std::size_t>(batch.atoms()) * static_cast<std::size_t>(eval_count) * sizeof(double)));
    if (size > 0) {
        check_cuda(cudaMemsetAsync(output, 0, size * sizeof(double), context.stream()),
            "could not clear MLIP-4 MTP output");
        constexpr unsigned block_size = 64;
        mtp4_cuda_kernel<<<static_cast<unsigned>((batch.atoms() + block_size - 1) / block_size),
            block_size, 0, context.stream()>>>(
            batch.numbers(), graph.offsets(), graph.atoms(), graph.shifts(), graph.displacements(), graph.distance2(),
            d_species.get(), static_cast<int>(species.size()), radial_kind, radial_basis_size,
            radial_funcs_count, min_dist, max_dist, radial_maxdist_sq,
            radial_maxdist_sq_minus_eps, radial_exp_ratio, radial_zeroth, d_recursive.get(),
            d_vdw_params.get(), d_parameters.get(), radial_scaling, d_moments.get(), moment_count,
            d_eval_kinds.get(), d_eval_linear_ids.get(), d_eval_linear_coefficients.get(), eval_count,
            d_eval_product_offsets.get(), d_eval_product_left.get(), d_eval_product_right.get(),
            d_eval_product_coefficients.get(), d_scalar_output_ids.get(), features, batch.atoms(),
            workspace, output);
        check_cuda(cudaGetLastError(), "MLIP-4 MTP CUDA kernel launch failed");
    }
    const auto values = download_output_with_gil_release(context, size);
    return atom_result(values, batch.atoms(), features, "MTP", options, false,
        std::vector<I64>(host_batch.offsets, host_batch.offsets + host_batch.structures + 1));
}

__device__ void mtp2_radial_basis_device(
    double distance,
    double min_dist,
    double max_dist,
    int basis_size,
    double scaling,
    bool repulsive,
    double* output) {
    const double radius = repulsive ? fmax(distance, min_dist) : distance;
    const double ksi = (2.0 * radius - (min_dist + max_dist)) / (max_dist - min_dist);
    const double edge = radius - max_dist;
    output[0] = edge * edge;
    if (basis_size > 1) output[1] = ksi * edge * edge;
    for (int index = 2; index < basis_size; ++index) {
        output[index] = 2.0 * ksi * output[index - 1] - output[index - 2];
    }
    for (int index = 0; index < basis_size; ++index) output[index] *= scaling;
}

__global__ void mtp2_cuda_kernel(
    const I32* numbers,
    const I64* graph_offsets,
    const I32* graph_atoms,
    const I32* graph_shifts,
    const double* graph_displacements,
    const double* graph_distance2,
    const I32* species,
    int species_count,
    int radial_basis_size,
    int radial_funcs_count,
    int alpha_moments_count,
    double min_dist,
    double max_dist,
    double scaling,
    bool repulsive,
    const double* radial_coefficients,
    const I32* alpha_basic,
    I64 alpha_basic_count,
    const I32* alpha_times,
    I64 alpha_times_count,
    const I32* moment_mapping,
    I64 features,
    I64 atoms,
    double* workspace,
    double* output) {
    const I64 center = static_cast<I64>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (center >= atoms || alpha_moments_count <= 0 || alpha_moments_count > 65536) return;
    const int center_type = species_index(numbers[center], species, species_count);
    if (center_type < 0) return;
    double* moments = workspace + center * alpha_moments_count;
    for (int index = 0; index < alpha_moments_count; ++index) moments[index] = 0.0;
    double rb[128]{};
    double radial_values[64]{};
    const I64 begin = graph_offsets[center];
    const I64 end = graph_offsets[center + 1];
    const I64 pair_stride = static_cast<I64>(radial_funcs_count) * radial_basis_size;
    for (I64 edge = begin; edge < end; ++edge) {
        const I32 atom = graph_atoms[edge];
        if (atom == center && graph_shifts[edge * 3] == 0
            && graph_shifts[edge * 3 + 1] == 0 && graph_shifts[edge * 3 + 2] == 0) continue;
        const double distance = sqrt(fmax(0.0, graph_distance2[edge]));
        if (distance <= 0.0 || distance > max_dist) continue;
        const int neighbor_type = species_index(numbers[atom], species, species_count);
        if (neighbor_type < 0) continue;
        mtp2_radial_basis_device(
            distance, min_dist, max_dist, radial_basis_size, scaling, repulsive, rb);
        const I64 pair_offset = (static_cast<I64>(center_type) * species_count + neighbor_type)
            * pair_stride;
        for (int function = 0; function < radial_funcs_count; ++function) {
            double value = 0.0;
            for (int radial = 0; radial < radial_basis_size; ++radial) {
                value += radial_coefficients[pair_offset + function * radial_basis_size + radial]
                    * rb[radial];
            }
            radial_values[function] = value;
        }
        const double* displacement = graph_displacements + edge * 3;
        for (I64 index = 0; index < alpha_basic_count; ++index) {
            const I32* alpha = alpha_basic + index * 4;
            if (alpha[0] < 0 || alpha[0] >= radial_funcs_count) continue;
            const int degree = alpha[1] + alpha[2] + alpha[3];
            double x_power = 1.0;
            double y_power = 1.0;
            double z_power = 1.0;
            double distance_power = 1.0;
            for (int power = 0; power < alpha[1]; ++power) x_power *= displacement[0];
            for (int power = 0; power < alpha[2]; ++power) y_power *= displacement[1];
            for (int power = 0; power < alpha[3]; ++power) z_power *= displacement[2];
            for (int power = 0; power < degree; ++power) distance_power *= distance;
            moments[index] += radial_values[alpha[0]] * x_power * y_power * z_power / distance_power;
        }
    }
    for (I64 index = 0; index < alpha_times_count; ++index) {
        const I32* alpha = alpha_times + index * 4;
        if (alpha[0] >= 0 && alpha[0] < alpha_moments_count
            && alpha[1] >= 0 && alpha[1] < alpha_moments_count
            && alpha[3] >= 0 && alpha[3] < alpha_moments_count) {
            moments[alpha[3]] += static_cast<double>(alpha[2])
                * moments[alpha[0]] * moments[alpha[1]];
        }
    }
    double* target = output + center * features;
    target[0] = 1.0;
    for (I64 index = 0; index < features - 1; ++index) {
        const int mapped = moment_mapping[index];
        if (mapped >= 0 && mapped < alpha_moments_count) target[index + 1] = moments[mapped];
    }
}

py::dict compute_mtp2_descriptor(
    CudaExecutionContext& context,
    DeviceBatch& batch,
    DeviceNeighborGraph& graph,
    const detail::StructureBatchView& host_batch,
    const py::dict& options) {
    const py::str payload_key("_cuda_payload");
    const py::dict payload = py::cast<py::dict>(options[payload_key]);
    const auto species = species_option(options);
    const auto radial_coefficients = vector_child(payload, "radial_coefficients");
    const auto alpha_basic = py::cast<std::vector<I32>>(payload["alpha_index_basic"]);
    const auto alpha_times = py::cast<std::vector<I32>>(payload["alpha_index_times"]);
    const auto moment_mapping = py::cast<std::vector<I32>>(payload["alpha_moment_mapping"]);
    const int species_count = py::cast<int>(payload["species_count"]);
    const int radial_basis_size = py::cast<int>(payload["radial_basis_size"]);
    const int radial_funcs_count = py::cast<int>(payload["radial_funcs_count"]);
    const int alpha_moments_count = py::cast<int>(payload["alpha_moments_count"]);
    const double min_dist = py::cast<double>(payload["radial_min_dist"]);
    const double max_dist = py::cast<double>(payload["radial_max_dist"]);
    const double scaling = py::cast<double>(payload["scaling"]);
    const std::string basis_type = option(payload, "radial_basis_type", std::string("RBChebyshev"));
    const bool repulsive = basis_type == "RBChebyshev_repuls";
    const I64 features = feature_count_option(options, 0);
    if (species.empty() || species_count != static_cast<int>(species.size())
        || radial_basis_size <= 0 || radial_basis_size > 128 || radial_funcs_count <= 0
        || radial_funcs_count > 64 || alpha_moments_count <= 0
        || alpha_basic.size() % 4U != 0 || alpha_times.size() % 4U != 0
        || moment_mapping.size() + 1 != static_cast<std::size_t>(features)
        || radial_coefficients.size() < static_cast<std::size_t>(species_count * species_count
            * radial_funcs_count * radial_basis_size)
        || min_dist < 0.0 || max_dist <= min_dist || features <= 0) {
        throw std::invalid_argument("invalid MLIP-2 MTP CUDA evaluator payload");
    }
    graph.build_dpa(context, batch, host_batch, max_dist, true, false, false);
    DeviceBuffer<I32> d_species;
    DeviceBuffer<double> d_radial_coefficients;
    DeviceBuffer<I32> d_alpha_basic;
    DeviceBuffer<I32> d_alpha_times;
    DeviceBuffer<I32> d_moment_mapping;
    d_species.upload(species.data(), species.size(), context.stream(), "could not upload MLIP-2 species");
    d_radial_coefficients.upload(radial_coefficients.data(), radial_coefficients.size(), context.stream(), "could not upload MLIP-2 radial coefficients");
    d_alpha_basic.upload(alpha_basic.data(), alpha_basic.size(), context.stream(), "could not upload MLIP-2 basic indices");
    d_alpha_times.upload(alpha_times.data(), alpha_times.size(), context.stream(), "could not upload MLIP-2 product indices");
    d_moment_mapping.upload(moment_mapping.data(), moment_mapping.size(), context.stream(), "could not upload MLIP-2 moment mapping");
    const std::size_t size = static_cast<std::size_t>(batch.atoms()) * static_cast<std::size_t>(features);
    double* output = context.output_buffer(size);
    auto* workspace = static_cast<double*>(context.workspace_buffer(
        static_cast<std::size_t>(batch.atoms()) * static_cast<std::size_t>(alpha_moments_count)
        * sizeof(double)));
    if (size > 0) {
        check_cuda(cudaMemsetAsync(output, 0, size * sizeof(double), context.stream()),
            "could not clear MLIP-2 MTP output");
        constexpr unsigned block_size = 64;
        mtp2_cuda_kernel<<<static_cast<unsigned>((batch.atoms() + block_size - 1) / block_size),
            block_size, 0, context.stream()>>>(
            batch.numbers(), graph.offsets(), graph.atoms(), graph.shifts(), graph.displacements(),
            graph.distance2(), d_species.get(), species_count, radial_basis_size,
            radial_funcs_count, alpha_moments_count, min_dist, max_dist, scaling, repulsive,
            d_radial_coefficients.get(), d_alpha_basic.get(), static_cast<I64>(alpha_basic.size() / 4U),
            d_alpha_times.get(), static_cast<I64>(alpha_times.size() / 4U), d_moment_mapping.get(),
            features, batch.atoms(), workspace, output);
        check_cuda(cudaGetLastError(), "MLIP-2 MTP CUDA kernel launch failed");
    }
    const auto values = download_output_with_gil_release(context, size);
    return atom_result(values, batch.atoms(), features, "MTP", options, false,
        std::vector<I64>(host_batch.offsets, host_batch.offsets + host_batch.structures + 1));
}


} // namespace

py::dict compute_extended_mtp(
    CudaExecutionContext& context,
    DeviceBatch& batch,
    DeviceNeighborGraph& graph,
    const detail::StructureBatchView& host_batch,
    const std::string& name,
    const py::dict& options,
    const py::object& control,
    RotationalPlanCache* rotational_plan) {
    (void)control;
    (void)rotational_plan;
    const py::str payload_key("_cuda_payload");
    if (options.contains(payload_key) && !options[payload_key].is_none()) {
        const py::dict payload = py::cast<py::dict>(options[payload_key]);
        if (payload.contains("model_parameters")) {
            return compute_mtp4_descriptor(context, batch, graph, host_batch, options);
        }
        if (payload.contains("alpha_index_basic")) {
            return compute_mtp2_descriptor(context, batch, graph, host_batch, options);
        }
    }
    return compute_extended_generic(
        context, batch, graph, host_batch, name, options, control, rotational_plan);
}

} // namespace mdescriptor::cuda
