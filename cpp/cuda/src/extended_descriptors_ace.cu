#include "extended_descriptors_common.cuh"

__global__ void ace_cuda_kernel(
    const I32* numbers,
    const I64* graph_offsets,
    const I32* graph_atoms,
    const I32* graph_shifts,
    const double* graph_displacements,
    const double* graph_distance2,
    const I32* species,
    int species_count,
    const I32* base_species,
    const I32* base_radial,
    const I32* base_angular,
    const I32* base_magnetic,
    I64 base_channels,
    int max_radial,
    int max_angular,
    double transform_a,
    double transform_p,
    double transform_r0,
    double t_left,
    double t_right,
    int p_left,
    int p_right,
    const double* radial_a,
    const double* radial_b,
    const double* radial_c,
    const I64* center_feature_offsets,
    const I64* feature_term_offsets,
    const I64* term_channel_offsets,
    const I32* term_channels,
    const double* term_coefficients,
    I64 features,
    I64 atoms,
    double* coefficient_workspace,
    double* output) {
    const I64 center = static_cast<I64>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (center >= atoms || base_channels > 2048) return;
    const int center_type = species_index(numbers[center], species, species_count);
    if (center_type < 0) return;
    DeviceComplex* values = reinterpret_cast<DeviceComplex*>(
        coefficient_workspace + center * base_channels * 2);
    for (I64 channel = 0; channel < base_channels; ++channel) values[channel] = {0.0, 0.0};
    const I64 begin = graph_offsets[center];
    const I64 end = graph_offsets[center + 1];
    double radial[256]{};
    DeviceComplex harmonics[441]{};
    for (I64 edge = begin; edge < end; ++edge) {
        const I32 atom = graph_atoms[edge];
        if (atom == center && graph_shifts[edge * 3] == 0
            && graph_shifts[edge * 3 + 1] == 0 && graph_shifts[edge * 3 + 2] == 0) continue;
        const double distance = sqrt(fmax(0.0, graph_distance2[edge]));
        if (distance <= 0.0) continue;
        const int atom_type = species_index(numbers[atom], species, species_count);
        if (atom_type < 0) continue;
        ace_radial_values(
            distance, transform_a, transform_p, transform_r0, t_left, t_right,
            p_left, p_right, radial_a, radial_b, radial_c, max_radial, radial);
        ace_spherical_harmonics(
            graph_displacements + edge * 3, max_angular, harmonics);
        for (I64 channel = 0; channel < base_channels; ++channel) {
            if (base_species[channel] != atom_type) continue;
            const int radial_index = base_radial[channel] - 1;
            const int angular = base_angular[channel];
            if (radial_index < 0 || radial_index >= max_radial) continue;
            const DeviceComplex angular_value = harmonics[
                angular * angular + angular + base_magnetic[channel]];
            values[channel] = complex_add(values[channel], complex_scale(
                angular_value, radial[radial_index]));
        }
    }
    const I64 feature_begin = center_feature_offsets[center_type];
    const I64 feature_end = center_feature_offsets[center_type + 1];
    double* target = output + center * features;
    for (I64 feature = feature_begin; feature < feature_end; ++feature) {
        double value = 0.0;
        const I64 term_begin = feature_term_offsets[feature];
        const I64 term_end = feature_term_offsets[feature + 1];
        for (I64 term = term_begin; term < term_end; ++term) {
            DeviceComplex product = {term_coefficients[term], 0.0};
            for (I64 index = term_channel_offsets[term];
                 index < term_channel_offsets[term + 1]; ++index) {
                product = complex_multiply(product, values[term_channels[index]]);
            }
            value += product.real;
        }
        target[feature - feature_begin] = value;
    }
}

py::dict compute_ace_descriptor(
    CudaExecutionContext& context,
    DeviceBatch& batch,
    DeviceNeighborGraph& graph,
    const detail::StructureBatchView& host_batch,
    const py::dict& options) {
    const py::str payload_key("_cuda_payload");
    if (!options.contains(payload_key) || options[payload_key].is_none()) {
        throw std::invalid_argument("ACE CUDA backend requires its generated basis payload");
    }
    const py::dict payload = py::cast<py::dict>(options[payload_key]);
    const auto species = species_option(options);
    const auto base_species = py::cast<std::vector<I32>>(payload["base_species"]);
    const auto base_radial = py::cast<std::vector<I32>>(payload["base_radial"]);
    const auto base_angular = py::cast<std::vector<I32>>(payload["base_angular"]);
    const auto base_magnetic = py::cast<std::vector<I32>>(payload["base_magnetic"]);
    const auto radial_a = vector_child(payload, "radial_a");
    const auto radial_b = vector_child(payload, "radial_b");
    const auto radial_c = vector_child(payload, "radial_c");
    const auto center_feature_offsets = py::cast<std::vector<I64>>(payload["center_feature_offsets"]);
    const auto feature_term_offsets = py::cast<std::vector<I64>>(payload["feature_term_offsets"]);
    const auto term_channel_offsets = py::cast<std::vector<I64>>(payload["term_channel_offsets"]);
    const auto term_channels = py::cast<std::vector<I32>>(payload["term_channels"]);
    const auto term_coefficients = vector_child(payload, "term_coefficients");
    const int max_radial = payload.contains("max_radial")
        ? py::cast<int>(payload["max_radial"]) : static_cast<int>(radial_a.size());
    const int max_angular = payload.contains("max_angular")
        ? py::cast<int>(payload["max_angular"]) : option(options, "N", 3);
    const py::dict transform = child_dict(options, "trans");
    const double transform_a = option(transform, "a", 1.0);
    const double transform_p = option(transform, "p", 2.0);
    const double transform_r0 = option(options, "r0", 2.5);
    const double cutoff = option(options, "rcut", 5.0);
    const I64 features = feature_count_option(options, 0);
    if (species.empty() || base_species.size() != base_radial.size()
        || base_species.size() != base_angular.size()
        || base_species.size() != base_magnetic.size() || base_species.empty()
        || base_species.size() > 2048 || radial_a.empty()
        || radial_a.size() != radial_b.size() || radial_a.size() != radial_c.size()
        || center_feature_offsets.size() != species.size() + 1
        || feature_term_offsets.size() != static_cast<std::size_t>(center_feature_offsets.back() + 1)
        || term_channel_offsets.empty()
        || term_channel_offsets.back() != static_cast<I64>(term_channels.size())
        || term_channel_offsets.size() != term_coefficients.size() + 1
        || features <= 0 || max_radial <= 0 || max_radial > 256
        || max_angular < 0 || max_angular > 20 || cutoff <= 0.0
        || transform_a + transform_r0 <= 0.0 || transform_p == 0.0) {
        throw std::invalid_argument("invalid ACE CUDA basis payload");
    }
    graph.build_dpa(context, batch, host_batch, cutoff, true, false, false);

    DeviceBuffer<I32> d_species;
    DeviceBuffer<I32> d_base_species;
    DeviceBuffer<I32> d_base_radial;
    DeviceBuffer<I32> d_base_angular;
    DeviceBuffer<I32> d_base_magnetic;
    DeviceBuffer<double> d_radial_a;
    DeviceBuffer<double> d_radial_b;
    DeviceBuffer<double> d_radial_c;
    DeviceBuffer<I64> d_center_feature_offsets;
    DeviceBuffer<I64> d_feature_term_offsets;
    DeviceBuffer<I64> d_term_channel_offsets;
    DeviceBuffer<I32> d_term_channels;
    DeviceBuffer<double> d_term_coefficients;
    d_species.upload(species.data(), species.size(), context.stream(), "could not upload ACE species");
    d_base_species.upload(base_species.data(), base_species.size(), context.stream(), "could not upload ACE base species");
    d_base_radial.upload(base_radial.data(), base_radial.size(), context.stream(), "could not upload ACE base radial indices");
    d_base_angular.upload(base_angular.data(), base_angular.size(), context.stream(), "could not upload ACE base angular indices");
    d_base_magnetic.upload(base_magnetic.data(), base_magnetic.size(), context.stream(), "could not upload ACE magnetic indices");
    d_radial_a.upload(radial_a.data(), radial_a.size(), context.stream(), "could not upload ACE radial recurrence");
    d_radial_b.upload(radial_b.data(), radial_b.size(), context.stream(), "could not upload ACE radial recurrence offset");
    d_radial_c.upload(radial_c.data(), radial_c.size(), context.stream(), "could not upload ACE radial recurrence second offset");
    d_center_feature_offsets.upload(center_feature_offsets.data(), center_feature_offsets.size(), context.stream(), "could not upload ACE center feature offsets");
    d_feature_term_offsets.upload(feature_term_offsets.data(), feature_term_offsets.size(), context.stream(), "could not upload ACE feature term offsets");
    d_term_channel_offsets.upload(term_channel_offsets.data(), term_channel_offsets.size(), context.stream(), "could not upload ACE term channel offsets");
    d_term_channels.upload(term_channels.data(), term_channels.size(), context.stream(), "could not upload ACE term channels");
    d_term_coefficients.upload(term_coefficients.data(), term_coefficients.size(), context.stream(), "could not upload ACE term coefficients");
    const std::size_t size = static_cast<std::size_t>(batch.atoms()) * static_cast<std::size_t>(features);
    double* output = context.output_buffer(size);
    auto* coefficient_workspace = static_cast<double*>(context.workspace_buffer(
        static_cast<std::size_t>(batch.atoms()) * base_species.size() * 2 * sizeof(double)));
    if (size > 0) {
        check_cuda(cudaMemsetAsync(output, 0, size * sizeof(double), context.stream()),
            "could not clear ACE output");
        constexpr unsigned block_size = 64;
        ace_cuda_kernel<<<static_cast<unsigned>((batch.atoms() + block_size - 1) / block_size),
            block_size, 0, context.stream()>>>(
            batch.numbers(), graph.offsets(), graph.atoms(), graph.shifts(), graph.displacements(), graph.distance2(),
            d_species.get(), static_cast<int>(species.size()), d_base_species.get(), d_base_radial.get(),
            d_base_angular.get(), d_base_magnetic.get(), static_cast<I64>(base_species.size()),
            max_radial, max_angular, transform_a, transform_p, transform_r0,
            py::cast<double>(payload["radial_t_left"]), py::cast<double>(payload["radial_t_right"]),
            py::cast<int>(payload["radial_p_left"]), py::cast<int>(payload["radial_p_right"]),
            d_radial_a.get(), d_radial_b.get(), d_radial_c.get(), d_center_feature_offsets.get(),
            d_feature_term_offsets.get(), d_term_channel_offsets.get(), d_term_channels.get(),
            d_term_coefficients.get(), features, batch.atoms(), coefficient_workspace, output);
        check_cuda(cudaGetLastError(), "ACE CUDA kernel launch failed");
    }
    const auto values = download_output_with_gil_release(context, size);
    return atom_result(values, batch.atoms(), features, "ACE", options, false,
        std::vector<I64>(host_batch.offsets, host_batch.offsets + host_batch.structures + 1));
}


} // namespace

py::dict compute_extended_ace(
    CudaExecutionContext& context,
    DeviceBatch& batch,
    DeviceNeighborGraph& graph,
    const detail::StructureBatchView& host_batch,
    const std::string& name,
    const py::dict& options,
    const py::object& control,
    RotationalPlanCache* rotational_plan) {
    (void)name;
    (void)control;
    (void)rotational_plan;
    return compute_ace_descriptor(context, batch, graph, host_batch, options);
}

} // namespace mdescriptor::cuda
