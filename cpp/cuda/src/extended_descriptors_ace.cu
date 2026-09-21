#include "extended_descriptors_common.cuh"

namespace {

__device__ void ace_radial_values(
    double distance,
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
    int radial_count,
    double* result) {
    for (int index = 0; index < radial_count; ++index) result[index] = 0.0;
    const double t = pow((transform_a + transform_r0)
        / (transform_a + distance), transform_p);
    if ((p_left > 0 && t < t_left) || (p_right > 0 && t > t_right)) return;
    const double envelope = pow(t - t_left, p_left) * pow(t - t_right, p_right);
    result[0] = radial_a[0] * envelope;
    if (radial_count == 1) return;
    result[1] = (radial_a[1] * t + radial_b[1]) * result[0];
    for (int n = 2; n < radial_count; ++n) {
        result[n] = (radial_a[n] * t + radial_b[n]) * result[n - 1]
            + radial_c[n] * result[n - 2];
    }
}

} // namespace

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
        if (exact_self_edge(center, atom, graph_shifts, edge)) continue;
        const double distance = sqrt(fmax(0.0, graph_distance2[edge]));
        if (distance <= 0.0) continue;
        const int atom_type = species_index(numbers[atom], species, species_count);
        if (atom_type < 0) continue;
        ace_radial_values(
            distance, transform_a, transform_p, transform_r0, t_left, t_right,
            p_left, p_right, radial_a, radial_b, radial_c, max_radial, radial);
        complex_spherical_harmonics_device<21>(
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

    const auto* d_species = static_cast<const I32*>(context.static_payload_buffer(
        0, species.data(), species.size() * sizeof(I32), "could not upload ACE species"));
    const auto* d_base_species = static_cast<const I32*>(context.static_payload_buffer(
        1, base_species.data(), base_species.size() * sizeof(I32),
        "could not upload ACE base species"));
    const auto* d_base_radial = static_cast<const I32*>(context.static_payload_buffer(
        2, base_radial.data(), base_radial.size() * sizeof(I32),
        "could not upload ACE base radial indices"));
    const auto* d_base_angular = static_cast<const I32*>(context.static_payload_buffer(
        3, base_angular.data(), base_angular.size() * sizeof(I32),
        "could not upload ACE base angular indices"));
    const auto* d_base_magnetic = static_cast<const I32*>(context.static_payload_buffer(
        4, base_magnetic.data(), base_magnetic.size() * sizeof(I32),
        "could not upload ACE magnetic indices"));
    const auto* d_radial_a = static_cast<const double*>(context.static_payload_buffer(
        5, radial_a.data(), radial_a.size() * sizeof(double),
        "could not upload ACE radial recurrence"));
    const auto* d_radial_b = static_cast<const double*>(context.static_payload_buffer(
        6, radial_b.data(), radial_b.size() * sizeof(double),
        "could not upload ACE radial recurrence offset"));
    const auto* d_radial_c = static_cast<const double*>(context.static_payload_buffer(
        7, radial_c.data(), radial_c.size() * sizeof(double),
        "could not upload ACE radial recurrence second offset"));
    const auto* d_center_feature_offsets = static_cast<const I64*>(context.static_payload_buffer(
        8, center_feature_offsets.data(), center_feature_offsets.size() * sizeof(I64),
        "could not upload ACE center feature offsets"));
    const auto* d_feature_term_offsets = static_cast<const I64*>(context.static_payload_buffer(
        9, feature_term_offsets.data(), feature_term_offsets.size() * sizeof(I64),
        "could not upload ACE feature term offsets"));
    const auto* d_term_channel_offsets = static_cast<const I64*>(context.static_payload_buffer(
        10, term_channel_offsets.data(), term_channel_offsets.size() * sizeof(I64),
        "could not upload ACE term channel offsets"));
    const auto* d_term_channels = static_cast<const I32*>(context.static_payload_buffer(
        11, term_channels.data(), term_channels.size() * sizeof(I32),
        "could not upload ACE term channels"));
    const auto* d_term_coefficients = static_cast<const double*>(context.static_payload_buffer(
        12, term_coefficients.data(), term_coefficients.size() * sizeof(double),
        "could not upload ACE term coefficients"));
    const std::size_t size = static_cast<std::size_t>(batch.atoms()) * static_cast<std::size_t>(features);
    double* output = context.output_buffer(size);
    auto* coefficient_workspace = static_cast<double*>(context.workspace_buffer(
        static_cast<std::size_t>(batch.atoms()) * base_species.size() * 2 * sizeof(double)));
    if (size > 0) {
        zeroed_output(context, output, size, "could not clear ACE output");
        constexpr unsigned block_size = 64;
        ace_cuda_kernel<<<static_cast<unsigned>((batch.atoms() + block_size - 1) / block_size),
            block_size, 0, context.stream()>>>(
            batch.numbers(), graph.offsets(), graph.atoms(), graph.shifts(), graph.displacements(), graph.distance2(),
            d_species, static_cast<int>(species.size()), d_base_species, d_base_radial,
            d_base_angular, d_base_magnetic, static_cast<I64>(base_species.size()),
            max_radial, max_angular, transform_a, transform_p, transform_r0,
            py::cast<double>(payload["radial_t_left"]), py::cast<double>(payload["radial_t_right"]),
            py::cast<int>(payload["radial_p_left"]), py::cast<int>(payload["radial_p_right"]),
            d_radial_a, d_radial_b, d_radial_c, d_center_feature_offsets,
            d_feature_term_offsets, d_term_channel_offsets, d_term_channels,
            d_term_coefficients, features, batch.atoms(), coefficient_workspace, output);
        check_cuda(cudaGetLastError(), "ACE CUDA kernel launch failed");
    }
    const auto values = download_output_with_gil_release(
        context, size, batch.atoms(), features);
    return atom_result(values, features, "ACE", options, false,
        host_row_offsets(host_batch));
}

py::dict compute_extended_ace(
    CudaExecutionContext& context,
    DeviceBatch& batch,
    DeviceNeighborGraph& graph,
    const detail::StructureBatchView& host_batch,
    const std::string& name,
    const py::dict& options,
    RotationalPlanCache* rotational_plan) {
    (void)name;
    (void)rotational_plan;
    return compute_ace_descriptor(context, batch, graph, host_batch, options);
}

} // namespace mdescriptor::cuda
