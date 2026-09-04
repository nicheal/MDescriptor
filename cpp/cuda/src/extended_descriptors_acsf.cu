#include "extended_descriptors_common.cuh"

__global__ void acsf_kernel(
    const I32* numbers,
    const I64* graph_offsets,
    const I32* graph_atoms,
    const double* graph_displacements,
    const double* graph_distance2,
    const I32* species,
    int species_count,
    double r_cut,
    const double* g2,
    int n_g2,
    const double* g3,
    int n_g3,
    const double* g4,
    int n_g4,
    const double* g5,
    int n_g5,
    I64 atoms,
    double* output) {
    const I64 center = static_cast<I64>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (center >= atoms) return;
    const int types = species_count;
    const int per_type = 1 + n_g2 + n_g3;
    const int angular_offset = types * per_type;
    double* values = output + center * (
        angular_offset + (n_g4 + n_g5) * types * (types + 1) / 2);
    const I64 begin = graph_offsets[center];
    const I64 end = graph_offsets[center + 1];
    for (I64 edge = begin; edge < end; ++edge) {
        const I32 atom = graph_atoms[edge];
        const int type = species_index(numbers[atom], species, species_count);
        if (type < 0) continue;
        const double distance = sqrt(fmax(0.0, graph_distance2[edge]));
        const double fc = 0.5 * (cos(kPi * distance / r_cut) + 1.0);
        I64 offset = static_cast<I64>(type) * per_type;
        values[offset++] += fc;
        for (int p = 0; p < n_g2; ++p) {
            const double eta = g2[p * 2 + 0];
            const double rs = g2[p * 2 + 1];
            values[offset++] += exp(-eta * (distance - rs) * (distance - rs)) * fc;
        }
        for (int p = 0; p < n_g3; ++p) {
            values[offset++] += cos(g3[p] * distance) * fc;
        }
    }
    for (I64 first = begin; first < end; ++first) {
        const I32 first_atom = graph_atoms[first];
        const int first_type = species_index(numbers[first_atom], species, species_count);
        if (first_type < 0) continue;
        const double first_distance2 = graph_distance2[first];
        const double first_distance = sqrt(fmax(0.0, first_distance2));
        const double first_cutoff = 0.5 * (cos(kPi * first_distance / r_cut) + 1.0);
        const double* first_vector = graph_displacements + first * 3;
        for (I64 second = begin; second < first; ++second) {
            const I32 second_atom = graph_atoms[second];
            const int second_type = species_index(numbers[second_atom], species, species_count);
            if (second_type < 0) continue;
            const double second_distance2 = graph_distance2[second];
            const double second_distance = sqrt(fmax(0.0, second_distance2));
            const double second_cutoff = 0.5 * (cos(kPi * second_distance / r_cut) + 1.0);
            const double* second_vector = graph_displacements + second * 3;
            const double dx = first_vector[0] - second_vector[0];
            const double dy = first_vector[1] - second_vector[1];
            const double dz = first_vector[2] - second_vector[2];
            const double third_distance2 = dx * dx + dy * dy + dz * dz;
            const double third_distance = sqrt(third_distance2);
            const double cosine = first_distance > 0.0 && second_distance > 0.0
                ? (first_vector[0] * second_vector[0]
                    + first_vector[1] * second_vector[1]
                    + first_vector[2] * second_vector[2])
                    / (first_distance * second_distance) : 0.0;
            const double clamped_cosine = fmin(1.0, fmax(-1.0, cosine));
            const int channel = pair_channel_device(first_type, second_type, species_count);
            const I64 base = angular_offset
                + static_cast<I64>(channel) * (n_g4 + n_g5);
            const double fc4 = first_cutoff * second_cutoff
                * (third_distance <= r_cut
                    ? 0.5 * (cos(kPi * third_distance / r_cut) + 1.0) : 0.0);
            const double fc5 = first_cutoff * second_cutoff;
            const double distance_sum = first_distance2 + second_distance2 + third_distance2;
            for (int p = 0; p < n_g4; ++p) {
                const double eta = g4[p * 3 + 0];
                const double zeta = g4[p * 3 + 1];
                const double lambda = g4[p * 3 + 2];
                const double angular = pow(
                    fmax(0.0, 0.5 * (1.0 + lambda * clamped_cosine)), zeta);
                const double radial = third_distance <= r_cut
                    ? exp(-eta * distance_sum) : 0.0;
                values[base + p] += 2.0 * angular * radial * fc4;
            }
            for (int p = 0; p < n_g5; ++p) {
                const double eta = g5[p * 3 + 0];
                const double zeta = g5[p * 3 + 1];
                const double lambda = g5[p * 3 + 2];
                values[base + n_g4 + p] += 2.0
                    * pow(fmax(0.0, 0.5 * (1.0 + lambda * clamped_cosine)), zeta)
                    * exp(-eta * (first_distance2 + second_distance2)) * fc5;
            }
        }
    }
}

std::vector<double> option_rows(
    const py::dict& options,
    const char* key,
    int columns) {
    std::vector<double> result;
    const py::str name(key);
    if (!options.contains(name) || options[name].is_none()) return result;
    const py::object value = options[name];
    if (py::isinstance<py::dict>(value)) {
        const py::dict object = py::cast<py::dict>(value);
        auto values = [&](const char* field) {
            const py::str field_name(field);
            if (!object.contains(field_name) || object[field_name].is_none()) {
                return std::vector<double>{};
            }
            return py::cast<std::vector<double>>(object[field_name]);
        };
        const auto first = values("eta");
        if (columns == 2) {
            auto second = values("Rs");
            if (second.empty()) second = values("rs");
            for (double left : first) for (double right : second) {
                result.push_back(left); result.push_back(right);
            }
        } else {
            const auto second = values("zeta");
            auto third = values("lambda");
            if (third.empty()) third = values("lambdas");
            for (double left : first) for (double middle : second) for (double right : third) {
                result.push_back(left); result.push_back(middle); result.push_back(right);
            }
        }
        return result;
    }
    try {
        const py::sequence rows = py::cast<py::sequence>(value);
        for (const py::handle item : rows) {
            const py::sequence row = py::cast<py::sequence>(item);
            if (py::len(row) != columns) {
                throw std::invalid_argument(std::string(key) + " has an invalid row width");
            }
            for (const py::handle component : row) result.push_back(py::cast<double>(component));
        }
    } catch (const py::cast_error&) {
        throw std::invalid_argument(std::string(key) + " must be an array or object");
    }
    return result;
}

std::vector<double> option_values(const py::dict& options, const char* key) {
    const py::str name(key);
    if (!options.contains(name) || options[name].is_none()) return {};
    try {
        return py::cast<std::vector<double>>(options[name]);
    } catch (const py::cast_error&) {
        throw std::invalid_argument(std::string(key) + " must be an array");
    }
}

py::dict compute_acsf_descriptor(
    CudaExecutionContext& context,
    DeviceBatch& batch,
    DeviceNeighborGraph& graph,
    const detail::StructureBatchView& host_batch,
    const py::dict& options) {
    const auto species = species_option(options);
    if (species.empty()) throw std::invalid_argument("ACSF species must not be empty");
    const double cutoff = option(options, "r_cut", 6.0);
    const auto g2 = option_rows(options, "g2_params", 2);
    const auto g3 = option_values(options, "g3_params");
    const auto g4 = option_rows(options, "g4_params", 3);
    const auto g5 = option_rows(options, "g5_params", 3);
    const int n_g2 = static_cast<int>(g2.size() / 2);
    const int n_g3 = static_cast<int>(g3.size());
    const int n_g4 = static_cast<int>(g4.size() / 3);
    const int n_g5 = static_cast<int>(g5.size() / 3);
    const I64 columns = static_cast<I64>(1 + n_g2 + n_g3) * species.size()
        + static_cast<I64>(n_g4 + n_g5) * species.size() * (species.size() + 1) / 2;
    graph.build_dpa(context, batch, host_batch, cutoff, true, false, false);
    const std::size_t size = static_cast<std::size_t>(batch.atoms())
        * static_cast<std::size_t>(columns);
    double* output = context.output_buffer(size);
    check_cuda(cudaMemsetAsync(output, 0, size * sizeof(double), context.stream()),
        "could not clear CUDA ACSF output");
    DeviceBuffer<I32> d_species;
    DeviceBuffer<double> d_g2;
    DeviceBuffer<double> d_g3;
    DeviceBuffer<double> d_g4;
    DeviceBuffer<double> d_g5;
    d_species.upload(species.data(), species.size(), context.stream(), "could not upload ACSF species");
    d_g2.upload(g2.data(), g2.size(), context.stream(), "could not upload ACSF G2 parameters");
    d_g3.upload(g3.data(), g3.size(), context.stream(), "could not upload ACSF G3 parameters");
    d_g4.upload(g4.data(), g4.size(), context.stream(), "could not upload ACSF G4 parameters");
    d_g5.upload(g5.data(), g5.size(), context.stream(), "could not upload ACSF G5 parameters");
    if (batch.atoms() > 0) {
        constexpr unsigned block_size = 64;
        acsf_kernel<<<static_cast<unsigned>((batch.atoms() + block_size - 1) / block_size),
            block_size, 0, context.stream()>>>(
                batch.numbers(), graph.offsets(), graph.atoms(), graph.displacements(),
                graph.distance2(), d_species.get(), static_cast<int>(species.size()), cutoff,
                d_g2.get(), n_g2, d_g3.get(), n_g3, d_g4.get(), n_g4, d_g5.get(), n_g5,
                batch.atoms(), output);
        check_cuda(cudaGetLastError(), "CUDA ACSF kernel launch failed");
    }
    const auto values = download_output_with_gil_release(context, size);
    return atom_result(values, batch.atoms(), columns, "ACSF", options, false,
        std::vector<I64>(host_batch.offsets, host_batch.offsets + host_batch.structures + 1));
}

} // namespace

py::dict compute_extended_acsf(
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
    return compute_acsf_descriptor(context, batch, graph, host_batch, options);
}

} // namespace mdescriptor::cuda
