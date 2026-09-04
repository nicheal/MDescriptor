#include "extended_descriptors_common.cuh"

__device__ double c00_spherical_bessel(int angular, double x) {
    const double absolute = fabs(x);
    if (absolute < 1e-4) {
        if (angular == 0) {
            const double x2 = x * x;
            return 1.0 - x2 / 6.0 + x2 * x2 / 120.0;
        }
        double denominator = 1.0;
        for (int value = 1; value <= angular; ++value) {
            denominator *= 2.0 * value + 1.0;
        }
        return pow(x, angular) / denominator
            * (1.0 - x * x / (2.0 * (2.0 * angular + 3.0)));
    }
    const double j0 = sin(x) / x;
    if (angular == 0) return j0;
    const double j1 = sin(x) / (x * x) - cos(x) / x;
    if (angular == 1) return j1;
    double previous = j0;
    double current = j1;
    for (int degree = 1; degree < angular; ++degree) {
        const double next = (2.0 * degree + 1.0) * current / x - previous;
        previous = current;
        current = next;
    }
    return current;
}

__device__ double c00_cutoff_value(int kind, double distance, double cutoff) {
    if (distance > cutoff) return 0.0;
    if (kind == 0) {
        return 0.5 * (cos(kPi * distance / cutoff) + 1.0);
    }
    if (kind == 1) {
        const double x = 4.0 * distance / cutoff - 3.0;
        if (x < -1.0) return 1.0;
        if (x < 1.0) return 0.25 * (x * x * x - 3.0 * x + 2.0);
        return 0.0;
    }
    constexpr double delta = 0.5;
    const double r1 = cutoff > delta ? cutoff - delta : 0.5 * cutoff;
    double value = 1.0;
    if (distance > r1) {
        const double cutoff2 = cutoff * cutoff;
        const double distance2 = distance * distance;
        value = (cutoff2 - distance2) * (cutoff2 - distance2)
            * (cutoff2 + 2.0 * distance2 - 3.0 * r1 * r1)
            / pow(cutoff2 - r1 * r1, 3.0);
    }
    if (kind == 3) value /= 1.0 + pow(distance / 2.0, 7.0);
    return value;
}

__device__ double c00_radial_value(
    double distance,
    int angular,
    int radial,
    int cutoff_kind,
    double cutoff,
    double sigma,
    const double* zeros,
    const double* norms,
    const double* tables,
    const I64* zero_offsets,
    const I64* norm_offsets,
    const I64* table_offsets,
    const I32* radial_counts,
    int table_width) {
    const int count = radial_counts[angular];
    if (radial < 0 || radial >= count) return 0.0;
    if (sigma > 0.0) {
        const double coordinate = fmin(1.0, fmax(0.0, distance / cutoff))
            * static_cast<double>(table_width - 1);
        const int left = min(table_width - 2, static_cast<int>(coordinate));
        const double fraction = coordinate - static_cast<double>(left);
        const I64 base = table_offsets[angular] + static_cast<I64>(radial) * table_width;
        const auto value = [&](int point) {
            const int bounded = max(0, min(table_width - 1, point));
            return tables[base + bounded];
        };
        const double p0 = value(left - 1);
        const double p1 = value(left);
        const double p2 = value(left + 1);
        const double p3 = value(left + 2);
        return p1 + 0.5 * fraction * (
            p2 - p0 + fraction * (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3
                + fraction * (3.0 * (p1 - p2) + p3 - p0)));
    }
    const double basis = c00_spherical_bessel(
        angular,
        zeros[zero_offsets[angular] + radial] * distance / cutoff);
    return c00_cutoff_value(cutoff_kind, distance, cutoff)
        * basis / norms[norm_offsets[angular] + radial];
}

__global__ void c00ps_mlff_kernel(
    const I32* numbers,
    const I64* graph_offsets,
    const I32* graph_atoms,
    const double* graph_displacements,
    const double* graph_distance2,
    const I32* species,
    int species_count,
    const I32* radial_counts,
    const I64* zero_offsets,
    const I64* norm_offsets,
    const I64* table_offsets,
    const double* zeros,
    const double* norms,
    const double* tables,
    const I64* coefficient_offsets,
    int cutoff_kind,
    double cutoff,
    double sigma,
    bool include_radial,
    bool include_angular,
    bool normalize_radial,
    bool normalize_angular,
    bool super_vector,
    bool exclude_self,
    double radial_weight,
    double angular_weight,
    int max_angular,
    int table_width,
    I64 features,
    I64 atoms,
    I64 coefficient_stride,
    double* workspace,
    double* output) {
    const I64 center = static_cast<I64>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (center >= atoms) return;
    const int center_type = species_index(numbers[center], species, species_count);
    if (center_type < 0) return;
    constexpr int MaxAngular = 20;
    double* coefficients = workspace + center * coefficient_stride;
    for (I64 index = 0; index < coefficient_stride; ++index) coefficients[index] = 0.0;
    const I64 begin = graph_offsets[center];
    const I64 end = graph_offsets[center + 1];
    for (I64 edge = begin; edge < end; ++edge) {
        const double distance = sqrt(fmax(0.0, graph_distance2[edge]));
        if (distance <= 1e-12 || distance > cutoff) continue;
        const int type = species_index(numbers[graph_atoms[edge]], species, species_count);
        if (type < 0) continue;
        double harmonics[441]{};
        harmonic_values<MaxAngular>(graph_displacements + edge * 3, harmonics, max_angular);
        for (int angular = 0; angular <= max_angular; ++angular) {
            const int count = radial_counts[angular];
            const I64 coefficient_base = coefficient_offsets[angular]
                + static_cast<I64>(type) * count * (2 * angular + 1);
            for (int radial = 0; radial < count; ++radial) {
                const double value = c00_radial_value(
                    distance, angular, radial, cutoff_kind, cutoff, sigma,
                    zeros, norms, tables,
                    zero_offsets, norm_offsets, table_offsets, radial_counts, table_width);
                const I64 destination = coefficient_base
                    + static_cast<I64>(radial) * (2 * angular + 1);
                for (int m = 0; m <= 2 * angular; ++m) {
                    coefficients[destination + m] += value
                        * harmonics[angular * angular + m];
                }
            }
        }
    }
    double* target = output + center * features;
    I64 output_index = 0;
    const int radial_channels = species_count * radial_counts[0];
    if (include_radial) {
        for (int channel = 0; channel < radial_channels; ++channel) {
            const int type = channel / radial_counts[0];
            const int radial = channel % radial_counts[0];
            target[output_index++] = coefficients[
                coefficient_offsets[0] + type * radial_counts[0] + radial];
        }
        if (normalize_radial) {
            double norm2 = 0.0;
            for (int index = 0; index < radial_channels; ++index) norm2 += target[index] * target[index];
            if (norm2 > 1e-20) {
                const double scale = 1.0 / sqrt(norm2);
                for (int index = 0; index < radial_channels; ++index) target[index] *= scale;
            }
        }
    }
    if (include_angular) {
        const I64 angular_offset = include_radial ? radial_channels : 0;
        I64 angular_index = 0;
        for (int angular = 0; angular <= max_angular; ++angular) {
            const int count = radial_counts[angular];
            const int channels = species_count * count;
            const double prefactor = sqrt(8.0 * kPi * kPi / (2.0 * angular + 1.0));
            for (int first = 0; first < channels; ++first) {
                for (int second = first; second < channels; ++second) {
                    double value = 0.0;
                    const int first_type = first / count;
                    const int first_radial = first % count;
                    const int second_type = second / count;
                    const int second_radial = second % count;
                    const I64 first_base = coefficient_offsets[angular]
                        + static_cast<I64>(first_type * count + first_radial) * (2 * angular + 1);
                    const I64 second_base = coefficient_offsets[angular]
                        + static_cast<I64>(second_type * count + second_radial) * (2 * angular + 1);
                    for (int m = 0; m <= 2 * angular; ++m) {
                        value += coefficients[first_base + m] * coefficients[second_base + m];
                    }
                    if (exclude_self && first_type == center_type && second_type == center_type) {
                        const double addition = (2.0 * angular + 1.0) / (4.0 * kPi);
                        for (I64 edge = begin; edge < end; ++edge) {
                            const double distance = sqrt(fmax(0.0, graph_distance2[edge]));
                            if (distance <= 1e-12 || distance > cutoff) continue;
                            const int type = species_index(numbers[graph_atoms[edge]], species, species_count);
                            if (type != center_type) continue;
                            const double left = c00_radial_value(
                                distance, angular, first_radial, cutoff_kind, cutoff, sigma,
                                zeros, norms, tables,
                                zero_offsets, norm_offsets, table_offsets, radial_counts, table_width);
                            const double right = c00_radial_value(
                                distance, angular, second_radial, cutoff_kind, cutoff, sigma,
                                zeros, norms, tables,
                                zero_offsets, norm_offsets, table_offsets, radial_counts, table_width);
                            value -= addition * left * right;
                        }
                    }
                    const double pair_weight = first_radial == second_radial ? 1.0 : sqrt(2.0);
                    target[angular_offset + angular_index++] = pair_weight * prefactor * value;
                }
            }
        }
    }
    if (normalize_angular) {
        const I64 angular_offset = include_radial ? radial_channels : 0;
        const I64 angular_size = features - angular_offset;
        double norm2 = 0.0;
        for (I64 index = 0; index < angular_size; ++index) {
            norm2 += target[angular_offset + index] * target[angular_offset + index];
        }
        if (norm2 > 1e-20) {
            const double scale = 1.0 / sqrt(norm2);
            for (I64 index = 0; index < angular_size; ++index) target[angular_offset + index] *= scale;
        }
    }
    if (super_vector) {
        const I64 radial_end = include_radial ? radial_channels : 0;
        if (include_radial) {
            const double scale = sqrt(radial_weight);
            for (I64 index = 0; index < radial_end; ++index) target[index] *= scale;
        }
        if (include_angular) {
            const double scale = sqrt(angular_weight);
            for (I64 index = radial_end; index < features; ++index) target[index] *= scale;
        }
        double norm2 = 0.0;
        for (I64 index = 0; index < features; ++index) norm2 += target[index] * target[index];
        if (norm2 > 1e-20) {
            const double scale = 1.0 / sqrt(norm2);
            for (I64 index = 0; index < features; ++index) target[index] *= scale;
        }
    }
}

py::dict compute_c00ps_mlff_descriptor(
    CudaExecutionContext& context,
    DeviceBatch& batch,
    DeviceNeighborGraph& graph,
    const detail::StructureBatchView& host_batch,
    const py::dict& options) {
    const py::str payload_key("_cuda_payload");
    if (!options.contains(payload_key) || options[payload_key].is_none()) {
        throw std::invalid_argument("C00PSMLFF CUDA backend requires its prepared basis payload");
    }
    const py::dict payload = py::cast<py::dict>(options[payload_key]);
    const auto species = species_option(options);
    const auto radial_counts = py::cast<std::vector<I32>>(payload["radial_counts"]);
    const auto zeros_nested = nested_payload_vectors(payload, "basis_zeros");
    const auto norms_nested = nested_payload_vectors(payload, "basis_norms");
    const auto tables_nested = nested_payload_vectors(payload, "basis_values");
    const int max_angular = option(options, "l_max", 4);
    const double cutoff = option(options, "r_cut", option(options, "cutoff", 6.0));
    const double sigma = option(options, "radial_sigma", 0.5);
    if (species.empty() || max_angular < 0 || max_angular >= static_cast<int>(radial_counts.size())
        || cutoff <= 0.0 || sigma < 0.0
        || zeros_nested.size() != radial_counts.size()
        || norms_nested.size() != radial_counts.size()
        || tables_nested.size() != radial_counts.size()) {
        throw std::invalid_argument("invalid C00PSMLFF CUDA basis payload");
    }
    const bool include_radial = option(options, "include_radial", true);
    const bool include_angular = option(options, "include_angular", true);
    const bool normalize_radial = option(options, "normalize_radial", false);
    const bool normalize_angular = option(options, "normalize_angular", false);
    const bool super_vector = option(options, "super_vector", false);
    const bool exclude_self = option(options, "exclude_self_interaction", true);
    const double radial_weight = option(options, "radial_weight", 1.0);
    const double angular_weight = option(options, "angular_weight", 1.0);
    const std::string cutoff_name = option(options, "cutoff_function", std::string("bp"));
    int cutoff_kind = cutoff_name == "bp" ? 0 : cutoff_name == "mo" ? 1
        : cutoff_name == "rj" ? 2 : cutoff_name == "wmc" ? 3 : -1;
    if (cutoff_kind < 0) throw std::invalid_argument("invalid C00PSMLFF cutoff function");

    std::vector<I64> zero_offsets(radial_counts.size(), 0);
    std::vector<I64> norm_offsets(radial_counts.size(), 0);
    std::vector<I64> table_offsets(radial_counts.size(), 0);
    std::vector<double> zeros;
    std::vector<double> norms;
    std::vector<double> tables;
    I64 coefficient_stride = 0;
    for (std::size_t angular = 0; angular < radial_counts.size(); ++angular) {
        zero_offsets[angular] = static_cast<I64>(zeros.size());
        norm_offsets[angular] = static_cast<I64>(norms.size());
        table_offsets[angular] = static_cast<I64>(tables.size());
        zeros.insert(zeros.end(), zeros_nested[angular].begin(), zeros_nested[angular].end());
        norms.insert(norms.end(), norms_nested[angular].begin(), norms_nested[angular].end());
        tables.insert(tables.end(), tables_nested[angular].begin(), tables_nested[angular].end());
        coefficient_stride += static_cast<I64>(species.size()) * radial_counts[angular]
            * (2 * static_cast<int>(angular) + 1);
    }
    std::vector<I64> coefficient_offsets(radial_counts.size(), 0);
    I64 coefficient_offset = 0;
    for (std::size_t angular = 0; angular < radial_counts.size(); ++angular) {
        coefficient_offsets[angular] = coefficient_offset;
        coefficient_offset += static_cast<I64>(species.size()) * radial_counts[angular]
            * (2 * static_cast<int>(angular) + 1);
    }
    const int table_width = 10001;
    if (sigma > 0.0) {
        for (std::size_t angular = 0; angular < tables_nested.size(); ++angular) {
            const std::size_t expected = static_cast<std::size_t>(radial_counts[angular]) * table_width;
            if (tables_nested[angular].size() != expected) {
                throw std::invalid_argument("C00PSMLFF CUDA radial table has an unexpected size");
            }
        }
    }
    const I64 features = feature_count_option(options, 0);
    const I64 radial_features = include_radial
        ? static_cast<I64>(species.size()) * radial_counts[0] : 0;
    I64 angular_features = 0;
    if (include_angular) {
        for (int angular = 0; angular <= max_angular; ++angular) {
            const I64 channels = static_cast<I64>(species.size()) * radial_counts[angular];
            angular_features += channels * (channels + 1) / 2;
        }
    }
    const I64 computed_features = radial_features + angular_features;
    if (features != computed_features || computed_features <= 0) {
        throw std::invalid_argument("C00PSMLFF CUDA feature count mismatch");
    }
    graph.build_dpa(context, batch, host_batch, cutoff, true, false, false);

    DeviceBuffer<I32> d_species;
    DeviceBuffer<I32> d_radial_counts;
    DeviceBuffer<I64> d_zero_offsets;
    DeviceBuffer<I64> d_norm_offsets;
    DeviceBuffer<I64> d_table_offsets;
    DeviceBuffer<double> d_zeros;
    DeviceBuffer<double> d_norms;
    DeviceBuffer<double> d_tables;
    DeviceBuffer<I64> d_coefficient_offsets;
    d_species.upload(species.data(), species.size(), context.stream(), "could not upload C00PS species");
    d_radial_counts.upload(radial_counts.data(), radial_counts.size(), context.stream(), "could not upload C00PS radial counts");
    d_zero_offsets.upload(zero_offsets.data(), zero_offsets.size(), context.stream(), "could not upload C00PS zero offsets");
    d_norm_offsets.upload(norm_offsets.data(), norm_offsets.size(), context.stream(), "could not upload C00PS norm offsets");
    d_table_offsets.upload(table_offsets.data(), table_offsets.size(), context.stream(), "could not upload C00PS table offsets");
    d_zeros.upload(zeros.data(), zeros.size(), context.stream(), "could not upload C00PS zeros");
    d_norms.upload(norms.data(), norms.size(), context.stream(), "could not upload C00PS norms");
    d_tables.upload(tables.data(), tables.size(), context.stream(), "could not upload C00PS radial tables");
    d_coefficient_offsets.upload(coefficient_offsets.data(), coefficient_offsets.size(), context.stream(), "could not upload C00PS coefficient offsets");
    const std::size_t size = static_cast<std::size_t>(batch.atoms()) * static_cast<std::size_t>(features);
    double* output = context.output_buffer(size);
    auto* workspace = static_cast<double*>(context.workspace_buffer(
        static_cast<std::size_t>(batch.atoms()) * static_cast<std::size_t>(coefficient_stride)
        * sizeof(double)));
    if (size > 0) {
        check_cuda(cudaMemsetAsync(output, 0, size * sizeof(double), context.stream()),
            "could not clear C00PS output");
        constexpr unsigned block_size = 64;
        c00ps_mlff_kernel<<<static_cast<unsigned>((batch.atoms() + block_size - 1) / block_size),
            block_size, 0, context.stream()>>>(
            batch.numbers(), graph.offsets(), graph.atoms(), graph.displacements(), graph.distance2(),
            d_species.get(), static_cast<int>(species.size()), d_radial_counts.get(),
            d_zero_offsets.get(), d_norm_offsets.get(), d_table_offsets.get(),
            d_zeros.get(), d_norms.get(), d_tables.get(), d_coefficient_offsets.get(),
            cutoff_kind, cutoff, sigma, include_radial, include_angular, normalize_radial,
            normalize_angular, super_vector, exclude_self, radial_weight, angular_weight,
            max_angular, table_width, features, batch.atoms(), coefficient_stride, workspace, output);
        check_cuda(cudaGetLastError(), "CUDA C00PSMLFF kernel launch failed");
    }
    const auto values = download_output_with_gil_release(context, size);
    return atom_result(values, batch.atoms(), features, "C00PSMLFF", options, false,
        std::vector<I64>(host_batch.offsets, host_batch.offsets + host_batch.structures + 1));
}

} // namespace

py::dict compute_extended_c00ps(
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
    return compute_c00ps_mlff_descriptor(context, batch, graph, host_batch, options);
}

} // namespace mdescriptor::cuda
