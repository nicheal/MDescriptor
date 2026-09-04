#include "extended_descriptors_common.cuh"

__global__ void generic_moment_kernel(
    const I32* numbers,
    const I64* graph_offsets,
    const I32* graph_atoms,
    const double* graph_displacements,
    const double* graph_distance2,
    const I32* species,
    int species_count,
    const I32* channel_species,
    const I32* channel_radial,
    int channels,
    int max_rank,
    int mode,
    int max_order,
    double min_dist,
    double max_dist,
    double soft_cutoff,
    double hard_cutoff,
    double radial_sigma,
    double radial_weight,
    double angular_weight,
    const double* center_weights,
    I64 features,
    I64 atoms,
    I64 moment_stride,
    int mtp_radial_basis_size,
    double* moment_workspace,
    double* output) {
    const I64 center = static_cast<I64>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (center >= atoms) return;
    if (channels > 256 || max_rank > 20) return;
    double local_moments[256 * 21]{};
    double* moments = local_moments;
    if (mode == 0) {
        if (moment_workspace == nullptr || max_rank > 5 || moment_stride <= 0) return;
        moments = moment_workspace + center * static_cast<I64>(channels) * moment_stride;
        for (I64 index = 0; index < static_cast<I64>(channels) * moment_stride; ++index) {
            moments[index] = 0.0;
        }
    }
    const I64 begin = graph_offsets[center];
    const I64 end = graph_offsets[center + 1];
    const int center_type = species_index(numbers[center], species, species_count);
    if (mode == 1 && center_type >= 0 && center_weights != nullptr) {
        for (int channel = 0; channel < channels; ++channel) {
            if (channel_species[channel] == center_type && channel_radial[channel] == 0) {
                moments[channel * (max_rank + 1)] += center_weights[center_type];
            }
        }
    }
    for (I64 edge = begin; edge < end; ++edge) {
        const double distance = sqrt(fmax(0.0, graph_distance2[edge]));
        if (distance < min_dist || distance > max_dist) continue;
        const int type = species_index(numbers[graph_atoms[edge]], species, species_count);
        if (type < 0) continue;
        const double reduced = (distance - min_dist) / fmax(max_dist - min_dist, 1e-12);
        const double cutoff = 0.5 * (1.0 + cos(kPi * reduced));
        double soft = 1.0;
        if (mode == 1) {
            if (distance >= hard_cutoff) continue;
            soft = distance <= soft_cutoff ? 1.0
                : 0.5 * (1.0 + cos(kPi * (distance - soft_cutoff)
                    / fmax(hard_cutoff - soft_cutoff, 1e-12)));
        }
        for (int channel = 0; channel < channels; ++channel) {
            if (channel_species[channel] != type) continue;
            const int radial = channel_radial[channel];
            double radial_value = 0.0;
            if (mode == 0) {
                const int basis = radial % max(1, mtp_radial_basis_size);
                const int radial_function = radial / max(1, mtp_radial_basis_size);
                const double radial_scale = radial_function == 0
                    ? 1.0 : pow(fmax(0.0, reduced), radial_function);
                radial_value = cutoff * cutoff * radial_scale
                    * chebyshev_device(basis, 2.0 * reduced - 1.0);
            } else {
                radial_value = cutoff * soft
                    * pow(fmax(0.0, reduced), radial % 4)
                    * cos((radial + 1.0) * kPi * reduced);
            }
            if (mode == 3 && radial_sigma > 0.0) {
                radial_value *= exp(-0.5 * distance * distance / (radial_sigma * radial_sigma));
            }
            if (mode == 0) {
                const double inverse_distance = 1.0 / distance;
                const double unit[3] = {
                    graph_displacements[edge * 3 + 0] * inverse_distance,
                    graph_displacements[edge * 3 + 1] * inverse_distance,
                    graph_displacements[edge * 3 + 2] * inverse_distance,
                };
                for (int rank = 0; rank <= max_rank; ++rank) {
                    I64 tensor_size = 1;
                    I64 tensor_offset = 0;
                    for (int previous = 0; previous < rank; ++previous) {
                        tensor_offset += tensor_size;
                        tensor_size *= 3;
                    }
                    for (I64 flat = 0; flat < tensor_size; ++flat) {
                        I64 value = flat;
                        double product = radial_value;
                        for (int component = 0; component < rank; ++component) {
                            const int axis = static_cast<int>(value % 3);
                            value /= 3;
                            product *= unit[axis];
                        }
                        moments[static_cast<I64>(channel) * moment_stride + tensor_offset + flat] += product;
                    }
                }
            } else {
                for (int rank = 0; rank <= max_rank; ++rank) {
                    moments[channel * (max_rank + 1) + rank] += radial_value
                        * pow(distance / fmax(max_dist, 1e-12), rank);
                }
            }
        }
    }
    double* target = output + center * features;
    for (I64 feature = 0; feature < features; ++feature) {
        double value = 0.0;
        if (mode == 0) {
            // Standalone MTP: traces first, followed by rank-wise pair
            // contractions, exactly matching its public feature count/order.
            I64 trace_count = static_cast<I64>(channels) * (max_rank / 2 + 1);
            if (feature < trace_count) {
                const int channel = static_cast<int>(feature / (max_rank / 2 + 1));
                const int rank = 2 * static_cast<int>(feature % (max_rank / 2 + 1));
                I64 tensor_offset = 0;
                I64 tensor_size = 1;
                for (int previous = 0; previous < rank; ++previous) {
                    tensor_offset += tensor_size;
                    tensor_size *= 3;
                }
                if (rank == 0) {
                    value = moments[static_cast<I64>(channel) * moment_stride];
                } else {
                    const int trace_pairs = rank / 2;
                    I64 combinations = 1;
                    for (int pair = 0; pair < trace_pairs; ++pair) combinations *= 3;
                    for (I64 combination = 0; combination < combinations; ++combination) {
                        I64 remainder = combination;
                        I64 flat = 0;
                        for (int pair = 0; pair < trace_pairs; ++pair) {
                            const int component = static_cast<int>(remainder % 3);
                            remainder /= 3;
                            flat = flat * 9 + component * 3 + component;
                        }
                        value += moments[static_cast<I64>(channel) * moment_stride
                            + tensor_offset + flat];
                    }
                }
            } else {
                I64 remainder = feature - trace_count;
                const I64 pair_count = static_cast<I64>(channels) * (channels + 1) / 2;
                const int rank = max_rank < 0 ? 0 : static_cast<int>(
                    remainder / pair_count) % (max_rank + 1);
                remainder %= pair_count;
                int first = 0;
                for (; first < channels; ++first) {
                    const int count = channels - first;
                    if (remainder < count) break;
                    remainder -= count;
                }
                const int second = first + static_cast<int>(remainder);
                I64 tensor_offset = 0;
                I64 tensor_size = 1;
                for (int previous = 0; previous < rank; ++previous) {
                    tensor_offset += tensor_size;
                    tensor_size *= 3;
                }
                for (I64 component = 0; component < tensor_size; ++component) {
                    value += moments[static_cast<I64>(first) * moment_stride
                        + tensor_offset + component]
                        * moments[static_cast<I64>(second) * moment_stride
                        + tensor_offset + component];
                }
            }
        } else {
            const int body = max(0, max_order);
            const I64 pair_count = static_cast<I64>(channels) * (channels + 1) / 2;
            const I64 pair = pair_count > 0 ? feature / (body + 1) % pair_count : 0;
            int first = 0;
            I64 remainder = pair;
            for (; first < channels; ++first) {
                const int count = channels - first;
                if (remainder < count) break;
                remainder -= count;
            }
            const int second = min(channels - 1, first + static_cast<int>(remainder));
            const int rank = max_rank == 0 ? 0 : static_cast<int>(feature % (max_rank + 1));
            value = moments[first * (max_rank + 1) + rank]
                * moments[second * (max_rank + 1) + rank];
            if (mode == 2) {
                const int order = body == 0 ? 1 : static_cast<int>(feature % (body + 1)) + 1;
                value *= pow(fabs(moments[first * (max_rank + 1)]) + 1e-12, order - 2);
            }
            if (mode == 3) {
                value *= feature < channels ? radial_weight : angular_weight;
            }
        }
        target[feature] = value;
    }
}

py::dict compute_generic_moment_descriptor(
    CudaExecutionContext& context,
    DeviceBatch& batch,
    DeviceNeighborGraph& graph,
    const detail::StructureBatchView& host_batch,
    const std::string& name,
    const py::dict& options) {
    const auto species = species_option(options);
    if (species.empty()) throw std::invalid_argument(name + " species must not be empty");

    int mode = 0;
    int max_rank = 2;
    int max_order = 0;
    double min_dist = 0.0;
    double max_dist = 5.0;
    double soft_cutoff = max_dist;
    double hard_cutoff = max_dist;
    double radial_sigma = 0.0;
    double radial_weight = 1.0;
    double angular_weight = 1.0;
    int mtp_radial_basis_size = 1;
    std::vector<I32> channel_species;
    std::vector<I32> channel_radial;
    std::vector<double> center_weights;
    I64 computed_features = 0;

    if (name == "SOAPTurbo") {
        mode = 1;
        const auto alpha = integer_vector_option(options, "alpha_max", 8, species.size());
        const int lmax = option(options, "l_max", 6);
        hard_cutoff = option(options, "rcut_hard", 5.0);
        soft_cutoff = option(options, "rcut_soft", hard_cutoff);
        max_dist = hard_cutoff;
        max_rank = lmax;
        radial_sigma = numeric_vector_option(options, "atom_sigma_r", 0.5, species.size())[0];
        center_weights = numeric_vector_option(options, "central_weight", 1.0, species.size());
        for (std::size_t type = 0; type < alpha.size(); ++type) {
            if (alpha[type] <= 0) throw std::invalid_argument("SOAPTurbo alpha_max must be positive");
            for (int radial = 0; radial < alpha[type]; ++radial) {
                channel_species.push_back(static_cast<I32>(type));
                channel_radial.push_back(static_cast<I32>(radial));
            }
        }
        const I64 channels = static_cast<I64>(channel_species.size());
        const std::string compression = option(options, "compression", std::string("off"));
        if (compression.empty() || compression == "off" || compression == "none") {
            computed_features = channels * (channels + 1) / 2 * (max_rank + 1);
        } else if (compression == "trivial") {
            std::vector<int> pivots;
            int pivot = 0;
            for (I32 count : alpha) {
                pivots.push_back(pivot);
                pivot += count;
            }
            I64 retained = 0;
            for (I64 first = 0; first < channels; ++first) {
                for (I64 second = first; second < channels; ++second) {
                    bool keep = false;
                    for (int value : pivots) keep = keep || first == value || second == value;
                    if (keep) ++retained;
                }
            }
            computed_features = retained * (max_rank + 1);
        } else if (compression.size() == 3 && compression[1] == '_') {
            const int nu_r = compression[0] - '0';
            const int nu_s = compression[2] - '0';
            if (nu_r < 0 || nu_r > 2 || nu_s < 0 || nu_s > 2
                || std::adjacent_find(alpha.begin(), alpha.end(), std::not_equal_to<>()) != alpha.end()) {
                throw std::invalid_argument("invalid SOAPTurbo compression");
            }
            const int n1 = nu_r > 0 ? alpha[0] : 1;
            const int n2 = nu_r == 2 ? alpha[0] : 1;
            const int s1 = nu_s > 0 ? static_cast<int>(species.size()) : 1;
            const int s2 = nu_s == 2 ? static_cast<int>(species.size()) : 1;
            if (nu_r % 2 == 0 && nu_s % 2 == 0) {
                const I64 compressed_channels = static_cast<I64>(n1) * s1;
                computed_features = compressed_channels * (compressed_channels + 1) / 2
                    * (max_rank + 1);
            } else {
                computed_features = static_cast<I64>(n1) * s1 * n2 * s2 * (max_rank + 1);
            }
        } else {
            throw std::invalid_argument("invalid SOAPTurbo compression");
        }
    } else if (name == "ACE") {
        mode = 2;
        const auto maxdeg = numeric_values_option(options, "maxdeg", 8.0);
        if (maxdeg.empty()) throw std::invalid_argument("ACE maxdeg must not be empty");
        const double largest_degree = *std::max_element(maxdeg.begin(), maxdeg.end());
        const int radial_count = std::max(1, static_cast<int>(std::ceil(largest_degree)) + 1);
        const int correlation = option(options, "N", 3);
        min_dist = option(options, "rin", 0.0);
        max_dist = option(options, "rcut", 5.0);
        max_rank = std::min(20, std::max(0, correlation));
        max_order = std::min(20, std::max(0, correlation));
        for (std::size_t type = 0; type < species.size(); ++type) {
            for (int radial = 0; radial < radial_count; ++radial) {
                channel_species.push_back(static_cast<I32>(type));
                channel_radial.push_back(static_cast<I32>(radial));
            }
        }
        const I64 channels = static_cast<I64>(channel_species.size());
        computed_features = channels * (max_rank / 2 + 1)
            + static_cast<I64>(max_rank + 1) * channels * (channels + 1) / 2;
    } else if (name == "MTP") {
        mode = 0;
        const int radial_basis_size = option(options, "radial_basis_size", 4);
        const int radial_funcs_count = option(options, "radial_funcs_count", 1);
        mtp_radial_basis_size = radial_basis_size;
        max_rank = option(options, "max_rank", 2);
        min_dist = option(options, "min_dist", 0.0);
        max_dist = option(options, "max_dist", option(options, "r_cut", 5.0));
        if (radial_basis_size <= 0 || radial_funcs_count <= 0) {
            throw std::invalid_argument("MTP radial basis sizes must be positive");
        }
        for (std::size_t type = 0; type < species.size(); ++type) {
            for (int function = 0; function < radial_funcs_count; ++function) {
                for (int radial = 0; radial < radial_basis_size; ++radial) {
                    channel_species.push_back(static_cast<I32>(type));
                    channel_radial.push_back(static_cast<I32>(function * radial_basis_size + radial));
                }
            }
        }
        const I64 channels = static_cast<I64>(channel_species.size());
        computed_features = channels * (max_rank / 2 + 1)
            + static_cast<I64>(max_rank + 1) * channels * (channels + 1) / 2;
    } else if (name == "C00PSMLFF") {
        mode = 3;
        const int radial_count = option(
            options, "n_radial", option(options, "n_max", 8));
        max_rank = option(options, "l_max", 4);
        max_dist = option(options, "r_cut", option(options, "cutoff", 6.0));
        radial_sigma = option(options, "radial_sigma", 0.5);
        radial_weight = option(options, "radial_weight", 1.0);
        angular_weight = option(options, "angular_weight", 1.0);
        const bool include_radial = option(options, "include_radial", true);
        const bool include_angular = option(options, "include_angular", true);
        if (radial_count <= 0) throw std::invalid_argument("C00PSMLFF n_radial must be positive");
        for (std::size_t type = 0; type < species.size(); ++type) {
            for (int radial = 0; radial < radial_count; ++radial) {
                channel_species.push_back(static_cast<I32>(type));
                channel_radial.push_back(static_cast<I32>(radial));
            }
        }
        const I64 channels = static_cast<I64>(channel_species.size());
        computed_features = include_radial ? channels : 0;
        if (include_angular) {
            computed_features += static_cast<I64>(max_rank + 1) * channels * (channels + 1) / 2;
        }
    } else {
        throw std::invalid_argument(name + " has no generic CUDA moment layout");
    }

    const I64 features = feature_count_option(options, computed_features);
    if (features <= 0 || channel_species.empty() || channel_species.size() > 256
        || max_rank < 0 || max_rank > 20 || min_dist < 0.0
        || max_dist <= min_dist || soft_cutoff < 0.0 || soft_cutoff > hard_cutoff
        || hard_cutoff <= 0.0) {
        throw std::invalid_argument("invalid CUDA generic descriptor parameters");
    }
    const I64 channels = static_cast<I64>(channel_species.size());
    I64 moment_stride = 0;
    if (mode == 0) {
        I64 tensor_size = 1;
        for (int rank = 0; rank <= max_rank; ++rank) {
            moment_stride += tensor_size;
            tensor_size *= 3;
        }
    }
    graph.build_dpa(context, batch, host_batch, max_dist, true, false, false);

    DeviceBuffer<I32> d_species;
    DeviceBuffer<I32> d_channel_species;
    DeviceBuffer<I32> d_channel_radial;
    DeviceBuffer<double> d_center_weights;
    d_species.upload(species.data(), species.size(), context.stream(),
        "could not upload generic descriptor species");
    d_channel_species.upload(channel_species.data(), channel_species.size(), context.stream(),
        "could not upload generic descriptor channel species");
    d_channel_radial.upload(channel_radial.data(), channel_radial.size(), context.stream(),
        "could not upload generic descriptor channel indices");
    d_center_weights.upload(center_weights.data(), center_weights.size(), context.stream(),
        "could not upload SOAPTurbo center weights");

    const std::size_t size = static_cast<std::size_t>(batch.atoms())
        * static_cast<std::size_t>(features);
    double* output = context.output_buffer(size);
    auto* moment_workspace = mode == 0
        ? static_cast<double*>(context.workspace_buffer(
            static_cast<std::size_t>(batch.atoms()) * static_cast<std::size_t>(channels)
            * static_cast<std::size_t>(moment_stride) * sizeof(double)))
        : nullptr;
    if (size > 0) {
        check_cuda(cudaMemsetAsync(output, 0, size * sizeof(double), context.stream()),
            "could not clear generic CUDA descriptor output");
        constexpr unsigned block_size = 64;
        generic_moment_kernel<<<static_cast<unsigned>((batch.atoms() + block_size - 1) / block_size),
            block_size, 0, context.stream()>>>(
            batch.numbers(), graph.offsets(), graph.atoms(), graph.displacements(), graph.distance2(), d_species.get(),
            static_cast<int>(species.size()), d_channel_species.get(), d_channel_radial.get(),
            static_cast<int>(channels), max_rank, mode, max_order, min_dist, max_dist,
            soft_cutoff, hard_cutoff, radial_sigma, radial_weight, angular_weight,
            d_center_weights.get(), features, batch.atoms(), moment_stride,
            mtp_radial_basis_size, moment_workspace, output);
        check_cuda(cudaGetLastError(), "generic CUDA descriptor kernel launch failed");
    }
    const auto values = download_output_with_gil_release(context, size);
    return atom_result(values, batch.atoms(), features, name, options, false,
        std::vector<I64>(host_batch.offsets, host_batch.offsets + host_batch.structures + 1));
}


} // namespace

py::dict compute_extended_generic(
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
    return compute_generic_moment_descriptor(context, batch, graph, host_batch, name, options);
}

} // namespace mdescriptor::cuda
