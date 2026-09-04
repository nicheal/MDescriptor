#include "extended_descriptors_common.cuh"

__device__ double integer_power_device(double value, int exponent) {
    double result = 1.0;
    for (int index = 0; index < exponent; ++index) result *= value;
    return result;
}

__global__ void ead_kernel(
    const I32* numbers,
    const I64* graph_offsets,
    const I32* graph_atoms,
    const double* graph_displacements,
    const double* graph_distance2,
    double cutoff,
    const double* eta,
    int n_eta,
    const double* centers,
    int n_centers,
    int max_degree,
    I64 atoms,
    double* output) {
    const I64 center = static_cast<I64>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (center >= atoms) return;
    const I64 begin = graph_offsets[center];
    const I64 end = graph_offsets[center + 1];
    const int radial_count = n_eta * n_centers;
    double* target = output + center * static_cast<I64>((max_degree + 1) * radial_count);
    for (int degree = 0; degree <= max_degree; ++degree) {
        for (int eta_index = 0; eta_index < n_eta; ++eta_index) {
            for (int center_index = 0; center_index < n_centers; ++center_index) {
                double squared_sum = 0.0;
                for (int power_degree = 0; power_degree <= degree; ++power_degree) {
                    for (int lx = 0; lx <= power_degree; ++lx) {
                        for (int ly = 0; ly <= power_degree - lx; ++ly) {
                            const int lz = power_degree - lx - ly;
                            double term = 0.0;
                            for (I64 edge = begin; edge < end; ++edge) {
                                const double distance2 = fmax(0.0, graph_distance2[edge]);
                                const double distance = sqrt(distance2);
                                if (distance <= 1e-12 || distance >= cutoff) continue;
                                const double smooth = 0.5 * (1.0 + cos(kPi * distance / cutoff));
                                const double delta = distance - centers[center_index];
                                const double gaussian = exp(-eta[eta_index] * delta * delta);
                                const double* vector = graph_displacements + edge * 3;
                                const double monomial = integer_power_device(vector[0], lx)
                                    * integer_power_device(vector[1], ly)
                                    * integer_power_device(vector[2], lz)
                                    / sqrt(tgamma(lx + 1.0) * tgamma(ly + 1.0) * tgamma(lz + 1.0));
                                term += monomial * gaussian * static_cast<double>(
                                    numbers[graph_atoms[edge]]) * smooth;
                            }
                            squared_sum += term * term;
                        }
                    }
                }
                target[degree * radial_count + eta_index * n_centers + center_index]
                    = tgamma(degree + 1.0) * squared_sum
                    / pow(cutoff, 2.0 * degree);
            }
        }
    }
}


py::dict compute_ead_descriptor(
    CudaExecutionContext& context,
    DeviceBatch& batch,
    DeviceNeighborGraph& graph,
    const detail::StructureBatchView& host_batch,
    const py::dict& options) {
    const py::dict parameters = child_dict(options, "parameters");
    const int max_degree = option(parameters, "L", 3);
    const auto eta = vector_child(parameters, "eta");
    const auto centers = vector_child(parameters, "Rs");
    const double cutoff = option(options, "Rc", 6.0);
    if (max_degree < 0 || max_degree > 8 || cutoff <= 0.0
        || eta.empty() || centers.empty()) {
        throw std::invalid_argument("invalid CUDA EAD parameters");
    }
    const I64 computed = static_cast<I64>(max_degree + 1) * eta.size() * centers.size();
    const I64 features = payload_or_option_feature_count(options, computed, "EAD");
    if (features != computed) throw std::invalid_argument("CUDA EAD feature count mismatch");
    graph.build_dpa(context, batch, host_batch, cutoff, true, false, false);
    DeviceBuffer<double> d_eta;
    DeviceBuffer<double> d_centers;
    d_eta.upload(eta.data(), eta.size(), context.stream(), "could not upload EAD eta");
    d_centers.upload(centers.data(), centers.size(), context.stream(), "could not upload EAD centers");
    const std::size_t size = static_cast<std::size_t>(batch.atoms())
        * static_cast<std::size_t>(features);
    double* output = context.output_buffer(size);
    if (size > 0) {
        check_cuda(cudaMemsetAsync(output, 0, size * sizeof(double), context.stream()),
            "could not clear CUDA EAD output");
        constexpr unsigned block_size = 64;
        ead_kernel<<<static_cast<unsigned>((batch.atoms() + block_size - 1) / block_size),
            block_size, 0, context.stream()>>>(
            batch.numbers(), graph.offsets(), graph.atoms(), graph.displacements(),
            graph.distance2(), cutoff, d_eta.get(), static_cast<int>(eta.size()),
            d_centers.get(), static_cast<int>(centers.size()), max_degree,
            batch.atoms(), output);
        check_cuda(cudaGetLastError(), "CUDA EAD kernel launch failed");
    }
    const auto values = download_output_with_gil_release(context, size);
    return atom_result(values, batch.atoms(), features, "EAD", options, false,
        std::vector<I64>(host_batch.offsets, host_batch.offsets + host_batch.structures + 1));
}

__device__ double lode_fourier_device(double k_norm, double sigma, int exponent) {
    if (k_norm <= 1e-12) return 0.0;
    const double sigma2 = sigma * sigma;
    const double x = 0.5 * k_norm * k_norm * sigma2;
    if (exponent == 1) return 4.0 * kPi * exp(-x) / (k_norm * k_norm);
    const double p_eff = 3.0 - exponent;
    const double factor = pow(kPi, 1.5) * pow(2.0 * sigma2, 0.5 * p_eff)
        / tgamma(0.5 * exponent);
    const double root_x = sqrt(x);
    double value = 0.0;
    if (exponent == 2) {
        value = sqrt(kPi / x) * erfc(root_x);
    } else if (exponent == 3) {
        double term = 1.0;
        double series = 0.0;
        if (x < 1.0) {
            for (int index = 1; index < 200; ++index) {
                term *= -x;
                const double add = term / (index * index);
                series += add;
                if (fabs(add) < 2e-16 * fmax(1.0, fabs(series))) break;
            }
            value = -0.5772156649015329 - log(x) - series;
        } else {
            series = 1.0;
            for (int index = 1; index < 100; ++index) {
                term *= -static_cast<double>(index) / x;
                series += term;
                if (fabs(term) > fabs(series)) break;
            }
            value = exp(-x) * series / x;
        }
    } else if (exponent == 4) {
        value = 2.0 * (exp(-x) - sqrt(kPi * x) * erfc(root_x));
    } else if (exponent == 5) {
        double e1 = 0.0;
        double term = 1.0;
        double series = 0.0;
        if (x < 1.0) {
            for (int index = 1; index < 200; ++index) {
                term *= -x;
                const double add = term / (index * index);
                series += add;
                if (fabs(add) < 2e-16 * fmax(1.0, fabs(series))) break;
            }
            e1 = -0.5772156649015329 - log(x) - series;
        } else {
            series = 1.0;
            for (int index = 1; index < 100; ++index) {
                term *= -static_cast<double>(index) / x;
                series += term;
                if (fabs(term) > fabs(series)) break;
            }
            e1 = exp(-x) * series / x;
        }
        value = exp(-x) - x * e1;
    } else if (exponent == 6) {
        double term = 1.0;
        double series = 0.0;
        if (x < 1.0) {
            for (int index = 1; index < 200; ++index) {
                term *= -x;
                const double add = term / (index * index);
                series += add;
                if (fabs(add) < 2e-16 * fmax(1.0, fabs(series))) break;
            }
            series = -0.5772156649015329 - log(x) - series;
        } else {
            series = 1.0;
            for (int index = 1; index < 100; ++index) {
                term *= -static_cast<double>(index) / x;
                series += term;
                if (fabs(term) > fabs(series)) break;
            }
            series = exp(-x) * series / x;
        }
        value = ((2.0 - 4.0 * x) * exp(-x)
            + 4.0 * sqrt(kPi) * pow(x, 1.5) * erfc(root_x)) / 3.0;
    } else if (exponent == 7) {
        double term = 1.0;
        double series = 0.0;
        if (x < 1.0) {
            for (int index = 1; index < 200; ++index) {
                term *= -x;
                const double add = term / (index * index);
                series += add;
                if (fabs(add) < 2e-16 * fmax(1.0, fabs(series))) break;
            }
            series = -0.5772156649015329 - log(x) - series;
        } else {
            series = 1.0;
            for (int index = 1; index < 100; ++index) {
                term *= -static_cast<double>(index) / x;
                series += term;
                if (fabs(term) > fabs(series)) break;
            }
            series = exp(-x) * series / x;
        }
        value = (1.0 - x) * exp(-x) / 2.0 + x * x * series / 2.0;
    } else if (exponent == 8) {
        value = -2.0 / 15.0 * ((-3.0 + 2.0 * x - 4.0 * x * x) * exp(-x)
            + 4.0 * sqrt(kPi) * pow(x, 2.5) * erfc(root_x));
    } else if (exponent == 9) {
        double term = 1.0;
        double series = 0.0;
        if (x < 1.0) {
            for (int index = 1; index < 200; ++index) {
                term *= -x;
                const double add = term / (index * index);
                series += add;
                if (fabs(add) < 2e-16 * fmax(1.0, fabs(series))) break;
            }
            series = -0.5772156649015329 - log(x) - series;
        } else {
            series = 1.0;
            for (int index = 1; index < 100; ++index) {
                term *= -static_cast<double>(index) / x;
                series += term;
                if (fabs(term) > fabs(series)) break;
            }
            series = exp(-x) * series / x;
        }
        value = (x * x - x + 2.0) * exp(-x) / 6.0
            - x * x * x * series / 6.0;
    }
    return factor * value;
}

__device__ double lode_hyp1f1_device(double a, double b, double x) {
    if (x >= 0.0) return positive_hypergeometric(a, b, x);
    const double magnitude = -x;
    const double difference = b - a;
    const double rounded = round(difference);
    if (difference <= 0.0 && fabs(difference - rounded) <= 1e-12) {
        const int degree = static_cast<int>(-rounded);
        double polynomial = 1.0;
        double term = 1.0;
        for (int index = 1; index <= degree; ++index) {
            term *= (-degree + index - 1.0) * magnitude
                / ((b + index - 1.0) * index);
            polynomial += term;
        }
        return exp(-magnitude) * polynomial;
    }
    if (magnitude > 8.0 && difference < 0.0) {
        // Kummer's transformation turns the alternating negative series into
        // a positive-argument series with a non-positive first parameter.
        double term = 1.0;
        double sum = 1.0;
        for (int index = 1; index <= 500; ++index) {
            term *= (difference + index - 1.0) * magnitude
                / ((b + index - 1.0) * index);
            sum += term;
            if (fabs(term) <= fabs(sum) * 1e-15) break;
        }
        return exp(-magnitude) * sum;
    }
    double term = 1.0;
    double sum = 1.0;
    for (int index = 1; index <= 500; ++index) {
        term *= (a + index - 1.0) * x
            / ((b + index - 1.0) * index);
        sum += term;
        if (fabs(term) <= fabs(sum) * 2e-15) break;
    }
    return sum;
}

__device__ double lode_radial_value_device(
    double k_norm,
    int angular,
    int target_radial,
    int radial_count,
    const double* widths,
    const double* lode_prefactors,
    const double* gamma_a,
    const double* gamma_b,
    const double* orthonormalization) {
    double value = 0.0;
    for (int raw = 0; raw < radial_count; ++raw) {
        const double sigma = widths[raw];
        const double k_sigma_sqrt2 = k_norm * sigma / sqrt(2.0);
        const double angular_factor = pow(k_sigma_sqrt2, angular);
        const double factor = sqrt(kPi) / sqrt(2.0)
            * lode_prefactors[raw] * angular_factor;
        const double z = -0.5 * k_norm * k_norm * sigma * sigma;
        const double a = 0.5 * (raw + angular + 3.0);
        const double b = angular + 1.5;
        const double raw_value = gamma_a[angular * radial_count + raw]
            / gamma_b[angular] * lode_hyp1f1_device(a, b, z) * factor;
        value += raw_value * orthonormalization[
            (angular * radial_count + raw) * radial_count + target_radial];
    }
    return value;
}

__global__ void lode_exact_kernel(
    const I32* numbers,
    const double* positions,
    const I64* offsets,
    const I32* species,
    int species_count,
    int radial_count,
    int max_angular,
    double density_width,
    int exponent,
    const I64* k_offsets,
    const double* k_vectors,
    const double* k_norms,
    const double* widths,
    const double* lode_prefactors,
    const double* gamma_a,
    const double* gamma_b,
    const double* orthonormalization,
    const double* cells,
    I64 structures,
    I64 atoms,
    I64 features,
    double* output) {
    const I64 center = static_cast<I64>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (center >= atoms) return;
    I64 structure = 0;
    while (structure + 1 < structures && offsets[structure + 1] <= center) ++structure;
    const int center_type = species_index(numbers[center], species, species_count);
    if (center_type < 0) return;
    const I64 begin = offsets[structure];
    const I64 end = offsets[structure + 1];
    const I64 k_begin = k_offsets[structure];
    const I64 k_end = k_offsets[structure + 1];
    const double volume = cell_volume_device(cells + structure * 9);
    if (volume <= 0.0) return;
    const I64 angular_block = static_cast<I64>(max_angular + 1)
        * static_cast<I64>(max_angular + 1);
    const I64 channel_stride = angular_block * radial_count;
    double* target = output + center * features
        + static_cast<I64>(center_type) * species_count * channel_stride;
    const double global_factor = 4.0 * kPi / volume;
    const double center_x = positions[center * 3 + 0];
    const double center_y = positions[center * 3 + 1];
    const double center_z = positions[center * 3 + 2];
    for (int neighbor_type = 0; neighbor_type < species_count; ++neighbor_type) {
        for (I64 k_index = k_begin; k_index < k_end; ++k_index) {
            const double kx = k_vectors[k_index * 3 + 0];
            const double ky = k_vectors[k_index * 3 + 1];
            const double kz = k_vectors[k_index * 3 + 2];
            const double k_norm = k_norms[k_index];
            const double center_phase = kx * center_x + ky * center_y + kz * center_z;
            const double center_cos = cos(center_phase);
            const double center_sin = sin(center_phase);
            double sum_cos = 0.0;
            double sum_sin = 0.0;
            for (I64 atom = begin; atom < end; ++atom) {
                if (species_index(numbers[atom], species, species_count) != neighbor_type) {
                    continue;
                }
                const double phase = kx * positions[atom * 3 + 0]
                    + ky * positions[atom * 3 + 1]
                    + kz * positions[atom * 3 + 2];
                sum_cos += cos(phase);
                sum_sin += sin(phase);
            }
            const double density = global_factor * lode_fourier_device(
                k_norm, density_width, exponent);
            const double even_weight = density * 2.0 * (
                center_cos * sum_cos + center_sin * sum_sin);
            const double odd_weight = density * 2.0 * (
                center_sin * sum_cos - center_cos * sum_sin);
            double direction[3] = {kx / k_norm, ky / k_norm, kz / k_norm};
            double harmonics[441]{};
            harmonic_values<20>(direction, harmonics, max_angular);
            for (int angular = 0; angular <= max_angular; ++angular) {
                const double phase = angular % 4 == 0 ? 1.0
                    : angular % 4 == 1 ? -1.0
                    : angular % 4 == 2 ? -1.0 : 1.0;
                const double selected_weight = angular % 2 == 0
                    ? even_weight : odd_weight;
                const I64 base = static_cast<I64>(angular * angular) * radial_count;
                for (int m = -angular; m <= angular; ++m) {
                    const double harmonic = harmonics[angular * angular + angular + m];
                    for (int radial = 0; radial < radial_count; ++radial) {
                        target[static_cast<I64>(neighbor_type) * channel_stride
                            + base + static_cast<I64>(angular + m) * radial_count + radial]
                            += phase * selected_weight * harmonic
                            * lode_radial_value_device(
                                k_norm, angular, radial, radial_count, widths,
                                lode_prefactors, gamma_a, gamma_b,
                                orthonormalization);
                    }
                }
            }
        }
    }
}


py::dict compute_lode_descriptor(
    CudaExecutionContext& context,
    DeviceBatch& batch,
    DeviceNeighborGraph& graph,
    const detail::StructureBatchView& host_batch,
    const py::dict& options) {
    (void)graph;
    const auto species = species_option(options);
    if (species.empty()) throw std::invalid_argument("LODE species must not be empty");
    const double cutoff = option(options, "cutoff", 6.0);
    const double sigma = option(options, "density_width", 0.3);
    const int max_radial = option(options, "max_radial", 6);
    const int max_angular = option(options, "max_angular", 4);
    const double k_cutoff = option(options, "k_cutoff", 2.5);
    const int exponent = option(options, "exponent", 1);
    const double radial_radius = option(options, "radial_radius", cutoff);
    if (cutoff <= 0.0 || sigma <= 0.0 || max_radial < 0 || max_angular < 0
        || max_angular > 20 || k_cutoff <= 0.0 || exponent < 1 || exponent > 9
        || radial_radius <= 0.0) {
        throw std::invalid_argument("invalid CUDA LODE parameters");
    }
    const int radial_count = max_radial + 1;
    const int angular_block = (max_angular + 1) * (max_angular + 1);
    const I64 computed = static_cast<I64>(species.size()) * species.size()
        * radial_count * angular_block;
    const I64 features = payload_or_option_feature_count(
        options, computed, "LodeSphericalExpansion");
    if (features != computed) throw std::invalid_argument("CUDA LODE feature count mismatch");

    // The LODE density is defined in reciprocal space.  Build only the small
    // per-structure k-vector and radial-basis metadata on the host; phases,
    // species sums, angular projection, and the final coefficients stay in
    // the CUDA kernel.
    std::vector<I64> k_offsets(static_cast<std::size_t>(host_batch.structures) + 1U, 0);
    std::vector<double> k_vectors;
    std::vector<double> k_norms;
    for (I64 structure = 0; structure < host_batch.structures; ++structure) {
        const auto vectors = detail::make_k_vectors(
            host_batch.cells + structure * 9, k_cutoff);
        if (vectors.empty()) {
            throw std::invalid_argument(
                "no LODE reciprocal vectors for the current cell and k_cutoff");
        }
        for (const auto& vector : vectors) {
            k_vectors.push_back(vector.vector[0]);
            k_vectors.push_back(vector.vector[1]);
            k_vectors.push_back(vector.vector[2]);
            k_norms.push_back(vector.norm);
        }
        k_offsets[static_cast<std::size_t>(structure + 1)] =
            static_cast<I64>(k_norms.size());
    }
    std::vector<double> widths(static_cast<std::size_t>(radial_count), 0.0);
    std::vector<double> lode_prefactors(static_cast<std::size_t>(radial_count), 0.0);
    std::vector<double> gamma_a(static_cast<std::size_t>(max_angular + 1) * radial_count, 0.0);
    std::vector<double> gamma_b(static_cast<std::size_t>(max_angular + 1), 0.0);
    std::vector<double> orthonormalization(
        static_cast<std::size_t>(max_angular + 1) * radial_count * radial_count, 0.0);
    for (int angular = 0; angular <= max_angular; ++angular) {
        const detail::GtoRadialBasis basis(radial_count, radial_radius, angular);
        if (angular == 0) {
            widths = basis.widths;
            lode_prefactors = basis.lode_prefactors;
        }
        std::copy(
            basis.gamma_a.begin(), basis.gamma_a.end(),
            gamma_a.begin() + static_cast<std::size_t>(angular) * radial_count);
        gamma_b[static_cast<std::size_t>(angular)] = basis.gamma_b;
        for (int raw = 0; raw < radial_count; ++raw) {
            for (int target = 0; target < radial_count; ++target) {
                orthonormalization[
                    (static_cast<std::size_t>(angular) * radial_count + raw) * radial_count
                        + target] = basis.orthonormalization[static_cast<std::size_t>(raw)]
                            [static_cast<std::size_t>(target)];
            }
        }
    }
    DeviceBuffer<I32> d_species;
    DeviceBuffer<I64> d_k_offsets;
    DeviceBuffer<double> d_k_vectors;
    DeviceBuffer<double> d_k_norms;
    DeviceBuffer<double> d_widths;
    DeviceBuffer<double> d_lode_prefactors;
    DeviceBuffer<double> d_gamma_a;
    DeviceBuffer<double> d_gamma_b;
    DeviceBuffer<double> d_orthonormalization;
    d_species.upload(species.data(), species.size(), context.stream(), "could not upload LODE species");
    d_k_offsets.upload(k_offsets.data(), k_offsets.size(), context.stream(), "could not upload LODE k offsets");
    d_k_vectors.upload(k_vectors.data(), k_vectors.size(), context.stream(), "could not upload LODE k vectors");
    d_k_norms.upload(k_norms.data(), k_norms.size(), context.stream(), "could not upload LODE k norms");
    d_widths.upload(widths.data(), widths.size(), context.stream(), "could not upload LODE radial widths");
    d_lode_prefactors.upload(
        lode_prefactors.data(), lode_prefactors.size(), context.stream(),
        "could not upload LODE radial prefactors");
    d_gamma_a.upload(gamma_a.data(), gamma_a.size(), context.stream(), "could not upload LODE gamma values");
    d_gamma_b.upload(gamma_b.data(), gamma_b.size(), context.stream(), "could not upload LODE gamma denominators");
    d_orthonormalization.upload(
        orthonormalization.data(), orthonormalization.size(), context.stream(),
        "could not upload LODE radial orthonormalization");
    const std::size_t size = static_cast<std::size_t>(batch.atoms()) * static_cast<std::size_t>(features);
    double* output = context.output_buffer(size);
    if (size > 0) {
        check_cuda(cudaMemsetAsync(output, 0, size * sizeof(double), context.stream()),
            "could not clear CUDA LODE output");
        constexpr unsigned block_size = 64;
        lode_exact_kernel<<<static_cast<unsigned>((batch.atoms() + block_size - 1) / block_size),
            block_size, 0, context.stream()>>>(
            batch.numbers(), batch.positions(), batch.offsets(), d_species.get(),
            static_cast<int>(species.size()), radial_count, max_angular, sigma, exponent,
            d_k_offsets.get(), d_k_vectors.get(), d_k_norms.get(), d_widths.get(),
            d_lode_prefactors.get(), d_gamma_a.get(), d_gamma_b.get(),
            d_orthonormalization.get(), batch.cells(), batch.structures(), batch.atoms(),
            features, output);
        check_cuda(cudaGetLastError(), "CUDA LODE kernel launch failed");
    }
    const auto values = download_output_with_gil_release(context, size);
    return atom_result(values, batch.atoms(), features, "LodeSphericalExpansion", options, false,
        std::vector<I64>(host_batch.offsets, host_batch.offsets + host_batch.structures + 1));
}

} // namespace

py::dict compute_extended_ead_lode(
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
    if (name == "EAD") {
        return compute_ead_descriptor(context, batch, graph, host_batch, options);
    }
    return compute_lode_descriptor(context, batch, graph, host_batch, options);
}

} // namespace mdescriptor::cuda

