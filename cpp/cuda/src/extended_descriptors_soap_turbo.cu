#include "extended_descriptors_common.cuh"


__device__ double turbo_radial_normalization(int n) {
    return sqrt(1.0 / (2.0 * static_cast<double>(n) + 5.0));
}

__device__ double turbo_smoothing_prefactor(
    double rj, double sigma, double soft, double hard, double nf) {
    if (hard == soft || soft - rj >= 4.0 * sigma) return 0.0;
    const double dr = hard - soft;
    const double sigma2 = sigma * sigma;
    return exp(-0.5 * (soft - rj) * (soft - rj)
        / (sigma2 + dr * dr / (nf * nf)));
}

__device__ void turbo_radial_coefficients_device(
    int type,
    double distance,
    bool central,
    int basis_kind,
    int basis_size,
    double rcut_hard,
    double rcut_soft,
    double nf,
    int radial_enhancement,
    double basis_sigma,
    double atom_sigma_r,
    double atom_sigma_r_scaling,
    double amplitude_scaling,
    double central_weight,
    const int* alpha_max,
    const double* transforms,
    double* result,
    double* primitive,
    double* filtered,
    double* raw) {
    for (int index = 0; index < basis_size; ++index) result[index] = 0.0;
    if (distance >= rcut_hard || (basis_kind == 1 && central)) return;

    const double hard = 1.0;
    const double soft = rcut_soft / rcut_hard;
    const double rj = distance / rcut_hard;
    const double dr = hard - soft;
    const double atom_sigma = atom_sigma_r / rcut_hard;
    const double atom_sigma_scaled = atom_sigma + atom_sigma_r_scaling * rj;
    const double sigma2 = atom_sigma_scaled * atom_sigma_scaled;
    double amplitude = 1.0 / atom_sigma_scaled;
    if (amplitude_scaling != 0.0) {
        const double rj2 = rj * rj;
        const double polynomial = 1.0 + 2.0 * rj2 * rj - 3.0 * rj2;
        if (polynomial <= 1e-10) return;
        amplitude *= pow(polynomial, amplitude_scaling);
    }
    if (central) amplitude *= central_weight;
    if (radial_enhancement == 1) {
        amplitude *= rj + sqrt(2.0 / kPi) * atom_sigma_scaled;
    } else if (radial_enhancement == 2) {
        amplitude *= rj * rj + sigma2
            + sqrt(8.0 / kPi) * atom_sigma_scaled * rj;
    }
    if (amplitude == 0.0) return;

    const int expansion_count = basis_kind == 1 ? basis_size - 1 : basis_size;
    for (int index = 0; index < basis_size; ++index) {
        primitive[index] = 0.0;
        filtered[index] = 0.0;
    }
    double integral_n = 0.0;
    double norm_n = 1.0;
    double norm_np1 = turbo_radial_normalization(-2);
    double integral_np1 = sqrt(kPi / 2.0) * atom_sigma_scaled
        * (erf((soft - rj) / (sqrt(2.0) * atom_sigma_scaled))
            - erf(-rj / (sqrt(2.0) * atom_sigma_scaled))) / norm_np1;
    double correction_soft = hard == soft ? 0.0
        : sigma2 / dr * exp(-0.5 * (soft - rj) * (soft - rj) / sigma2);
    double correction_hard = sigma2 * exp(-0.5 * rj * rj / sigma2);
    for (int n = -1; n <= expansion_count; ++n) {
        correction_soft *= dr;
        correction_hard *= hard;
        const double norm_np2 = turbo_radial_normalization(n);
        const double integral_np2 = sigma2 * static_cast<double>(n + 1)
                * norm_n / norm_np2 * integral_n
            - norm_np1 * (rj - hard) / norm_np2 * integral_np1
            + correction_soft / norm_np2 - correction_hard / norm_np2;
        if (n > 0) primitive[n - 1] = integral_np2;
        norm_n = norm_np1;
        norm_np1 = norm_np2;
        integral_n = integral_np1;
        integral_np1 = integral_np2;
    }

    const double prefactor = turbo_smoothing_prefactor(
        rj, atom_sigma_scaled, soft, hard, nf);
    if (prefactor != 0.0) {
        const double nf_width2 = dr * dr / (nf * nf);
        const double filtered_sigma = atom_sigma_scaled * dr / nf
            / sqrt(sigma2 + nf_width2);
        const double filtered_sigma2 = filtered_sigma * filtered_sigma;
        const double filtered_center = (sigma2 * soft + nf_width2 * rj)
            / (sigma2 + nf_width2);
        integral_n = 0.0;
        norm_n = 1.0;
        norm_np1 = turbo_radial_normalization(-2);
        integral_np1 = sqrt(kPi / 2.0) * filtered_sigma
            * (erf((hard - filtered_center) / (sqrt(2.0) * filtered_sigma))
                - erf((soft - filtered_center) / (sqrt(2.0) * filtered_sigma))) / norm_np1;
        double filtered_correction = filtered_sigma2 / dr
            * exp(-0.5 * (soft - filtered_center) * (soft - filtered_center)
                / filtered_sigma2);
        for (int n = -1; n <= expansion_count; ++n) {
            filtered_correction *= dr;
            const double norm_np2 = turbo_radial_normalization(n);
            const double integral_np2 = filtered_sigma2 * static_cast<double>(n + 1)
                    * norm_n / norm_np2 * integral_n
                - norm_np1 * (filtered_center - hard) / norm_np2 * integral_np1
                - filtered_correction / norm_np2;
            if (n > 0) filtered[n - 1] = integral_np2;
            norm_n = norm_np1;
            norm_np1 = norm_np2;
            integral_n = integral_np1;
            integral_np1 = integral_np2;
        }
    }

    if (basis_kind == 1) {
        const double sigma_star = sqrt(basis_sigma * basis_sigma + sigma2);
        primitive[basis_size - 1] = exp(-0.5 * rj * rj / (sigma_star * sigma_star))
            * sqrt(kPi / 2.0) * atom_sigma_scaled * basis_sigma / sigma_star
            * (1.0 + erf(basis_sigma / atom_sigma_scaled * rj
                / (sqrt(2.0) * sigma_star)))
            * sqrt(2.0 / basis_sigma) / pow(kPi, 0.25);
    }

    for (int index = 0; index < basis_size; ++index) {
        raw[index] = amplitude * (primitive[index] + prefactor * filtered[index]);
    }
    int transform_offset = 0;
    for (int previous = 0; previous < type; ++previous) {
        transform_offset += alpha_max[previous] * alpha_max[previous];
    }
    const double* transform = transforms + transform_offset;
    for (int target = 0; target < basis_size; ++target) {
        double value = 0.0;
        for (int source = 0; source < basis_size; ++source) {
            value += transform[target * basis_size + source] * raw[source];
        }
        result[target] = value * sqrt(rcut_hard);
    }
}

__device__ double turbo_associated_legendre(int l, int m, double x) {
    double pmm = 1.0;
    const double root = sqrt(fmax(0.0, 1.0 - x * x));
    for (int order = 1; order <= m; ++order) {
        pmm *= -(2.0 * order - 1.0) * root;
    }
    if (l == m) return pmm;
    double pmp1m = x * (2.0 * m + 1.0) * pmm;
    if (l == m + 1) return pmp1m;
    for (int degree = m + 2; degree <= l; ++degree) {
        const double value = (x * (2.0 * degree - 1.0) * pmp1m
            - (degree + m - 1.0) * pmm) / (degree - m);
        pmm = pmp1m;
        pmp1m = value;
    }
    return pmp1m;
}

__device__ void turbo_modified_spherical_bessel(
    int l_max, double x, double* values, double* semifactorial) {
    for (int l = 0; l <= l_max; ++l) semifactorial[l] = 1.0;
    for (int l = 1; l <= l_max; ++l) {
        semifactorial[l] = semifactorial[l - 1] * (2.0 * l + 1.0);
    }
    const double x2 = x * x;
    const double x4 = x2 * x2;
    const double xcut = 1e-7;
    double flm2 = 1.0;
    double flm1 = 0.0;
    if (x > 0.0) {
        flm2 = fabs((1.0 - exp(-2.0 * x2)) / (2.0 * x2));
        flm1 = fabs((x2 - 1.0 + exp(-2.0 * x2) * (x2 + 1.0)) / (2.0 * x4));
    }
    for (int l = 0; l <= l_max; ++l) {
        if (l == 0) {
            values[0] = x < xcut ? 1.0 - x2 : flm2;
        } else if (l == 1) {
            values[1] = x2 / 1000.0 < xcut ? (x2 - x4) / semifactorial[1] : flm1;
        } else if (pow(x2, l) / semifactorial[l] * l < xcut) {
            values[l] = pow(x2, l) / semifactorial[l];
        } else {
            values[l] = fabs(flm2 - (2.0 * l - 1.0) / x2 * flm1);
        }
        if (l >= 2) {
            flm2 = flm1;
            flm1 = values[l];
        }
    }
}

__device__ void turbo_angular_coefficients(
    int l_max,
    double distance,
    const double* displacement,
    double sigma,
    double rcut_hard,
    DeviceComplex* result,
    double* ilexp,
    double* semifactorial) {
    const int packed_count = 1 + l_max * (l_max + 1) / 2 + l_max;
    for (int index = 0; index < packed_count; ++index) result[index] = {0.0, 0.0};
    if (distance >= rcut_hard) return;
    const double x = distance < 1e-14 ? 1.0
        : fmax(-1.0, fmin(1.0, displacement[2] / distance));
    const double phi = distance < 1e-14 ? 0.0 : atan2(displacement[1], displacement[0]);
    turbo_modified_spherical_bessel(l_max, distance / sigma, ilexp, semifactorial);
    const double amplitude = rcut_hard * rcut_hard / (sigma * sigma);
    const double phase_real = phi == 0.0 ? 1.0 : cos(-phi);
    const double phase_imag = phi == 0.0 ? 0.0 : sin(-phi);
    int packed = 0;
    for (int l = 0; l <= l_max; ++l) {
        double phase_real_power = 1.0;
        double phase_imag_power = 0.0;
        for (int m = 0; m <= l; ++m) {
            if (m > 0) {
                const double next_real = phase_real_power * phase_real
                    - phase_imag_power * phase_imag;
                const double next_imag = phase_real_power * phase_imag
                    + phase_imag_power * phase_real;
                phase_real_power = next_real;
                phase_imag_power = next_imag;
            }
            const double factorial_minus = tgamma(static_cast<double>(l - m) + 1.0);
            const double factorial_plus = tgamma(static_cast<double>(l + m) + 1.0);
            const double prefactor = sqrt((2.0 * l + 1.0) / (4.0 * kPi)
                * factorial_minus / factorial_plus);
            const double scale = amplitude * prefactor
                * turbo_associated_legendre(l, m, x) * ilexp[l];
            result[packed++] = {scale * phase_real_power, scale * phase_imag_power};
        }
    }
}

__device__ void turbo_accumulate_atom(
    int center_type,
    int atom_type,
    double distance,
    const double* displacement,
    bool central,
    int basis_kind,
    int l_max,
    double rcut_hard,
    double rcut_soft,
    double nf,
    int radial_enhancement,
    double basis_sigma,
    const double* atom_sigma_r,
    const double* atom_sigma_r_scaling,
    const double* atom_sigma_t,
    const double* atom_sigma_t_scaling,
    const double* amplitude_scaling,
    const double* central_weights,
    const int* alpha_max,
    const int* channel_offsets,
    const double* transforms,
    DeviceComplex* coefficients) {
    constexpr int MaxBasis = 11;
    constexpr int MaxPacked = 231;
    const int basis_size = alpha_max[atom_type];
    double radial[MaxBasis]{};
    double primitive[MaxBasis]{};
    double filtered[MaxBasis]{};
    double raw[MaxBasis]{};
    turbo_radial_coefficients_device(
        atom_type, distance, central, basis_kind, basis_size, rcut_hard, rcut_soft,
        nf, radial_enhancement, basis_sigma, atom_sigma_r[atom_type], atom_sigma_r_scaling[atom_type],
        amplitude_scaling[atom_type], central_weights[center_type], alpha_max,
        transforms, radial, primitive, filtered, raw);
    const int packed_count = 1 + l_max * (l_max + 1) / 2 + l_max;
    DeviceComplex angular[MaxPacked]{};
    double ilexp[21]{};
    double semifactorial[21]{};
    const double sigma = atom_sigma_t[atom_type]
        + atom_sigma_t_scaling[atom_type] * distance;
    turbo_angular_coefficients(
        l_max, distance, displacement, sigma, rcut_hard,
        angular, ilexp, semifactorial);
    const int offset = channel_offsets[atom_type];
    for (int n = 0; n < basis_size; ++n) {
        DeviceComplex* target = coefficients + (offset + n) * packed_count;
        for (int index = 0; index < packed_count; ++index) {
            target[index] = complex_add(target[index], complex_scale(
                angular[index], 4.0 * kPi * radial[n]));
        }
    }
}

__global__ void soap_turbo_cuda_kernel(
    const I32* numbers,
    const I64* graph_offsets,
    const I32* graph_atoms,
    const I32* graph_shifts,
    const double* graph_displacements,
    const double* graph_distance2,
    const I32* species,
    int species_count,
    const I32* alpha_max,
    const I32* channel_offsets,
    const I32* central_allowed,
    const double* atom_sigma_r,
    const double* atom_sigma_r_scaling,
    const double* atom_sigma_t,
    const double* atom_sigma_t_scaling,
    const double* amplitude_scaling,
    const double* central_weights,
    const double* transforms,
    const double* gaussian_columns,
    int basis_kind,
    int l_max,
    double rcut_hard,
    double rcut_soft,
    double nf,
    int radial_enhancement,
    I64 channels,
    I64 packed_count,
    I64 dense_features,
    I64 features,
    const I64* compression_offsets,
    const I64* compression_sources,
    const double* compression_factors,
    I64 coefficient_stride,
    I64 atoms,
    double* coefficient_workspace,
    double* dense_workspace,
    double* output) {
    const I64 center = static_cast<I64>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (center >= atoms) return;
    const int center_type = species_index(numbers[center], species, species_count);
    if (center_type < 0 || central_allowed == nullptr || central_allowed[center_type] == 0) return;
    DeviceComplex* coefficients = reinterpret_cast<DeviceComplex*>(
        coefficient_workspace + center * coefficient_stride);
    for (I64 index = 0; index < channels * packed_count; ++index) {
        coefficients[index] = {0.0, 0.0};
    }
    const double origin[3] = {0.0, 0.0, 0.0};
    turbo_accumulate_atom(
        center_type, center_type, 0.0, origin, true, basis_kind, l_max,
        rcut_hard, rcut_soft, nf, radial_enhancement,
        atom_sigma_r[center_type] / rcut_hard, atom_sigma_r,
        atom_sigma_r_scaling, atom_sigma_t, atom_sigma_t_scaling,
        amplitude_scaling, central_weights, alpha_max, channel_offsets,
        transforms, coefficients);
    if (basis_kind == 1 && central_weights[center_type] != 0.0) {
        int transform_offset = 0;
        int gaussian_offset = 0;
        for (int type = 0; type < center_type; ++type) {
            transform_offset += alpha_max[type] * alpha_max[type];
            gaussian_offset += alpha_max[type];
        }
        const int basis_size = alpha_max[center_type];
        const double sigma_r = atom_sigma_r[center_type];
        const double sigma_t = atom_sigma_t[center_type];
        const double enhancement = radial_enhancement == 1
            ? sqrt(2.0 / kPi) * sigma_r / rcut_hard
            : radial_enhancement == 2
                ? sigma_r * sigma_r / (rcut_hard * rcut_hard) : 1.0;
        const double prefactor = enhancement * central_weights[center_type]
            * sqrt(4.0 * kPi) * pow(kPi, 0.25) * sqrt(sigma_r / 2.0)
            * pow(rcut_hard, 3.0) / (sigma_t * sigma_t * sigma_r);
        const double* transform = transforms + transform_offset;
        const double* column = gaussian_columns + gaussian_offset;
        const int coefficient_offset = channel_offsets[center_type] * packed_count;
        for (int n = 0; n < basis_size; ++n) {
            double value = 0.0;
            for (int source = 0; source < basis_size; ++source) {
                value += transform[n * basis_size + source] * column[source];
            }
            coefficients[coefficient_offset + n * packed_count].real += prefactor * value;
        }
    }
    const I64 begin = graph_offsets[center];
    const I64 end = graph_offsets[center + 1];
    for (I64 edge = begin; edge < end; ++edge) {
        const I32 atom = graph_atoms[edge];
        if (atom == center && graph_shifts[edge * 3] == 0
            && graph_shifts[edge * 3 + 1] == 0 && graph_shifts[edge * 3 + 2] == 0) continue;
        const int atom_type = species_index(numbers[atom], species, species_count);
        if (atom_type < 0) continue;
        const double distance = sqrt(fmax(0.0, graph_distance2[edge]));
        if (distance >= rcut_hard) continue;
        turbo_accumulate_atom(
            center_type, atom_type, distance, graph_displacements + edge * 3, false,
            basis_kind, l_max, rcut_hard, rcut_soft, nf, radial_enhancement,
            atom_sigma_r[atom_type] / rcut_hard,
            atom_sigma_r, atom_sigma_r_scaling, atom_sigma_t, atom_sigma_t_scaling,
            amplitude_scaling, central_weights, alpha_max, channel_offsets,
            transforms, coefficients);
    }

    double* row = output + center * features;
    double* dense = compression_offsets != nullptr && dense_features > 0
        ? dense_workspace + center * dense_features : row;
    I64 dense_index = 0;
    for (I64 first = 0; first < channels; ++first) {
        for (I64 second = first; second < channels; ++second) {
            for (int l = 0; l <= l_max; ++l) {
                double value = 0.0;
                const int packed_offset = l * (l + 1) / 2;
                const double multiplicity_pair = first == second ? 1.0 : sqrt(2.0);
                for (int m = 0; m <= l; ++m) {
                    const DeviceComplex left = coefficients[first * packed_count + packed_offset + m];
                    const DeviceComplex right = coefficients[second * packed_count + packed_offset + m];
                    value += multiplicity_pair * (m > 0 ? 2.0 : 1.0)
                        * (left.real * right.real + left.imag * right.imag);
                }
                dense[dense_index++] = value;
            }
        }
    }
    if (compression_offsets != nullptr && dense_features > 0
        && features != dense_features) {
        for (I64 feature = 0; feature < features; ++feature) {
            double value = 0.0;
            for (I64 term = compression_offsets[feature]; term < compression_offsets[feature + 1]; ++term) {
                value += compression_factors[term] * dense[compression_sources[term]];
            }
            row[feature] = value;
        }
    } else if (dense != row) {
        for (I64 feature = 0; feature < features; ++feature) row[feature] = dense[feature];
    }
    double norm = 0.0;
    for (I64 feature = 0; feature < features; ++feature) norm += row[feature] * row[feature];
    norm = sqrt(norm);
    if (norm < 1e-5) norm = 1.0;
    for (I64 feature = 0; feature < features; ++feature) row[feature] /= norm;
}

py::dict compute_soap_turbo_descriptor(
    CudaExecutionContext& context,
    DeviceBatch& batch,
    DeviceNeighborGraph& graph,
    const detail::StructureBatchView& host_batch,
    const py::dict& options) {
    const py::str payload_key("_cuda_payload");
    if (!options.contains(payload_key) || options[payload_key].is_none()) {
        throw std::invalid_argument("SOAPTurbo CUDA backend requires its prepared basis payload");
    }
    const py::dict payload = py::cast<py::dict>(options[payload_key]);
    const auto species = species_option(options);
    const auto alpha_max = py::cast<std::vector<I32>>(payload["alpha_max"]);
    const auto channel_offsets = py::cast<std::vector<I32>>(payload["channel_offsets"]);
    const auto central_allowed = py::cast<std::vector<I32>>(payload["central_allowed"]);
    const auto transforms = vector_child(payload, "basis_transforms");
    const auto gaussian_columns = vector_child(payload, "gaussian_columns");
    const auto compression_offsets = py::cast<std::vector<I64>>(payload["compression_offsets"]);
    const auto compression_sources = py::cast<std::vector<I64>>(payload["compression_sources"]);
    const auto compression_factors = vector_child(payload, "compression_factors");
    const std::string basis_name = option(options, "basis", std::string("poly3"));
    const int basis_kind = basis_name == "poly3gauss" ? 1 : 0;
    const int l_max = option(options, "l_max", 6);
    const double rcut_hard = option(options, "rcut_hard", 5.0);
    const double rcut_soft = option(options, "rcut_soft", rcut_hard);
    const double nf = option(options, "nf", 1.0);
    const int radial_enhancement = option(options, "radial_enhancement", 0);
    const auto atom_sigma_r = numeric_vector_option(options, "atom_sigma_r", 0.5, species.size());
    const auto atom_sigma_r_scaling = numeric_vector_option(options, "atom_sigma_r_scaling", 0.0, species.size());
    const auto atom_sigma_t = numeric_vector_option(options, "atom_sigma_t", 0.5, species.size());
    const auto atom_sigma_t_scaling = numeric_vector_option(options, "atom_sigma_t_scaling", 0.0, species.size());
    const auto amplitude_scaling = numeric_vector_option(options, "amplitude_scaling", 0.0, species.size());
    const auto central_weights = numeric_vector_option(options, "central_weight", 1.0, species.size());
    const I64 features = feature_count_option(options, 0);
    const I64 channels = channel_offsets.empty() ? 0 : channel_offsets.back();
    const I64 packed_count = payload.contains("packed_count")
        ? py::cast<I64>(payload["packed_count"])
        : 1 + l_max * (l_max + 1) / 2 + l_max;
    const I64 dense_features = payload.contains("dense_feature_count")
        ? py::cast<I64>(payload["dense_feature_count"])
        : channels * (channels + 1) / 2 * (l_max + 1);
    if (species.empty() || alpha_max.size() != species.size()
        || channel_offsets.size() != species.size() + 1
        || central_allowed.size() != species.size() || features <= 0
        || channels <= 0 || channels > 256 || l_max < 0 || l_max > 20
        || rcut_hard <= 0.0 || rcut_soft <= 0.0 || rcut_soft > rcut_hard
        || nf <= 0.0 || (basis_kind != 0 && basis_kind != 1)
        || transforms.empty() || (basis_kind == 1 && gaussian_columns.empty())
        || compression_offsets.empty()
        || compression_sources.size() != compression_factors.size()) {
        throw std::invalid_argument("invalid SOAPTurbo CUDA basis payload");
    }
    for (const I32 count : alpha_max) {
        if (count <= 0 || count > (basis_kind == 0 ? 10 : 11)) {
            throw std::invalid_argument("SOAPTurbo alpha_max exceeds the supported CUDA range");
        }
    }
    const bool identity = compression_sources.empty() && features == dense_features;
    if (!identity && (compression_offsets.size() != static_cast<std::size_t>(features + 1)
        || compression_offsets.back() != static_cast<I64>(compression_sources.size()))) {
        throw std::invalid_argument("invalid SOAPTurbo CUDA compression map");
    }
    graph.build_dpa(context, batch, host_batch, rcut_hard, true, false, false);

    DeviceBuffer<I32> d_species;
    DeviceBuffer<I32> d_alpha_max;
    DeviceBuffer<I32> d_channel_offsets;
    DeviceBuffer<I32> d_central_allowed;
    DeviceBuffer<double> d_atom_sigma_r;
    DeviceBuffer<double> d_atom_sigma_r_scaling;
    DeviceBuffer<double> d_atom_sigma_t;
    DeviceBuffer<double> d_atom_sigma_t_scaling;
    DeviceBuffer<double> d_amplitude_scaling;
    DeviceBuffer<double> d_central_weights;
    DeviceBuffer<double> d_transforms;
    DeviceBuffer<double> d_gaussian_columns;
    DeviceBuffer<I64> d_compression_offsets;
    DeviceBuffer<I64> d_compression_sources;
    DeviceBuffer<double> d_compression_factors;
    d_species.upload(species.data(), species.size(), context.stream(), "could not upload SOAPTurbo species");
    d_alpha_max.upload(alpha_max.data(), alpha_max.size(), context.stream(), "could not upload SOAPTurbo alpha_max");
    d_channel_offsets.upload(channel_offsets.data(), channel_offsets.size(), context.stream(), "could not upload SOAPTurbo channel offsets");
    d_central_allowed.upload(central_allowed.data(), central_allowed.size(), context.stream(), "could not upload SOAPTurbo central mask");
    d_atom_sigma_r.upload(atom_sigma_r.data(), atom_sigma_r.size(), context.stream(), "could not upload SOAPTurbo radial widths");
    d_atom_sigma_r_scaling.upload(atom_sigma_r_scaling.data(), atom_sigma_r_scaling.size(), context.stream(), "could not upload SOAPTurbo radial scaling");
    d_atom_sigma_t.upload(atom_sigma_t.data(), atom_sigma_t.size(), context.stream(), "could not upload SOAPTurbo angular widths");
    d_atom_sigma_t_scaling.upload(atom_sigma_t_scaling.data(), atom_sigma_t_scaling.size(), context.stream(), "could not upload SOAPTurbo angular scaling");
    d_amplitude_scaling.upload(amplitude_scaling.data(), amplitude_scaling.size(), context.stream(), "could not upload SOAPTurbo amplitudes");
    d_central_weights.upload(central_weights.data(), central_weights.size(), context.stream(), "could not upload SOAPTurbo center weights");
    d_transforms.upload(transforms.data(), transforms.size(), context.stream(), "could not upload SOAPTurbo radial transforms");
    d_gaussian_columns.upload(gaussian_columns.data(), gaussian_columns.size(), context.stream(), "could not upload SOAPTurbo Gaussian column");
    d_compression_offsets.upload(compression_offsets.data(), compression_offsets.size(), context.stream(), "could not upload SOAPTurbo compression offsets");
    d_compression_sources.upload(compression_sources.data(), compression_sources.size(), context.stream(), "could not upload SOAPTurbo compression sources");
    d_compression_factors.upload(compression_factors.data(), compression_factors.size(), context.stream(), "could not upload SOAPTurbo compression factors");
    const std::size_t size = static_cast<std::size_t>(batch.atoms()) * static_cast<std::size_t>(features);
    double* output = context.output_buffer(size);
    const I64 coefficient_stride = channels * packed_count * 2;
    const std::size_t coefficient_values = static_cast<std::size_t>(batch.atoms())
        * static_cast<std::size_t>(coefficient_stride);
    const std::size_t dense_values = identity ? 0U
        : static_cast<std::size_t>(batch.atoms()) * static_cast<std::size_t>(dense_features);
    if (coefficient_values > std::numeric_limits<std::size_t>::max() - dense_values
        || coefficient_values + dense_values > std::numeric_limits<std::size_t>::max() / sizeof(double)) {
        throw CudaOutOfMemory("SOAPTurbo CUDA workspace is too large");
    }
    auto* workspace = static_cast<unsigned char*>(context.workspace_buffer(
        (coefficient_values + dense_values) * sizeof(double)));
    auto* coefficient_workspace = reinterpret_cast<double*>(workspace);
    auto* dense_workspace = identity ? nullptr
        : reinterpret_cast<double*>(workspace + coefficient_values * sizeof(double));
    if (size > 0) {
        check_cuda(cudaMemsetAsync(output, 0, size * sizeof(double), context.stream()),
            "could not clear SOAPTurbo output");
        constexpr unsigned block_size = 64;
        soap_turbo_cuda_kernel<<<static_cast<unsigned>((batch.atoms() + block_size - 1) / block_size),
            block_size, 0, context.stream()>>>(
            batch.numbers(), graph.offsets(), graph.atoms(), graph.shifts(), graph.displacements(),
            graph.distance2(), d_species.get(), static_cast<int>(species.size()), d_alpha_max.get(),
            d_channel_offsets.get(), d_central_allowed.get(), d_atom_sigma_r.get(),
            d_atom_sigma_r_scaling.get(), d_atom_sigma_t.get(), d_atom_sigma_t_scaling.get(),
            d_amplitude_scaling.get(), d_central_weights.get(), d_transforms.get(),
            d_gaussian_columns.get(), basis_kind, l_max, rcut_hard, rcut_soft, nf,
            radial_enhancement, channels, packed_count, dense_features, features,
            identity ? nullptr : d_compression_offsets.get(),
            d_compression_sources.get(), d_compression_factors.get(), coefficient_stride,
            batch.atoms(), coefficient_workspace, dense_workspace, output);
        check_cuda(cudaGetLastError(), "SOAPTurbo CUDA kernel launch failed");
    }
    const auto values = download_output_with_gil_release(context, size);
    return atom_result(values, batch.atoms(), features, "SOAPTurbo", options, false,
        std::vector<I64>(host_batch.offsets, host_batch.offsets + host_batch.structures + 1));
}

} // namespace

py::dict compute_extended_soap_turbo(
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
    return compute_soap_turbo_descriptor(context, batch, graph, host_batch, options);
}

} // namespace mdescriptor::cuda
