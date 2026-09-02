#include "mdescriptor/cuda/local_descriptors.hpp"

#include "mdescriptor/neighbor.hpp"
#include "local_layout.hpp"
#include "local_spherical_common.hpp"

#include <cuda_runtime.h>

#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace mdescriptor::cuda {
namespace {

using mdescriptor::detail::GtoRadialBasis;
using mdescriptor::detail::RadialBasisSet;
using mdescriptor::LocalDescriptorKind;
using mdescriptor::LocalDescriptorOptions;

void check_cuda(cudaError_t status, const char* operation) {
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

__device__ int species_index(
    std::int32_t number,
    const std::int32_t* species,
    int species_count) {
    for (int index = 0; index < species_count; ++index) {
        if (species[index] == number) {
            return index;
        }
    }
    return -1;
}

template <int MaxAngular>
__device__ void harmonic_values(
    const double* vector,
    double* output) {
    // Compute the complete real-harmonic block once per graph edge.  The old
    // coefficient kernel recomputed the Legendre recurrence independently for
    // every (l, m) coefficient of that edge.
    constexpr int max_angular = MaxAngular;
    double legendre[(MaxAngular + 1) * (MaxAngular + 2) / 2]{};
    const double norm = sqrt(
        vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2]);
    double direction[3] = {vector[0], vector[1], vector[2]};
    if (norm < 1e-6) {
        direction[0] = 0.0;
        direction[1] = 0.0;
        direction[2] = 1.0;
    } else {
        direction[0] /= norm;
        direction[1] /= norm;
        direction[2] /= norm;
    }
    auto legendre_index = [](int angular, int m) {
        return m + angular * (angular + 1) / 2;
    };
    constexpr double sqrt_1_over_2pi = 0.398942280401432677939946059934;
    constexpr double sqrt_3 = 1.732050807568877293527446341505872;
    constexpr double sqrt_3_over_2 = 1.224744871391589049098642;
    const double xy = hypot(direction[0], direction[1]);
    const double cos_theta = direction[2];
    const double sin_theta = xy;
    legendre[legendre_index(0, 0)] = sqrt_1_over_2pi;
    if (max_angular > 0) {
        legendre[legendre_index(1, 0)] = cos_theta * sqrt_3 * sqrt_1_over_2pi;
        double value = -sqrt_3_over_2 * sin_theta * sqrt_1_over_2pi;
        legendre[legendre_index(1, 1)] = value;
        for (int angular = 2; angular <= max_angular; ++angular) {
            for (int m = 0; m < angular - 1; ++m) {
                const double ls = static_cast<double>(angular * angular);
                const double lm1s = static_cast<double>((angular - 1) * (angular - 1));
                const double ms = static_cast<double>(m * m);
                const double a = sqrt((4.0 * ls - 1.0) / (ls - ms));
                const double b = -sqrt((lm1s - ms) / (4.0 * lm1s - 1.0));
                legendre[legendre_index(angular, m)] = a * (
                    cos_theta * legendre[legendre_index(angular - 1, m)]
                    + b * legendre[legendre_index(angular - 2, m)]);
            }
            legendre[legendre_index(angular, angular - 1)] = cos_theta
                * sqrt(2.0 * angular + 1.0) * value;
            value *= -sqrt(1.0 + 0.5 / angular) * sin_theta;
            legendre[legendre_index(angular, angular)] = value;
        }
    }
    for (int angular = 0; angular <= max_angular; ++angular) {
        output[angular * angular + angular] = legendre[legendre_index(angular, 0)]
            / 1.414213562373095048801688724209698079;
    }
    const double cos_phi = xy > DBL_EPSILON ? direction[0] / xy : 1.0;
    const double sin_phi = xy > DBL_EPSILON ? direction[1] / xy : 0.0;
    double cos_previous = 1.0;
    double sin_previous = 0.0;
    double cos_current = -cos_phi;
    double sin_current = sin_phi;
    const double minus_two_cos = -2.0 * cos_phi;
    for (int m = 1; m <= max_angular; ++m) {
        const double sin_m = minus_two_cos * sin_previous - sin_current;
        const double cos_m = minus_two_cos * cos_previous - cos_current;
        sin_current = sin_previous;
        sin_previous = sin_m;
        cos_current = cos_previous;
        cos_previous = cos_m;
        for (int angular = m; angular <= max_angular; ++angular) {
            output[angular * angular + angular + m]
                = legendre[legendre_index(angular, m)] * cos_m;
            output[angular * angular + angular - m]
                = legendre[legendre_index(angular, m)] * sin_m;
        }
    }
}

__device__ double positive_hypergeometric(double a, double b, double x) {
    if (x > 30.0) {
        double sum = 1.0;
        double term = 1.0;
        for (int index = 1; index <= 30; ++index) {
            term = -term * (b - a + index - 1.0) * (a - index) / (x * index);
            sum += term;
        }
        return sum;
    }
    double sum = 1.0;
    double term = 1.0;
    for (int index = 1; index <= 500; ++index) {
        term *= (a + index - 1.0) * x / ((b + index - 1.0) * index);
        sum += term;
        if (fabs(term) <= fabs(sum) * 2e-15) {
            break;
        }
    }
    return sum;
}

__device__ double radial_value(
    double distance,
    int angular,
    int target_radial,
    int radial_count,
    double density_width,
    const double* gto_constants,
    const double* gamma_a,
    const double* gamma_b,
    const double* orthonormalization) {
    const double density_width2 = density_width * density_width;
    const double density_constant = 1.0 / (2.0 * density_width2);
    const double global_factor = pow(3.141592653589793238462643383279502884 / density_width2, 0.75);
    const double c_r = density_constant * distance;
    const double factor = global_factor * exp(-distance * c_r) * pow(c_r, angular);
    double value = 0.0;
    for (int raw_index = 0; raw_index < radial_count; ++raw_index) {
        const double gto_constant = gto_constants[angular * radial_count + raw_index];
        const double z = c_r * c_r / (density_constant + gto_constant);
        const double a = 0.5 * (raw_index + angular + 3.0);
        const double b = angular + 1.5;
        double raw;
        if (z > 30.0) {
            const double logarithm = log(global_factor) - distance * c_r
                + static_cast<double>(angular) * log(c_r)
                - a * log(density_constant + gto_constant)
                + z + (a - b) * log(z);
            raw = exp(logarithm) * positive_hypergeometric(a, b, z);
        } else {
            raw = gamma_a[angular * radial_count + raw_index]
                / gamma_b[angular] * positive_hypergeometric(a, b, z)
                * pow(density_constant + gto_constant, -a) * factor;
        }
        value += raw * orthonormalization[
            (angular * radial_count + raw_index) * radial_count + target_radial];
    }
    return value;
}

__device__ double cutoff_value(double distance, double cutoff) {
    if (distance >= cutoff) {
        return 0.0;
    }
    const double width = fmin(0.5, cutoff);
    if (distance <= cutoff - width) {
        return 1.0;
    }
    return 0.5 * (1.0 + cos(
        3.141592653589793238462643383279502884
        * (distance - cutoff + width) / width));
}

template <int MaxAngular>
__global__ void compute_edge_basis(
    const std::int32_t* numbers,
    const std::int32_t* graph_atoms,
    const double* graph_displacements,
    const double* graph_distance2,
    const std::int32_t* species,
    int species_count,
    double cutoff,
    double density_width,
    int radial_count,
    std::size_t total,
    double* edge_basis,
    std::int32_t* edge_species,
    const double* gto_constants,
    const double* gamma_a,
    const double* gamma_b,
    const double* orthonormalization) {
    const std::size_t linear = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (linear >= total) {
        return;
    }
    constexpr int max_angular = MaxAngular;
    const int angular_count = max_angular + 1;
    const int radial_stride = angular_count * radial_count;
    const int harmonic_stride = angular_count * angular_count;
    double* radial_output = edge_basis + linear * (radial_stride + harmonic_stride);
    double* harmonic_output = radial_output + radial_stride;
    edge_species[linear] = species_index(
        numbers[graph_atoms[linear]], species, species_count);

    const double* displacement = graph_displacements + linear * 3;
    const double distance = sqrt(fmax(0.0, graph_distance2[linear]));
    const double scaling = cutoff_value(distance, cutoff);
    if (scaling == 0.0) {
        for (int index = 0; index < radial_stride + harmonic_stride; ++index) {
            radial_output[index] = 0.0;
        }
        return;
    }
    harmonic_values<MaxAngular>(displacement, harmonic_output);
    for (int angular = 0; angular < angular_count; ++angular) {
        for (int radial = 0; radial < radial_count; ++radial) {
            radial_output[angular * radial_count + radial] = scaling * radial_value(
                distance, angular, radial, radial_count, density_width,
                gto_constants, gamma_a, gamma_b, orthonormalization);
        }
    }
}

template <int MaxAngular>
void launch_edge_basis(
    int requested_angular,
    cudaStream_t stream,
    const std::int32_t* numbers,
    const std::int32_t* graph_atoms,
    const double* graph_displacements,
    const double* graph_distance2,
    const std::int32_t* species,
    int species_count,
    double cutoff,
    double density_width,
    int radial_count,
    std::size_t total,
    double* edge_basis,
    std::int32_t* edge_species,
    const double* gto_constants,
    const double* gamma_a,
    const double* gamma_b,
    const double* orthonormalization) {
    if (requested_angular == MaxAngular) {
        constexpr unsigned int block_size = 128;
        const auto blocks = static_cast<unsigned int>(
            (total + block_size - 1) / block_size);
        compute_edge_basis<MaxAngular><<<blocks, block_size, 0, stream>>>(
            numbers, graph_atoms, graph_displacements, graph_distance2, species,
            species_count, cutoff, density_width, radial_count, total, edge_basis,
            edge_species, gto_constants, gamma_a, gamma_b, orthonormalization);
        return;
    }
    if constexpr (MaxAngular < 31) {
        launch_edge_basis<MaxAngular + 1>(
            requested_angular, stream, numbers, graph_atoms, graph_displacements,
            graph_distance2, species, species_count, cutoff, density_width, radial_count,
            total, edge_basis, edge_species, gto_constants, gamma_a, gamma_b,
            orthonormalization);
    } else {
        throw std::invalid_argument("unsupported CUDA angular order");
    }
}

__global__ void compute_coefficients(
    const std::int64_t* graph_offsets,
    const std::int32_t* edge_species,
    int radial_count,
    int max_angular,
    int coefficient_size,
    int edge_basis_stride,
    std::size_t total,
    double* coefficients,
    const double* edge_basis) {
    const std::size_t linear = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (linear >= total) {
        return;
    }
    const std::int64_t center = static_cast<std::int64_t>(linear / coefficient_size);
    const std::size_t local = linear % coefficient_size;
    const int angular_count = max_angular + 1;
    const int radial_stride = angular_count * radial_count;
    const int angular_block = angular_count * angular_count;
    const int per_species = radial_count * angular_block;
    const int coefficient_species = static_cast<int>(local / per_species);
    const int species_local = static_cast<int>(local % per_species);
    const int radial = species_local / angular_block;
    const int angular_local = species_local % angular_block;
    int angular = 0;
    while (angular + 1 < angular_count
        && (angular + 1) * (angular + 1) <= angular_local) {
        ++angular;
    }
    const int m = angular_local - angular * angular - angular;
    double result = 0.0;
    if (m <= angular && m >= -angular) {
        const std::int64_t begin = graph_offsets[center];
        const std::int64_t end = graph_offsets[center + 1];
        const int harmonic = angular * angular + angular + m;
        for (std::int64_t index = begin; index < end; ++index) {
            if (edge_species[index] != coefficient_species) {
                continue;
            }
            const double* edge = edge_basis + index * edge_basis_stride;
            result += edge[angular * radial_count + radial]
                * edge[radial_stride + harmonic];
        }
    }
    coefficients[linear] = result;
}

__device__ double coefficient_at(
    const double* coefficients,
    std::int64_t center,
    int species,
    int radial,
    int angular,
    int m,
    int radial_count,
    int max_angular,
    int coefficient_size) {
    return coefficients[
        static_cast<std::size_t>(center) * coefficient_size
        + mdescriptor::detail::local_coefficient_index(
            static_cast<std::size_t>(species), radial, angular, m,
            radial_count, max_angular)];
}

__global__ void assemble_features(
    const std::int32_t* numbers,
    const std::int32_t* species,
    int species_count,
    int radial_count,
    int max_angular,
    int coefficient_max_angular,
    int kind,
    int coefficient_size,
    int features,
    int active_features,
    std::size_t total,
    const double* coefficients,
    double* output) {
    const std::size_t linear = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (linear >= total) {
        return;
    }
    const std::int64_t center = static_cast<std::int64_t>(linear / active_features);
    const int feature = static_cast<int>(linear % active_features);
    const int center_species = species_index(numbers[center], species, species_count);
    if (center_species < 0) {
        return;
    }
    const std::size_t output_index = static_cast<std::size_t>(center) * features
        + static_cast<std::size_t>(center_species) * active_features + feature;
    const int angular_count = max_angular + 1;
    if (kind == 2) {
        const int local = feature;
        const int neighbor_species = local / radial_count;
        const int radial = local % radial_count;
        output[output_index] = coefficient_at(
            coefficients, center, neighbor_species, radial, 0, 0,
            radial_count, coefficient_max_angular, coefficient_size);
        return;
    }
    if (kind == 3) {
        const int pair_stride = angular_count * radial_count * radial_count;
        const int local = feature;
        int pair = local / pair_stride;
        int remainder = local % pair_stride;
        int first = 0;
        int second = 0;
        mdescriptor::detail::local_decode_species_pair(
            pair, species_count, first, second);
        const int angular = remainder / (radial_count * radial_count);
        remainder %= radial_count * radial_count;
        const int first_radial = remainder / radial_count;
        const int second_radial = remainder % radial_count;
        double value = 0.0;
        for (int m = -angular; m <= angular; ++m) {
            value += coefficient_at(
                coefficients, center, first, first_radial, angular, m,
                radial_count, coefficient_max_angular, coefficient_size)
                * coefficient_at(
                    coefficients, center, second, second_radial, angular, m,
                    radial_count, coefficient_max_angular, coefficient_size);
        }
        const double pair_scale = first == second ? 1.0 : 1.414213562373095048801688724209698079;
        const double angular_scale = (angular % 2 == 0 ? 1.0 : -1.0)
            / sqrt(2.0 * angular + 1.0);
        output[output_index] = pair_scale * angular_scale * value;
        return;
    }

    const int local = feature;
    const int neighbor_species = local / (angular_count * angular_count * radial_count);
    int remainder = local % (angular_count * angular_count * radial_count);
    int found_angular = 0;
    int found_m = 0;
    int found_radial = 0;
    mdescriptor::detail::local_decode_spherical_feature(
        remainder, radial_count, max_angular,
        found_radial, found_angular, found_m);
    output[output_index] = coefficient_at(
        coefficients, center, neighbor_species, found_radial, found_angular, found_m,
        radial_count, coefficient_max_angular, coefficient_size);
}

} // namespace

std::vector<double> compute_local_descriptors(
    CudaExecutionContext& context,
    const DeviceBatch& batch,
    const DeviceNeighborGraph& graph,
    const std::vector<std::int32_t>& species,
    double cutoff,
    double density_width,
    int max_radial,
    int max_angular,
    std::int32_t kind) {
    if (max_angular < 0 || max_angular > 31) {
        throw std::invalid_argument("CUDA local descriptors support max_angular up to 31");
    }
    if (species.empty()) {
        throw std::invalid_argument("species must not be empty");
    }
    if (max_radial < 0) {
        throw std::invalid_argument("max_radial must be non-negative");
    }
    if (kind != 0 && kind != 2 && kind != 3) {
        throw std::invalid_argument("unsupported CUDA local descriptor kind");
    }

    LocalDescriptorOptions options;
    options.species = species;
    options.cutoff = cutoff;
    options.density_width = density_width;
    options.max_radial = max_radial;
    options.max_angular = max_angular;
    const auto descriptor_kind = static_cast<LocalDescriptorKind>(kind);
    const auto features = detail::local_layout_feature_count(options, descriptor_kind);
    const int radial_count = max_radial + 1;
    // SoapRadialSpectrum only consumes the l=0 coefficient block.  Avoid
    // constructing the unused higher-angular channels on the device.
    const int coefficient_max_angular = detail::local_coefficient_max_angular(
        options, descriptor_kind);
    const int coefficient_angular_count = coefficient_max_angular + 1;
    const int coefficient_size = static_cast<int>(detail::local_coefficient_count(
        options, descriptor_kind));
    const std::size_t output_size = static_cast<std::size_t>(batch.atoms())
        * static_cast<std::size_t>(features);
    const std::size_t coefficient_total = static_cast<std::size_t>(batch.atoms())
        * static_cast<std::size_t>(coefficient_size);
    const int active_features = static_cast<int>(
        features / static_cast<std::int64_t>(species.size()));
    const std::size_t active_output_size = static_cast<std::size_t>(batch.atoms())
        * static_cast<std::size_t>(active_features);

    RadialBasisSet radial_bases;
    radial_bases.reset(max_radial, coefficient_max_angular, cutoff);
    std::vector<double> gto_constants;
    std::vector<double> gamma_a;
    std::vector<double> gamma_b;
    std::vector<double> orthonormalization;
    gto_constants.reserve(static_cast<std::size_t>(coefficient_angular_count) * radial_count);
    gamma_a.reserve(static_cast<std::size_t>(coefficient_angular_count) * radial_count);
    gamma_b.reserve(static_cast<std::size_t>(coefficient_angular_count));
    orthonormalization.reserve(
        static_cast<std::size_t>(coefficient_angular_count) * radial_count * radial_count);
    for (int angular = 0; angular < coefficient_angular_count; ++angular) {
        const GtoRadialBasis& basis = radial_bases[static_cast<std::size_t>(angular)];
        gto_constants.insert(gto_constants.end(), basis.gto_constants.begin(), basis.gto_constants.end());
        gamma_a.insert(gamma_a.end(), basis.gamma_a.begin(), basis.gamma_a.end());
        gamma_b.push_back(basis.gamma_b);
        for (const auto& row : basis.orthonormalization) {
            orthonormalization.insert(orthonormalization.end(), row.begin(), row.end());
        }
    }

    const int edge_radial_stride = coefficient_angular_count * radial_count;
    const int edge_harmonic_stride = coefficient_angular_count * coefficient_angular_count;
    const int edge_basis_stride = edge_radial_stride + edge_harmonic_stride;
    const std::size_t edge_count = graph.pairs();

    auto align_up = [](std::size_t value, std::size_t alignment) {
        return (value + alignment - 1) / alignment * alignment;
    };
    std::size_t workspace_size = 0;
    auto reserve_workspace = [&workspace_size, &align_up](
        std::size_t bytes, std::size_t alignment) {
        workspace_size = align_up(workspace_size, alignment);
        const std::size_t offset = workspace_size;
        workspace_size += bytes;
        return offset;
    };
    const std::size_t species_offset = reserve_workspace(
        species.size() * sizeof(std::int32_t), alignof(std::int32_t));
    const std::size_t gto_constants_offset = reserve_workspace(
        gto_constants.size() * sizeof(double), alignof(double));
    const std::size_t gamma_a_offset = reserve_workspace(
        gamma_a.size() * sizeof(double), alignof(double));
    const std::size_t gamma_b_offset = reserve_workspace(
        gamma_b.size() * sizeof(double), alignof(double));
    const std::size_t orthonormalization_offset = reserve_workspace(
        orthonormalization.size() * sizeof(double), alignof(double));
    const std::size_t edge_basis_offset = reserve_workspace(
        edge_count * static_cast<std::size_t>(edge_basis_stride) * sizeof(double), alignof(double));
    const std::size_t edge_species_offset = reserve_workspace(
        edge_count * sizeof(std::int32_t), alignof(std::int32_t));
    const std::size_t coefficient_offset = reserve_workspace(
        coefficient_total * sizeof(double), alignof(double));
    auto* workspace = static_cast<unsigned char*>(context.workspace_buffer(workspace_size));
    auto* device_species = reinterpret_cast<std::int32_t*>(workspace + species_offset);
    auto* device_gto_constants = reinterpret_cast<double*>(workspace + gto_constants_offset);
    auto* device_gamma_a = reinterpret_cast<double*>(workspace + gamma_a_offset);
    auto* device_gamma_b = reinterpret_cast<double*>(workspace + gamma_b_offset);
    auto* device_orthonormalization = reinterpret_cast<double*>(
        workspace + orthonormalization_offset);
    auto* edge_basis = reinterpret_cast<double*>(workspace + edge_basis_offset);
    auto* edge_species = reinterpret_cast<std::int32_t*>(workspace + edge_species_offset);
    auto* coefficient_data = reinterpret_cast<double*>(workspace + coefficient_offset);
    auto upload = [&context](void* destination, const void* source, std::size_t bytes, const char* operation) {
        if (bytes == 0) {
            return;
        }
        check_cuda(
            cudaMemcpyAsync(
                destination, source, bytes, cudaMemcpyHostToDevice, context.stream()),
            operation);
    };
    upload(
        device_species, species.data(), species.size() * sizeof(std::int32_t),
        "could not upload CUDA species");
    upload(
        device_gto_constants, gto_constants.data(), gto_constants.size() * sizeof(double),
        "could not upload CUDA radial constants");
    upload(
        device_gamma_a, gamma_a.data(), gamma_a.size() * sizeof(double),
        "could not upload CUDA radial gamma values");
    upload(
        device_gamma_b, gamma_b.data(), gamma_b.size() * sizeof(double),
        "could not upload CUDA radial gamma denominators");
    upload(
        device_orthonormalization, orthonormalization.data(),
        orthonormalization.size() * sizeof(double),
        "could not upload CUDA radial orthonormalization");

    double* output = context.output_buffer(output_size);
    if (coefficient_total > 0) {
        constexpr unsigned int block_size = 128;
        if (edge_count > 0) {
            launch_edge_basis<0>(
                coefficient_max_angular, context.stream(), batch.numbers(), graph.atoms(),
                graph.displacements(), graph.distance2(), device_species,
                static_cast<int>(species.size()), cutoff, density_width, radial_count,
                edge_count, edge_basis, edge_species, device_gto_constants, device_gamma_a,
                device_gamma_b, device_orthonormalization);
            check_cuda(cudaGetLastError(), "CUDA edge basis kernel launch failed");
        }
        const auto coefficient_blocks = static_cast<unsigned int>(
            (coefficient_total + block_size - 1) / block_size);
        compute_coefficients<<<coefficient_blocks, block_size, 0, context.stream()>>>(
            graph.offsets(), edge_species, radial_count, coefficient_max_angular,
            coefficient_size, edge_basis_stride, coefficient_total, coefficient_data, edge_basis);
        check_cuda(cudaGetLastError(), "CUDA spherical coefficient kernel launch failed");
        check_cuda(
            cudaMemsetAsync(
                output, 0, output_size * sizeof(double), context.stream()),
            "could not clear CUDA descriptor output");
        if (active_output_size > 0) {
            const auto output_blocks = static_cast<unsigned int>(
                (active_output_size + block_size - 1) / block_size);
            assemble_features<<<output_blocks, block_size, 0, context.stream()>>>(
                batch.numbers(), device_species, static_cast<int>(species.size()),
                radial_count, max_angular, coefficient_max_angular, kind, coefficient_size,
                static_cast<int>(features), active_features, active_output_size,
                coefficient_data, output);
            check_cuda(cudaGetLastError(), "CUDA local descriptor kernel launch failed");
        }
        return context.download_output(output_size);
    }
    return {};
}

} // namespace mdescriptor::cuda
