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

template <typename Value>
class DeviceBuffer {
public:
    DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    ~DeviceBuffer() noexcept {
        if (data_ != nullptr) {
            (void)cudaFree(data_);
        }
    }

    void upload(
        const std::vector<Value>& values,
        cudaStream_t stream,
        const char* operation) {
        if (values.empty()) {
            return;
        }
        check_cuda(
            cudaMalloc(reinterpret_cast<void**>(&data_), values.size() * sizeof(Value)),
            operation);
        try {
            check_cuda(
                cudaMemcpyAsync(
                    data_, values.data(), values.size() * sizeof(Value),
                    cudaMemcpyHostToDevice, stream),
                operation);
        } catch (...) {
            (void)cudaFree(data_);
            data_ = nullptr;
            throw;
        }
    }

    Value* data() const noexcept { return data_; }

private:
    Value* data_ = nullptr;
};

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

__device__ void harmonic_values(
    const double* vector,
    int max_angular,
    double* output) {
    // Compute the complete real-harmonic block once per graph edge.  The old
    // coefficient kernel recomputed the Legendre recurrence independently for
    // every (l, m) coefficient of that edge.
    double legendre[528]{};
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

__device__ std::size_t coefficient_index(
    int species,
    int radial,
    int angular,
    int m,
    int radial_count,
    int max_angular) {
    return ((static_cast<std::size_t>(species) * radial_count + radial)
        * (max_angular + 1) + angular)
        * (2 * max_angular + 1) + max_angular + m;
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
    int max_angular,
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
    harmonic_values(displacement, max_angular, harmonic_output);
    for (int angular = 0; angular < angular_count; ++angular) {
        for (int radial = 0; radial < radial_count; ++radial) {
            radial_output[angular * radial_count + radial] = scaling * radial_value(
                distance, angular, radial, radial_count, density_width,
                gto_constants, gamma_a, gamma_b, orthonormalization);
        }
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
    const int width = 2 * max_angular + 1;
    const int angular_count = max_angular + 1;
    const int radial_stride = angular_count * radial_count;
    const int harmonic_stride = angular_count * angular_count;
    const int per_species = radial_count * angular_count * width;
    const int coefficient_species = static_cast<int>(local / per_species);
    const int species_local = static_cast<int>(local % per_species);
    const int radial = species_local / (angular_count * width);
    const int angular_local = species_local % (angular_count * width);
    const int angular = angular_local / width;
    const int m = angular_local % width - max_angular;
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
        + coefficient_index(species, radial, angular, m, radial_count, max_angular)];
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
    std::size_t total,
    const double* coefficients,
    double* output) {
    const std::size_t linear = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (linear >= total) {
        return;
    }
    const std::int64_t center = static_cast<std::int64_t>(linear / features);
    const int feature = static_cast<int>(linear % features);
    const int center_species = species_index(numbers[center], species, species_count);
    if (center_species < 0) {
        output[linear] = 0.0;
        return;
    }
    const int angular_count = max_angular + 1;
    if (kind == 2) {
        const int per_center = species_count * radial_count;
        const int center_offset = center_species * per_center;
        if (feature < center_offset || feature >= center_offset + per_center) {
            output[linear] = 0.0;
            return;
        }
        const int local = feature - center_offset;
        const int neighbor_species = local / radial_count;
        const int radial = local % radial_count;
        output[linear] = coefficient_at(
            coefficients, center, neighbor_species, radial, 0, 0,
            radial_count, coefficient_max_angular, coefficient_size);
        return;
    }
    if (kind == 3) {
        const int pair_count = species_count * (species_count + 1) / 2;
        const int pair_stride = angular_count * radial_count * radial_count;
        const int per_center = pair_count * pair_stride;
        const int center_offset = center_species * per_center;
        if (feature < center_offset || feature >= center_offset + per_center) {
            output[linear] = 0.0;
            return;
        }
        const int local = feature - center_offset;
        int pair = local / pair_stride;
        int remainder = local % pair_stride;
        int first = 0;
        int second = 0;
        for (int candidate = 0; candidate < species_count; ++candidate) {
            const int count = species_count - candidate;
            if (pair < count) {
                first = candidate;
                second = candidate + pair;
                break;
            }
            pair -= count;
        }
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
        output[linear] = pair_scale * angular_scale * value;
        return;
    }

    const int center_block = species_count * radial_count * angular_count * angular_count;
    const int center_offset = center_species * center_block;
    if (feature < center_offset || feature >= center_offset + center_block) {
        output[linear] = 0.0;
        return;
    }
    const int local = feature - center_offset;
    const int neighbor_species = local / (angular_count * angular_count * radial_count);
    int remainder = local % (angular_count * angular_count * radial_count);
    // The spherical expansion layout is (species, l, m, n), while the
    // coefficient storage reserves the maximum m width for every l.  Decode
    // the compact l/m/n ordering without using a second layout convention.
    int found_angular = 0;
    int found_m = 0;
    int found_radial = 0;
    for (int candidate_l = 0; candidate_l <= max_angular; ++candidate_l) {
        const int block = (2 * candidate_l + 1) * radial_count;
        if (remainder < block) {
            found_angular = candidate_l;
            found_m = remainder / radial_count - candidate_l;
            found_radial = remainder % radial_count;
            break;
        }
        remainder -= block;
    }
    output[linear] = coefficient_at(
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
    const int angular_count = max_angular + 1;
    const int coefficient_width = 2 * max_angular + 1;
    const int coefficient_size = static_cast<int>(species.size()) * radial_count
        * angular_count * coefficient_width;
    const std::size_t output_size = static_cast<std::size_t>(batch.atoms())
        * static_cast<std::size_t>(features);
    const std::size_t coefficient_total = static_cast<std::size_t>(batch.atoms())
        * static_cast<std::size_t>(coefficient_size);

    std::vector<double> gto_constants;
    std::vector<double> gamma_a;
    std::vector<double> gamma_b;
    std::vector<double> orthonormalization;
    gto_constants.reserve(static_cast<std::size_t>(angular_count) * radial_count);
    gamma_a.reserve(static_cast<std::size_t>(angular_count) * radial_count);
    gamma_b.reserve(static_cast<std::size_t>(angular_count));
    orthonormalization.reserve(
        static_cast<std::size_t>(angular_count) * radial_count * radial_count);
    for (int angular = 0; angular < angular_count; ++angular) {
        const GtoRadialBasis basis(radial_count, cutoff, angular);
        gto_constants.insert(gto_constants.end(), basis.gto_constants.begin(), basis.gto_constants.end());
        gamma_a.insert(gamma_a.end(), basis.gamma_a.begin(), basis.gamma_a.end());
        gamma_b.push_back(basis.gamma_b);
        for (const auto& row : basis.orthonormalization) {
            orthonormalization.insert(orthonormalization.end(), row.begin(), row.end());
        }
    }

    DeviceBuffer<std::int32_t> device_species;
    DeviceBuffer<double> device_gto_constants;
    DeviceBuffer<double> device_gamma_a;
    DeviceBuffer<double> device_gamma_b;
    DeviceBuffer<double> device_orthonormalization;
    device_species.upload(species, context.stream(), "could not upload CUDA species");
    device_gto_constants.upload(gto_constants, context.stream(), "could not upload CUDA radial constants");
    device_gamma_a.upload(gamma_a, context.stream(), "could not upload CUDA radial gamma values");
    device_gamma_b.upload(gamma_b, context.stream(), "could not upload CUDA radial gamma denominators");
    device_orthonormalization.upload(
        orthonormalization, context.stream(), "could not upload CUDA radial orthonormalization");
    context.synchronize();

    double* output = context.output_buffer(output_size);
    if (coefficient_total > 0) {
        // Allocate the coefficient workspace directly because it is written
        // by the first kernel and consumed by the second one.
        double* coefficient_data = nullptr;
        check_cuda(
            cudaMalloc(
                reinterpret_cast<void**>(&coefficient_data),
                coefficient_total * sizeof(double)),
            "could not allocate CUDA spherical coefficients");
        try {
            constexpr unsigned int block_size = 128;
            const auto coefficient_blocks = static_cast<unsigned int>(
                (coefficient_total + block_size - 1) / block_size);
            compute_coefficients<<<coefficient_blocks, block_size, 0, context.stream()>>>(
                batch.numbers(), graph.offsets(), graph.atoms(), graph.displacements(),
                graph.distance2(), device_species.data(), static_cast<int>(species.size()),
                cutoff, density_width, radial_count, max_angular, coefficient_size,
                coefficient_total, coefficient_data, device_gto_constants.data(), device_gamma_a.data(),
                device_gamma_b.data(), device_orthonormalization.data());
            check_cuda(cudaGetLastError(), "CUDA spherical coefficient kernel launch failed");
            if (output_size > 0) {
                const auto output_blocks = static_cast<unsigned int>(
                    (output_size + block_size - 1) / block_size);
                assemble_features<<<output_blocks, block_size, 0, context.stream()>>>(
                    batch.numbers(), device_species.data(), static_cast<int>(species.size()),
                    radial_count, max_angular, kind, coefficient_size,
                    static_cast<int>(features), output_size, coefficient_data, output);
                check_cuda(cudaGetLastError(), "CUDA local descriptor kernel launch failed");
            }
            const auto result = context.download_output(output_size);
            double* allocated_coefficients = coefficient_data;
            coefficient_data = nullptr;
            check_cuda(cudaFree(allocated_coefficients), "could not release CUDA spherical coefficients");
            return result;
        } catch (...) {
            (void)cudaFree(coefficient_data);
            throw;
        }
    }
    return {};
}

} // namespace mdescriptor::cuda
