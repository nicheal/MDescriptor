#pragma once
// Private CUDA support for the basic/local pair descriptor family.
#include "extended_descriptors_common.cuh"

namespace {

template <typename T>
std::vector<T> download(
    const T* source,
    std::size_t count,
    CudaExecutionContext& context,
    const char* operation) {
    std::vector<T> result(count);
    if (count != 0) {
        // CUDA work is asynchronous.  Let another Python thread cancel the
        // public ComputeControl while this stream waits for the device.
        py::gil_scoped_release release;
        check_cuda(
            cudaMemcpyAsync(
                result.data(), source, count * sizeof(T), cudaMemcpyDeviceToHost,
                context.stream()),
            operation);
        context.synchronize();
    }
    return result;
}

__device__ I64 center_for_edge(const I64* offsets, I64 atoms, I64 edge) {
    I64 left = 0;
    I64 right = atoms;
    while (left + 1 < right) {
        const I64 middle = left + (right - left) / 2;
        if (offsets[middle] <= edge) left = middle;
        else right = middle;
    }
    return left;
}

__global__ void atomic_composition_kernel(
    const I32* numbers,
    const I64* offsets,
    I64 structures,
    I64 atoms,
    const I32* species,
    int species_count,
    bool per_system,
    double* output) {
    const I64 row = static_cast<I64>(blockIdx.x) * blockDim.x + threadIdx.x;
    const I64 rows = per_system ? structures : atoms;
    if (row >= rows) return;
    if (per_system) {
        for (I64 atom = offsets[row]; atom < offsets[row + 1]; ++atom) {
            const int type = species_index(numbers[atom], species, species_count);
            if (type >= 0) output[row * species_count + type] += 1.0;
        }
        return;
    }
    const int type = species_index(numbers[row], species, species_count);
    if (type >= 0) output[row * species_count + type] = 1.0;
}

__global__ void sorted_distances_kernel(
    const I32* numbers,
    const I64* graph_offsets,
    const I32* graph_atoms,
    const double* graph_distance2,
    const I32* species,
    int species_count,
    int max_neighbors,
    bool separate,
    double cutoff,
    I64 atoms,
    double* output) {
    const I64 center = static_cast<I64>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (center >= atoms) return;
    const I64 begin = graph_offsets[center];
    const I64 end = graph_offsets[center + 1];
    if (separate) {
        for (int wanted = 0; wanted < species_count; ++wanted) {
            int count = 0;
            const I64 base = center * static_cast<I64>(species_count * max_neighbors)
                + static_cast<I64>(wanted * max_neighbors);
            for (I64 edge = begin; edge < end && count < max_neighbors; ++edge) {
                if (species_index(numbers[graph_atoms[edge]], species, species_count) != wanted) {
                    continue;
                }
                output[base + count++] = sqrt(fmax(0.0, graph_distance2[edge]));
            }
            if (count > 0) {
                for (int index = count; index < max_neighbors; ++index) {
                    output[base + index] = cutoff;
                }
            }
        }
        return;
    }
    const I64 base = center * max_neighbors;
    int count = 0;
    for (I64 edge = begin; edge < end && count < max_neighbors; ++edge) {
        output[base + count++] = sqrt(fmax(0.0, graph_distance2[edge]));
    }
    if (count > 0) {
        for (int index = count; index < max_neighbors; ++index) output[base + index] = cutoff;
    }
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
    const double global_factor = pow(kPi / density_width2, 0.75);
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
                - a * log(density_constant + gto_constant) + z + (a - b) * log(z);
            raw = exp(logarithm) * positive_hypergeometric(a, b, z);
        } else {
            raw = gamma_a[angular * radial_count + raw_index] / gamma_b[angular]
                * positive_hypergeometric(a, b, z)
                * pow(density_constant + gto_constant, -a) * factor;
        }
        value += raw * orthonormalization[
            (angular * radial_count + raw_index) * radial_count + target_radial];
    }
    return value;
}

__device__ double smooth_cutoff(double distance, double cutoff) {
    if (distance >= cutoff) return 0.0;
    const double width = fmin(0.5, cutoff);
    if (distance <= cutoff - width) return 1.0;
    return 0.5 * (1.0 + cos(kPi * (distance - cutoff + width) / width));
}

template <int MaxAngular>
__global__ void spherical_pair_kernel(
    const I64* graph_offsets,
    const I32* graph_atoms,
    const I32* graph_shifts,
    const double* graph_displacements,
    const double* graph_distance2,
    I64 atoms,
    double cutoff,
    double density_width,
    int radial_count,
    int max_angular,
    const double* gto_constants,
    const double* gamma_a,
    const double* gamma_b,
    const double* orthonormalization,
    double* records,
    double* output) {
    const I64 edge = static_cast<I64>(blockIdx.x) * blockDim.x + threadIdx.x;
    const I64 total = graph_offsets[atoms];
    if (edge >= total) return;
    const I64 center = center_for_edge(graph_offsets, atoms, edge);
    const I32 atom = graph_atoms[edge];
    records[edge * 5 + 0] = static_cast<double>(center);
    records[edge * 5 + 1] = static_cast<double>(atom);
    records[edge * 5 + 2] = static_cast<double>(graph_shifts[edge * 3 + 0]);
    records[edge * 5 + 3] = static_cast<double>(graph_shifts[edge * 3 + 1]);
    records[edge * 5 + 4] = static_cast<double>(graph_shifts[edge * 3 + 2]);
    const double distance = sqrt(fmax(0.0, graph_distance2[edge]));
    const double scale = smooth_cutoff(distance, cutoff);
    const I64 feature_count = static_cast<I64>((max_angular + 1) * (max_angular + 1) * radial_count);
    double harmonics[(MaxAngular + 1) * (MaxAngular + 1)]{};
    if (scale != 0.0) harmonic_values<MaxAngular>(graph_displacements + edge * 3, harmonics, max_angular);
    double* row = output + edge * feature_count;
    if (scale == 0.0) {
        for (I64 feature = 0; feature < feature_count; ++feature) row[feature] = 0.0;
        return;
    }
    for (int angular = 0; angular <= max_angular; ++angular) {
        for (int radial = 0; radial < radial_count; ++radial) {
            const double radial_component = radial_value(
                distance, angular, radial, radial_count, density_width,
                gto_constants, gamma_a, gamma_b, orthonormalization);
            for (int m = -angular; m <= angular; ++m) {
                const I64 harmonic_index = static_cast<I64>(angular * angular + angular + m);
                row[harmonic_index * radial_count + radial] = scale
                    * harmonics[harmonic_index] * radial_component;
            }
        }
    }
}

} // namespace
