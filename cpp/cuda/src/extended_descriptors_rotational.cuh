#pragma once
// Private CUDA support for the rotational descriptor family.
#include "mdescriptor/detail/rotational_bispectrum.hpp"
#include "rotational_math.hpp"
#include "extended_descriptors_common.cuh"

namespace {

constexpr int kRotationalUCapacity = static_cast<int>(
    mdescriptor::detail::rotational::u_total_size(10));
using mdescriptor::detail::rotational::complex_add;
using mdescriptor::detail::rotational::complex_multiply;

inline std::vector<double> so3_basis_host(
    int nmax,
    int lmax,
    double cutoff,
    double alpha,
    int* quadrature_count) {
    if (quadrature_count != nullptr) {
        *quadrature_count = (nmax + lmax + 1) * 10;
    }
    return mdescriptor::detail::so3_radial_basis(nmax, lmax, cutoff, alpha);
}

struct RotationalCudaOptions {
    int kind = 0;
    int nmax = 3;
    int lmax = 3;
    int twojmax = 3;
    int diagonal = 3;
    double cutoff = 3.5;
    double alpha = 2.0;
    double rfac0 = 1.0;
    double rmin0 = 0.0;
    double rcutfac = 1.0;
    bool weight_on = false;
    bool normalize_u = false;
};

RotationalCudaOptions rotational_options(
    const std::string& name, const py::dict& options) {
    RotationalCudaOptions result;
    if (name == "SO3") {
        result.kind = 0;
        result.nmax = option(options, "nmax", 3);
    } else if (name == "SO4") {
        result.kind = 1;
        result.nmax = 1;
        result.rfac0 = option(options, "rfac0", 1.0);
    } else if (name == "SNAP") {
        result.kind = 2;
        result.nmax = 1;
        result.rfac0 = option(options, "rfac0", 0.99363);
    } else if (name == "LBispectrum") {
        result.kind = 3;
        result.nmax = 1;
        result.rfac0 = option(options, "rfac0", 0.99363);
    } else {
        throw std::invalid_argument("unknown CUDA rotational descriptor: " + name);
    }
    result.lmax = option(options, "lmax", 3);
    result.twojmax = option(options, "twojmax", 3);
    result.diagonal = option(options, "diagonal", 3);
    result.cutoff = option(options, "rcut", 3.5);
    result.alpha = option(options, "alpha", 2.0);
    result.rmin0 = option(options, "rmin0", 0.0);
    result.rcutfac = option(options, "rcutfac", 1.0);
    result.weight_on = option(options, "weight_on", false);
    result.normalize_u = option(options, "normalize_U", false);
    return result;
}

__device__ double smooth_radial_device(double distance, double cutoff) {
    if (distance >= cutoff) return 0.0;
    return 0.5 * (1.0 + cos(kPi * distance / cutoff));
}

__device__ double legendre_device(int degree, double x) {
    if (degree == 0) return 1.0;
    if (degree == 1) return x;
    double previous = 1.0;
    double current = x;
    for (int l = 2; l <= degree; ++l) {
        const double next = ((2.0 * l - 1.0) * x * current - (l - 1.0) * previous) / l;
        previous = current;
        current = next;
    }
    return current;
}

__device__ void modified_spherical_bessel_device(
    double x,
    int max_angular,
    double* result) {
    const double absolute = fabs(x);
    if (absolute < 1.0) {
        const double square = absolute * absolute;
        for (int angular = 0; angular <= max_angular; ++angular) {
            double term = 0.0;
            if (angular == 0) {
                term = 1.0;
            } else if (absolute > 0.0) {
                term = exp(
                    angular * log(absolute) + 0.5 * log(kPi)
                    - (angular + 1.0) * log(2.0)
                    - lgamma(angular + 1.5));
            }
            double sum = term;
            for (int index = 0; index < 80; ++index) {
                term *= square / (4.0 * (index + 1.0) * (index + angular + 1.5));
                sum += term;
                if (fabs(term) <= fabs(sum) * 1e-16) break;
            }
            result[angular] = sum;
        }
        return;
    }
    result[0] = sinh(absolute) / absolute;
    if (max_angular == 0) return;
    result[1] = (absolute * cosh(absolute) - sinh(absolute))
        / (absolute * absolute);
    for (int angular = 1; angular < max_angular; ++angular) {
                result[angular + 1] = result[angular - 1]
            - (2.0 * angular + 1.0) / absolute * result[angular];
    }
}


__global__ void so3_kernel(
    const I32* numbers,
    const I64* graph_offsets,
    const I32* graph_atoms,
    const double* graph_displacements,
    const double* graph_distance2,
    int nmax,
    int lmax,
    double cutoff,
    double alpha,
    bool weight_on,
    int quadrature_count,
    const double* basis,
    I64 features,
    I64 atoms,
    double* output) {
    const I64 center = static_cast<I64>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (center >= atoms) return;
    DeviceComplex coefficients[8 * 9 * 17]{};
    const I64 begin = graph_offsets[center];
    const I64 end = graph_offsets[center + 1];
    for (I64 edge = begin; edge < end; ++edge) {
        const double radius = sqrt(fmax(0.0, graph_distance2[edge]));
        if (radius <= 0.0 || radius >= cutoff) continue;
        const I32 atom = graph_atoms[edge];
        DeviceComplex harmonics[81]{};
        complex_spherical_harmonics_device<9>(
            graph_displacements + edge * 3, lmax, harmonics);
        const double cutoff_value = 0.5 * (cos(kPi * radius / cutoff) + 1.0);
        const double sign = weight_on && numbers[atom] != numbers[center] ? -1.0 : 1.0;
        const double atom_weight = sign * static_cast<double>(numbers[atom])
            * 4.0 * kPi * exp(-alpha * radius * radius) * cutoff_value;
        for (int radial = 0; radial < nmax; ++radial) {
            for (int angular = 0; angular <= lmax; ++angular) {
                double radial_value = 0.0;
                for (int q_index = 0; q_index < quadrature_count; ++q_index) {
                    const double x = cos(
                        (2.0 * (q_index + 1) - 1.0) * kPi
                        / (2.0 * quadrature_count));
                    const double q = cutoff * 0.5 * (x + 1.0);
                    double bessel[9]{};
                    modified_spherical_bessel_device(
                        2.0 * alpha * radius * q, lmax, bessel);
                    radial_value += basis[radial * quadrature_count + q_index]
                        * bessel[angular];
                }
                const double angular_normalization = sqrt(
                    2.0 * sqrt(2.0) * kPi / sqrt(2.0 * angular + 1.0));
                const I64 base = static_cast<I64>(
                    radial * (lmax + 1) + angular) * (2 * lmax + 1);
                for (int m = -angular; m <= angular; ++m) {
                    coefficients[base + lmax + m] = complex_add(
                        coefficients[base + lmax + m], complex_scale(
                            harmonics[angular * angular + angular + m],
                            atom_weight * radial_value * angular_normalization));
                }
            }
        }
    }
    double* target = output + center * features;
    I64 offset = 0;
    for (int first = 0; first < nmax; ++first) {
        for (int second = 0; second <= first; ++second) {
            for (int angular = 0; angular <= lmax; ++angular) {
                double value = 0.0;
                for (int m = -angular; m <= angular; ++m) {
                    const DeviceComplex left = coefficients[
                        (second * (lmax + 1) + angular) * (2 * lmax + 1)
                            + lmax + m];
                    const DeviceComplex right = coefficients[
                        (first * (lmax + 1) + angular) * (2 * lmax + 1)
                            + lmax + m];
                    value += left.real * right.real + left.imag * right.imag;
                }
                if (offset < features) target[offset] = value;
                ++offset;
            }
        }
    }
}

__device__ int rotational_u_offset(int angular) {
    return static_cast<int>(
        mdescriptor::detail::rotational::u_block_offset(angular));
}

__device__ int rotational_u_size(int order) {
    return static_cast<int>(mdescriptor::detail::rotational::u_total_size(order));
}

__device__ void hyperspherical_u_device(
    const double* vector,
    int order,
    double cutoff,
    double rmin0,
    double rfac0,
    DeviceComplex* output) {
    mdescriptor::detail::rotational::hyperspherical_u(
        vector[0], vector[1], vector[2], order, cutoff, rfac0, rmin0, output);
}

__device__ double bispectrum_component_device(
    const DeviceComplex* total,
    int component,
    const I64* z_inner_offsets,
    const I64* inner_term_offsets,
    const double* inner_outer_coefficients,
    const I64* term_first_indices,
    const I64* term_second_indices,
    const double* term_coefficients,
    const I64* projection_offsets,
    const I64* projection_u_indices,
    const I64* projection_z_indices,
    const double* projection_scales) {
    DeviceComplex bispectrum{};
    for (I64 projection = projection_offsets[component];
         projection < projection_offsets[component + 1]; ++projection) {
        DeviceComplex z{};
        const I64 z_index = projection_z_indices[projection];
        for (I64 inner = z_inner_offsets[z_index];
             inner < z_inner_offsets[z_index + 1]; ++inner) {
            DeviceComplex value{};
            for (I64 term = inner_term_offsets[inner];
                 term < inner_term_offsets[inner + 1]; ++term) {
                value = complex_add(value, complex_multiply(
                    complex_scale(
                        total[term_first_indices[term]], term_coefficients[term]),
                    total[term_second_indices[term]]));
            }
            z = complex_add(z, complex_scale(
                value, inner_outer_coefficients[inner]));
        }
        bispectrum = complex_add(bispectrum, complex_scale(
            complex_multiply(
                complex_conjugate(total[projection_u_indices[projection]]), z),
            projection_scales[projection]));
    }
    return 2.0 * bispectrum.real;
}

__global__ void rotational_kernel(
    const I32* numbers,
    const I64* graph_offsets,
    const I32* graph_atoms,
    const double* graph_displacements,
    const double* graph_distance2,
    const double* neighbor_weights,
    const double* neighbor_radii,
    const I64* bispectrum_z_inner_offsets,
    const I64* bispectrum_inner_term_offsets,
    const double* bispectrum_inner_outer_coefficients,
    const I64* bispectrum_term_first_indices,
    const I64* bispectrum_term_second_indices,
    const double* bispectrum_term_coefficients,
    const I64* bispectrum_projection_offsets,
    const I64* bispectrum_projection_u_indices,
    const I64* bispectrum_projection_z_indices,
    const double* bispectrum_projection_scales,
    int kind,
    int nmax,
    int lmax,
    int twojmax,
    double cutoff,
    double rfac0,
    double rmin0,
    double rcutfac,
    bool normalize_u,
    int features,
    I64 atoms,
    double* output) {
    const I64 center = static_cast<I64>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (center >= atoms) return;
    double* target = output + center * static_cast<I64>(features);
    const I64 begin = graph_offsets[center];
    const I64 end = graph_offsets[center + 1];
    int expansion = kind == 3 ? max(0, twojmax) : max(0, 2 * lmax);
    if (kind != 0) {
        // SO4, SNAP and L-Bispectrum use the same hyperspherical U and
        // Clebsch--Gordan contraction as the CPU reference.  Keeping the
        // complete per-center contraction in one CUDA thread avoids any
        // order-dependent atomic reductions.
        if (expansion > 10) return;
        DeviceComplex total[kRotationalUCapacity]{};
        DeviceComplex values[kRotationalUCapacity];
        const double center_weight = kind == 1
            ? static_cast<double>(numbers[center]) : 1.0;
        const int total_size = rotational_u_size(expansion);
        for (int angular = 0; angular <= expansion; ++angular) {
            const int base = rotational_u_offset(angular);
            for (int m = 0; m <= angular; ++m) {
                total[base + m * (angular + 1) + m] = {center_weight, 0.0};
            }
        }
        for (I64 first = begin; first < end; ++first) {
            const double radius = sqrt(fmax(0.0, graph_distance2[first]));
            if (radius <= mdescriptor::detail::rotational::kBispectrumMinimumRadius) continue;
            const I32 first_atom = graph_atoms[first];
            const double neighbor_cutoff = neighbor_radii == nullptr ? cutoff
                : (neighbor_radii[center] + neighbor_radii[first_atom]) * rcutfac;
            if (radius > neighbor_cutoff || neighbor_cutoff <= rmin0) continue;
            const double* vector = graph_displacements + first * 3;
            hyperspherical_u_device(
                vector, expansion, neighbor_cutoff, rmin0, rfac0, values);
            const double cutoff_value =
                mdescriptor::detail::rotational::bispectrum_cutoff(
                    radius, neighbor_cutoff, rmin0);
            const double neighbor_weight = kind == 1
                ? static_cast<double>(numbers[first_atom])
                : (neighbor_weights == nullptr ? 1.0 : neighbor_weights[first_atom]);
            for (int index = 0; index < total_size; ++index) {
                total[index] = complex_add(total[index], complex_scale(
                    values[index], cutoff_value * neighbor_weight));
            }
        }
        if (normalize_u) {
            for (int angular = 0; angular <= expansion; ++angular) {
                const double scale = 4.0 * kPi / sqrt(angular + 1.0);
                const int base = rotational_u_offset(angular);
                for (int mb = 0; mb <= angular; ++mb) {
                    for (int ma = 0; ma <= angular; ++ma) {
                        total[base + mb * (angular + 1) + ma] = complex_scale(
                            total[base + mb * (angular + 1) + ma], scale);
                    }
                }
            }
        }
        for (int feature = 0; feature < features; ++feature) {
            target[feature] = bispectrum_component_device(
                total, feature, bispectrum_z_inner_offsets,
                bispectrum_inner_term_offsets,
                bispectrum_inner_outer_coefficients,
                bispectrum_term_first_indices,
                bispectrum_term_second_indices,
                bispectrum_term_coefficients,
                bispectrum_projection_offsets,
                bispectrum_projection_u_indices,
                bispectrum_projection_z_indices,
                bispectrum_projection_scales);
        }
        return;
    }
    for (int feature = 0; feature < features; ++feature) {
        int wanted_l = 0;
        int n1 = 0;
        int n2 = 0;
        if (kind == 0) {
            int remainder = feature;
            for (n1 = 0; n1 < nmax; ++n1) {
                const int block = (n1 + 1) * (lmax + 1);
                if (remainder < block) break;
                remainder -= block;
            }
            n2 = remainder / (lmax + 1);
            wanted_l = remainder % (lmax + 1);
        }
        double value = 0.0;
        for (I64 first = begin; first < end; ++first) {
            const double first_distance = sqrt(fmax(0.0, graph_distance2[first]));
            if (first_distance <= 1e-12) continue;
            const I32 first_atom = graph_atoms[first];
            const double first_cutoff = neighbor_radii == nullptr ? cutoff
                : (neighbor_radii[center] + neighbor_radii[first_atom]) * rcutfac;
            if (first_distance > first_cutoff) continue;
            const double first_weight = neighbor_weights == nullptr
                ? (kind == 1 ? static_cast<double>(numbers[first_atom]) : 1.0)
                : neighbor_weights[first_atom];
            const double first_radial = smooth_radial_device(first_distance, first_cutoff)
                * pow(fmax(0.0, (first_distance - rmin0) / fmax(first_cutoff - rmin0, 1e-12)), n1 + 1)
                * first_weight;
            for (I64 second = first; second < end; ++second) {
                const double second_distance = sqrt(fmax(0.0, graph_distance2[second]));
                if (second_distance <= 1e-12) continue;
                const I32 second_atom = graph_atoms[second];
                const double second_cutoff = neighbor_radii == nullptr ? cutoff
                    : (neighbor_radii[center] + neighbor_radii[second_atom]) * rcutfac;
                if (second_distance > second_cutoff) continue;
                const double second_weight = neighbor_weights == nullptr
                    ? (kind == 1 ? static_cast<double>(numbers[second_atom]) : 1.0)
                    : neighbor_weights[second_atom];
                const double second_radial = smooth_radial_device(second_distance, second_cutoff)
                    * pow(fmax(0.0, (second_distance - rmin0) / fmax(second_cutoff - rmin0, 1e-12)), n2 + 1)
                    * second_weight;
                const double* first_vector = graph_displacements + first * 3;
                const double* second_vector = graph_displacements + second * 3;
                const double denominator = first_distance * second_distance;
                const double cosine = denominator > 0.0
                    ? fmin(1.0, fmax(-1.0, (first_vector[0] * second_vector[0]
                        + first_vector[1] * second_vector[1]
                        + first_vector[2] * second_vector[2]) / denominator)) : 1.0;
                value += first_radial * second_radial * legendre_device(wanted_l, cosine);
            }
        }
        if (normalize_u) value *= 4.0 * kPi / sqrt(wanted_l + 1.0);
        target[feature] = value;
    }
    (void)rfac0;
}

} // namespace
