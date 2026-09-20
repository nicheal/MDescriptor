#pragma once
// Private CUDA support for the SOAP descriptor family.
#include "extended_descriptors_common.cuh"

namespace {

constexpr int kSoapGridBound = 128;
constexpr int kSoapMaxAngular = 20;

__device__ double soap_weight_device(
    int function,
    double r0,
    double c,
    double d,
    double m,
    double threshold,
    double w0,
    bool has_w0,
    bool exact_self,
    double distance,
    double species_weight) {
    double value = 1.0;
    const double ratio = r0 > 0.0 ? distance / r0 : 0.0;
    if (function == 1) {
        value = distance > r0 ? 0.0
            : c * pow(fmax(0.0, 1.0 + 2.0 * ratio * ratio * ratio
                - 3.0 * ratio * ratio), m);
    } else if (function == 2) {
        value = c / (d + pow(fmax(ratio, 1e-30), m));
    } else if (function == 3) {
        value = c / (d + exp(-ratio));
    }
    if (exact_self && has_w0) value = w0;
    return value * species_weight;
}

__device__ double soap_polynomial_flir(
    double distance,
    double radial_coordinate,
    int angular,
    double sigma) {
    const double eta = 1.0 / (2.0 * sigma * sigma);
    const double radial2 = radial_coordinate * radial_coordinate;
    if (distance <= 1e-14) {
        return angular == 0 ? exp(-eta * radial2) : 0.0;
    }
    const double denominator = eta * distance * radial_coordinate;
    if (fabs(denominator) <= 1e-30) return 0.0;
    const double prefactor = 0.25 / denominator;
    const double minus = exp(-eta * (radial_coordinate - distance)
        * (radial_coordinate - distance));
    const double plus = exp(-eta * (radial_coordinate + distance)
        * (radial_coordinate + distance));
    double previous = prefactor * (minus - plus);
    if (angular == 0) return previous;
    double current = prefactor * (minus + plus - 2.0 * previous);
    if (angular == 1) return current;
    for (int degree = 2; degree <= angular; ++degree) {
        const double next = fmax(0.0, previous - prefactor
            * (4.0 * degree - 2.0) * current);
        previous = current;
        current = next;
    }
    return current;
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

// Block-cooperative form of :c:func:`harmonic_values` for the SOAP
// coefficient kernel.  The serial version chains 231 local-memory recurrence
// steps through one lane per edge; here every value is computed by exactly
// one lane with the identical expression from shared-memory scratch, so the
// results are bit-identical while the work spreads across the block.
// ``output`` (441), ``legendre`` (231) and ``scalars`` (8) must be
// shared-memory arrays visible to the whole block; the caller enters this
// function with every lane and must not hold divergent barriers.
__device__ void harmonic_values_block(
    const double* vector,
    double* output,
    double* legendre,
    double* scalars,
    int requested,
    int lane,
    int threads) {
    auto legendre_index = [](int angular, int m) {
        return m + angular * (angular + 1) / 2;
    };
    constexpr double sqrt_1_over_2pi = 0.398942280401432677939946059934;
    constexpr double sqrt_3 = 1.732050807568877293527446341505872;
    constexpr double sqrt_3_over_2 = 1.224744871391589049098642;
    // scalars: [0] cos_theta, [1] sin_theta, [2] cos_phi, [3] sin_phi,
    //          [4] xy, [5] cos_m, [6] sin_m
    double value = 0.0;
    if (lane == 0) {
        const double norm = sqrt(vector[0] * vector[0] + vector[1] * vector[1]
            + vector[2] * vector[2]);
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
        const double xy = hypot(direction[0], direction[1]);
        const double cos_theta = direction[2];
        const double sin_theta = xy;
        scalars[0] = cos_theta;
        scalars[1] = sin_theta;
        scalars[4] = xy;
        legendre[legendre_index(0, 0)] = sqrt_1_over_2pi;
        value = -sqrt_3_over_2 * sin_theta * sqrt_1_over_2pi;
        if (requested > 0) {
            legendre[legendre_index(1, 0)] = cos_theta * sqrt_3 * sqrt_1_over_2pi;
            legendre[legendre_index(1, 1)] = value;
        }
        scalars[2] = xy > DBL_EPSILON ? direction[0] / xy : 1.0;
        scalars[3] = xy > DBL_EPSILON ? direction[1] / xy : 0.0;
    }
    __syncthreads();
    for (int angular = 2; angular <= requested; ++angular) {
        for (int m = lane; m < angular - 1; m += threads) {
            const double ls = static_cast<double>(angular * angular);
            const double lm1s = static_cast<double>((angular - 1) * (angular - 1));
            const double ms = static_cast<double>(m * m);
            const double a_coef = sqrt((4.0 * ls - 1.0) / (ls - ms));
            const double b_coef = -sqrt((lm1s - ms) / (4.0 * lm1s - 1.0));
            legendre[legendre_index(angular, m)] = a_coef * (
                scalars[0] * legendre[legendre_index(angular - 1, m)]
                + b_coef * legendre[legendre_index(angular - 2, m)]);
        }
        if (lane == 0) {
            legendre[legendre_index(angular, angular - 1)] = scalars[0]
                * sqrt(2.0 * angular + 1.0) * value;
            value *= -sqrt(1.0 + 0.5 / angular) * scalars[1];
            legendre[legendre_index(angular, angular)] = value;
        }
        __syncthreads();
    }
    for (int angular = lane; angular <= requested; angular += threads) {
        output[angular * angular + angular] =
            legendre[legendre_index(angular, 0)] / 1.414213562373095048801688724209698079;
    }
    const double minus_two_cos = -2.0 * scalars[2];
    double cos_previous = 1.0;
    double sin_previous = 0.0;
    double cos_current = -scalars[2];
    double sin_current = scalars[3];
    for (int m = 1; m <= requested; ++m) {
        if (lane == 0) {
            const double sin_m = minus_two_cos * sin_previous - sin_current;
            const double cos_m = minus_two_cos * cos_previous - cos_current;
            sin_current = sin_previous;
            sin_previous = sin_m;
            cos_current = cos_previous;
            cos_previous = cos_m;
            scalars[5] = cos_m;
            scalars[6] = sin_m;
        }
        __syncthreads();
        for (int angular = lane + m; angular <= requested; angular += threads) {
            output[angular * angular + angular + m] =
                legendre[legendre_index(angular, m)] * scalars[5];
            output[angular * angular + angular - m] =
                legendre[legendre_index(angular, m)] * scalars[6];
        }
        __syncthreads();
    }
}


__global__ void soap_coefficients_block_kernel(
    const I32* numbers,
    const I64* graph_offsets,
    const I32* graph_atoms,
    const I32* graph_shifts,
    const double* graph_displacements,
    const double* graph_distance2,
    const I32* species,
    int species_count,
    int coefficient_types,
    int radial_count,
    int max_angular,
    int radial_basis,
    double cutoff,
    double graph_cutoff,
    double sigma,
    const double* alphas,
    const double* betas,
    const double* radial_grid,
    int radial_grid_count,
    const double* radial_weights,
    const double* radial_values,
    int weighting_function,
    double weighting_r0,
    double weighting_c,
    double weighting_d,
    double weighting_m,
    double weighting_threshold,
    double weighting_w0,
    bool weighting_has_w0,
    const double* species_weights,
    I64 atoms,
    double* coefficients) {
    __shared__ double soap_eta_power[kSoapMaxAngular + 1];
    __shared__ double soap_radius_power[kSoapMaxAngular + 1];
    __shared__ double soap_prefactor[(kSoapMaxAngular + 1) * 32];
    __shared__ double soap_flir[(kSoapMaxAngular + 1) * kSoapGridBound];
    __shared__ double soap_radial_value[(kSoapMaxAngular + 1) * 32];
    __shared__ double soap_harmonics[(kSoapMaxAngular + 1) * (kSoapMaxAngular + 1)];
    __shared__ double soap_legendre[(kSoapMaxAngular + 1) * (kSoapMaxAngular + 2) / 2];
    __shared__ double soap_harmonics_scalars[8];
    __shared__ int soap_a_of_h[(kSoapMaxAngular + 1) * (kSoapMaxAngular + 1)];
    // One thread block per atom.  The previous one-thread-per-atom form left
    // the GPU idle for typical batches (256 atoms -> 256 threads).  Here the
    // block cooperates on every edge while each coefficient element is still
    // updated exactly once per edge, in edge order, so the per-element
    // accumulation sequence -- and therefore the bits -- match the serial
    // kernel.
    const I64 center = static_cast<I64>(blockIdx.x);
    if (center >= atoms) return;
    const int lane = static_cast<int>(threadIdx.x);
    const int harmonic_count = (max_angular + 1) * (max_angular + 1);
    const I64 coefficient_size = static_cast<I64>(coefficient_types)
        * radial_count * harmonic_count;
    double* target = coefficients + center * coefficient_size;
    for (I64 index = lane; index < coefficient_size; index += blockDim.x) {
        target[index] = 0.0;
    }
    // Harmonic index -> angular degree: degree a owns the contiguous index
    // range [a*a, a*a + 2a], i.e. exactly the integers with floor(sqrt(h))==a.
    for (int h = lane; h < harmonic_count; h += blockDim.x) {
        soap_a_of_h[h] = static_cast<int>(sqrt(static_cast<double>(h)));
    }
    const double eta = 1.0 / (2.0 * sigma * sigma);
    // pow(eta, angular) does not depend on the edge; evaluate it once.
    for (int a = lane; a <= max_angular; a += blockDim.x) {
        soap_eta_power[a] = pow(eta, a);
    }
    __syncthreads();
    const I64 begin = graph_offsets[center];
    const I64 end = graph_offsets[center + 1];
    for (I64 edge = begin; edge < end; ++edge) {
        const I32 atom = graph_atoms[edge];
        const int type = species_index(numbers[atom], species, species_count);
        if (type < 0) continue;
        const double distance2 = fmax(0.0, graph_distance2[edge]);
        const double distance = sqrt(distance2);
        if (distance >= graph_cutoff) continue;
        const bool exact_self = exact_self_edge(center, atom, graph_shifts, edge);
        const double weight = soap_weight_device(
            weighting_function, weighting_r0, weighting_c, weighting_d, weighting_m,
            weighting_threshold, weighting_w0, weighting_has_w0, exact_self,
            distance, species_weights == nullptr ? 1.0 : species_weights[type]);
        if (weight == 0.0) continue;
        harmonic_values_block(
            graph_displacements + edge * 3, soap_harmonics, soap_legendre,
            soap_harmonics_scalars, max_angular, lane,
            static_cast<int>(blockDim.x));
        for (int a = lane; a <= max_angular; a += blockDim.x) {
            soap_radius_power[a] = pow(distance, a);
        }
        // The GTO prefactor and the polynomial grid Flir depend only on the
        // (angular, raw) / (angular, grid) pair -- never on the radial
        // channel -- so they are evaluated once and shared by every radial.
        if (radial_basis == 0) {
            for (int idx = lane; idx < (max_angular + 1) * radial_count; idx += blockDim.x) {
                const int angular = idx / radial_count;
                const int raw = idx % radial_count;
                const double alpha = alphas[angular * radial_count + raw];
                const double denominator = alpha + eta;
                soap_prefactor[idx] = soap_eta_power[angular]
                    * pow(denominator, -angular - 1.5)
                    * exp(-alpha * eta / denominator * distance2);
            }
        } else {
            for (int idx = lane; idx < (max_angular + 1) * radial_grid_count; idx += blockDim.x) {
                const int angular = idx / radial_grid_count;
                const int q = idx % radial_grid_count;
                soap_flir[idx] = soap_polynomial_flir(
                    distance, radial_grid[q], angular, sigma);
            }
        }
        __syncthreads();
        for (int idx = lane; idx < (max_angular + 1) * radial_count; idx += blockDim.x) {
            const int angular = idx / radial_count;
            const int radial = idx % radial_count;
            double radial_value = 0.0;
            if (radial_basis == 0) {
                for (int raw = 0; raw < radial_count; ++raw) {
                    radial_value += betas[(angular * radial_count + radial)
                        * radial_count + raw] * soap_prefactor[angular * radial_count + raw];
                }
                radial_value = kPi * sqrt(kPi) * radial_value;
            } else {
                for (int q = 0; q < radial_grid_count; ++q) {
                    radial_value += radial_weights[q] * radial_grid[q] * radial_grid[q]
                        * soap_flir[angular * radial_grid_count + q]
                        * radial_values[radial * radial_grid_count + q];
                }
                radial_value *= 4.0 * kPi;
            }
            soap_radial_value[idx] = radial_value;
        }
        __syncthreads();
        // One lane per (radial, harmonic) element per edge; the element's
        // update order across edges is unchanged from the serial kernel.
        const int destination_type = coefficient_types == 1 ? 0 : type;
        const int touched = radial_count * harmonic_count;
        for (int t = lane; t < touched; t += blockDim.x) {
            const int radial = t / harmonic_count;
            const int h = t - radial * harmonic_count;
            const int angular = soap_a_of_h[h];
            const double angular_factor = radial_basis == 0
                ? soap_radius_power[angular] : 1.0;
            target[(destination_type * radial_count + radial) * harmonic_count + h]
                += weight * soap_radial_value[angular * radial_count + radial]
                * angular_factor * soap_harmonics[h];
        }
        __syncthreads();
        // mu2 combines species densities in one coefficient block; the loop
        // above already accumulates every neighbor into that block.
    }
}


// Two-phase SOAP expansion.  Phase 1 runs one thread per edge and computes
// the spherical harmonics, the species-summed radial values, and the
// per-degree distance powers exactly as the serial kernel did, into a
// per-edge scratch slot; running it across all edges of an atom tile keeps
// the GPU busy with no block-wide barriers.  Phase 2 runs one thread per
// atom and streams the scratch into the coefficient rows in edge order, so
// every accumulator sees the same addition sequence as the serial kernel and
// the results stay bit-identical.
//
// Scratch slot layout (doubles):
//   [0]                       weight (0.0 marks a skipped edge)
//   [1]                       destination species block
//   [2 .. 2+(L+1)*R)          radial values rv[angular * radial_count + radial]
//   [.. + (L+1))              radius_power[angular] (GTO only; 1.0 for poly)
//   [.. + harmonic_count)     spherical harmonics


__global__ void soap_edge_scratch_kernel(
    const I32* numbers,
    const I64* graph_offsets,
    const I32* graph_atoms,
    const I32* graph_shifts,
    const double* graph_displacements,
    const double* graph_distance2,
    const I32* species,
    int species_count,
    int coefficient_types,
    int radial_count,
    int max_angular,
    int radial_basis,
    double cutoff,
    double graph_cutoff,
    double sigma,
    const double* alphas,
    const double* betas,
    const double* radial_grid,
    int radial_grid_count,
    const double* radial_weights,
    const double* radial_values,
    int weighting_function,
    double weighting_r0,
    double weighting_c,
    double weighting_d,
    double weighting_m,
    double weighting_threshold,
    double weighting_w0,
    bool weighting_has_w0,
    const double* species_weights,
    I64 atoms,
    I64 tile_edge_begin,
    I64 tile_edge_end,
    I64 scratch_stride,
    double* scratch) {
    const I64 edge = tile_edge_begin
        + static_cast<I64>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (edge >= tile_edge_end) return;
    double* slot = scratch + (edge - tile_edge_begin) * scratch_stride;
    const int harmonic_count = (max_angular + 1) * (max_angular + 1);
    const I64 center = center_for_edge(graph_offsets, atoms, edge);
    const I32 atom = graph_atoms[edge];
    const int type = species_index(numbers[atom], species, species_count);
    const double distance2 = fmax(0.0, graph_distance2[edge]);
    const double distance = sqrt(distance2);
    const bool exact_self = exact_self_edge(center, atom, graph_shifts, edge);
    double weight = 0.0;
    if (type >= 0 && distance < graph_cutoff) {
        weight = soap_weight_device(
            weighting_function, weighting_r0, weighting_c, weighting_d, weighting_m,
            weighting_threshold, weighting_w0, weighting_has_w0, exact_self,
            distance, species_weights == nullptr ? 1.0 : species_weights[type]);
    }
    if (weight == 0.0) {
        // Phase 2 skips edges whose weight is zero, matching the serial
        // kernel's skip conditions (unknown species, cutoff, zero weight).
        slot[0] = 0.0;
        return;
    }
    const int destination_type = coefficient_types == 1 ? 0 : type;
    double harmonics[441]{};
    harmonic_values<20>(graph_displacements + edge * 3, harmonics, max_angular);
    const double eta = 1.0 / (2.0 * sigma * sigma);
    double* rv = slot + 2;
    double* radius_power = rv + (max_angular + 1) * radial_count;
    double* harmonics_out = radius_power + (max_angular + 1);
    for (int angular = 0; angular <= max_angular; ++angular) {
        radius_power[angular] = pow(distance, angular);
        // The GTO prefactor and the polynomial grid Flir depend only on the
        // (angular, raw) / (angular, grid) pair -- never on the radial
        // channel -- so they are evaluated once per angular degree and shared
        // by every radial (the serial kernel recomputed them radial_count
        // times per edge).
        const double eta_power = pow(eta, angular);
        double prefactors[kSoapGridBound];
        if (radial_basis == 0) {
            for (int raw = 0; raw < radial_count; ++raw) {
                const double alpha = alphas[angular * radial_count + raw];
                const double denominator = alpha + eta;
                prefactors[raw] = eta_power
                    * pow(denominator, -angular - 1.5)
                    * exp(-alpha * eta / denominator * distance2);
            }
        } else {
            for (int q = 0; q < radial_grid_count; ++q) {
                prefactors[q] = soap_polynomial_flir(
                    distance, radial_grid[q], angular, sigma);
            }
        }
        for (int radial = 0; radial < radial_count; ++radial) {
            double radial_value = 0.0;
            if (radial_basis == 0) {
                for (int raw = 0; raw < radial_count; ++raw) {
                    radial_value += betas[(angular * radial_count + radial)
                        * radial_count + raw] * prefactors[raw];
                }
                radial_value = kPi * sqrt(kPi) * radial_value;
            } else {
                for (int q = 0; q < radial_grid_count; ++q) {
                    radial_value += radial_weights[q] * radial_grid[q] * radial_grid[q]
                        * prefactors[q]
                        * radial_values[radial * radial_grid_count + q];
                }
                radial_value *= 4.0 * kPi;
            }
            rv[angular * radial_count + radial] = radial_value;
        }
    }
    slot[0] = weight;
    slot[1] = static_cast<double>(destination_type);
    for (int h = 0; h < harmonic_count; ++h) harmonics_out[h] = harmonics[h];
}

// Phase 2a of the two-phase expansion: one thread per (atom, edge segment)
// folds its segment's precomputed per-edge values into a private partial
// coefficient row.  Edges are visited in ascending order, so each partial is
// a contiguous in-order partial sum of the serial accumulation.
__global__ void soap_segment_accumulate_kernel(
    const I64* graph_offsets,
    I64 tile_atom_begin,
    I64 tile_atom_end,
    I64 tile_edge_begin,
    I64 segment_size,
    int reduce_segments,
    int coefficient_types,
    int radial_count,
    int max_angular,
    int radial_basis,
    I64 scratch_stride,
    const double* scratch,
    double* partials) {
    const int index = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int tile_atoms = static_cast<int>(tile_atom_end - tile_atom_begin);
    if (index >= tile_atoms * reduce_segments) return;
    const int atom_in_tile = index / reduce_segments;
    const int segment = index % reduce_segments;
    const I64 center = tile_atom_begin + atom_in_tile;
    const int harmonic_count = (max_angular + 1) * (max_angular + 1);
    const I64 coefficient_size = static_cast<I64>(coefficient_types)
        * radial_count * harmonic_count;
    double* partial = partials
        + (static_cast<I64>(atom_in_tile) * reduce_segments + segment)
        * coefficient_size;
    for (I64 element = 0; element < coefficient_size; ++element) {
        partial[element] = 0.0;
    }
    const I64 atom_edge_begin = graph_offsets[center];
    const I64 atom_edge_end = graph_offsets[center + 1];
    const I64 segment_begin = atom_edge_begin
        + static_cast<I64>(segment) * segment_size;
    const I64 segment_end = min(atom_edge_end,
        atom_edge_begin + static_cast<I64>(segment + 1) * segment_size);
    int angular_of[(kSoapMaxAngular + 1) * (kSoapMaxAngular + 1)];
    for (int h = 0; h < harmonic_count; ++h) {
        angular_of[h] = static_cast<int>(sqrt(static_cast<double>(h)));
    }
    for (I64 edge = segment_begin; edge < segment_end; ++edge) {
        const double* slot = scratch + (edge - tile_edge_begin) * scratch_stride;
        const double weight = slot[0];
        if (weight == 0.0) continue;
        const int destination_type = static_cast<int>(slot[1]);
        const double* rv = slot + 2;
        const double* radius_power = rv + (max_angular + 1) * radial_count;
        const double* harmonics = radius_power + (max_angular + 1);
        for (int radial = 0; radial < radial_count; ++radial) {
            double* destination = partial
                + (destination_type * radial_count + radial) * harmonic_count;
            for (int h = 0; h < harmonic_count; ++h) {
                const int angular = angular_of[h];
                const double angular_factor = radial_basis == 0
                    ? radius_power[angular] : 1.0;
                destination[h] += weight * rv[angular * radial_count + radial]
                    * angular_factor * harmonics[h];
            }
        }
    }
}

// Phase 2b: combine each atom's segment partials in segment order.  With a
// single non-empty segment this is a verbatim copy of the serial
// accumulation; with several, it is a fixed-order reduction whose only
// rounding differences sit at the double-precision epsilon level.
__global__ void soap_segment_reduce_kernel(
    const I64* graph_offsets,
    I64 tile_atom_begin,
    I64 tile_atom_end,
    I64 segment_size,
    int reduce_segments,
    I64 row_size,
    const double* partials,
    double* coefficients) {
    const I64 index = static_cast<I64>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int tile_atoms = static_cast<int>(tile_atom_end - tile_atom_begin);
    if (index >= static_cast<I64>(tile_atoms) * row_size) return;
    const int atom_in_tile = static_cast<int>(index / row_size);
    const I64 element = index - static_cast<I64>(atom_in_tile) * row_size;
    const I64 center = tile_atom_begin + atom_in_tile;
    const I64 atom_edge_count = graph_offsets[center + 1] - graph_offsets[center];
    const int segments = static_cast<int>((atom_edge_count
        + segment_size - 1) / segment_size);
    double value = 0.0;
    for (int segment = 0; segment < segments; ++segment) {
        value += partials[(static_cast<I64>(atom_in_tile) * reduce_segments
            + segment) * row_size + element];
    }
    coefficients[center * row_size + element] = value;
}

__device__ double soap_coefficient_at(
    const double* coefficients,
    int type,
    int radial,
    int harmonic,
    int radial_count,
    int harmonic_count) {
    return coefficients[(type * radial_count + radial) * harmonic_count + harmonic];
}

__device__ double soap_power_feature(
    const double* coefficients,
    int feature,
    int species_count,
    int coefficient_types,
    int radial_count,
    int max_angular,
    int compression) {
    const int harmonic_count = (max_angular + 1) * (max_angular + 1);
    int first = 0;
    int second = 0;
    int angular = 0;
    int n1 = 0;
    int n2 = 0;
    int remainder = feature;
    if (compression == 1) {
        for (angular = 0; angular <= max_angular; ++angular) {
            const int block = radial_count * (radial_count + 1) / 2;
            if (remainder < block) break;
            remainder -= block;
        }
        for (n1 = 0; n1 < radial_count; ++n1) {
            const int block = radial_count - n1;
            if (remainder < block) { n2 = n1 + remainder; break; }
            remainder -= block;
        }
    } else if (compression == 2) {
        const int per_type = (max_angular + 1) * radial_count * radial_count;
        first = feature / per_type;
        remainder = feature % per_type;
        angular = remainder / (radial_count * radial_count);
        remainder %= radial_count * radial_count;
        n1 = remainder / radial_count;
        n2 = remainder % radial_count;
        second = -1; // mu1nu1 uses a sum over the second species below.
    } else if (compression == 3) {
        const int per_type = (max_angular + 1) * radial_count * (radial_count + 1) / 2;
        first = feature / per_type;
        remainder = feature % per_type;
        for (angular = 0; angular <= max_angular; ++angular) {
            const int block = radial_count * (radial_count + 1) / 2;
            if (remainder < block) break;
            remainder -= block;
        }
        for (n1 = 0; n1 < radial_count; ++n1) {
            const int block = radial_count - n1;
            if (remainder < block) { n2 = n1 + remainder; break; }
            remainder -= block;
        }
        second = first;
    } else {
        // The uncompressed CPU layout uses triangular radial blocks for
        // same-species pairs and rectangular blocks for cross-species pairs.
        // Walk those variable-sized blocks before decoding l,n1,n2.
        bool decoded = false;
        for (first = 0; first < species_count; ++first) {
            for (second = first; second < species_count; ++second) {
                const int radial_pairs = first == second
                    ? radial_count * (radial_count + 1) / 2
                    : radial_count * radial_count;
                const int block = (max_angular + 1) * radial_pairs;
                if (remainder < block) {
                    angular = remainder / radial_pairs;
                    remainder %= radial_pairs;
                    if (first == second) {
                        for (n1 = 0; n1 < radial_count; ++n1) {
                            const int count = radial_count - n1;
                            if (remainder < count) {
                                n2 = n1 + remainder;
                                break;
                            }
                            remainder -= count;
                        }
                    } else {
                        n1 = remainder / radial_count;
                        n2 = remainder % radial_count;
                    }
                    decoded = true;
                    break;
                }
                remainder -= block;
            }
            if (decoded) break;
        }
    }
    double sum = 0.0;
    const int first_type = coefficient_types == 1 ? 0 : first;
    for (int m = -angular; m <= angular; ++m) {
        sum += soap_coefficient_at(
            coefficients, first_type, n1, angular * angular + angular + m,
            radial_count, harmonic_count)
            * soap_coefficient_at(
                coefficients, coefficient_types == 1 ? 0 : second, n2,
                angular * angular + angular + m, radial_count, harmonic_count);
    }
    return kPi * sqrt(8.0 / (2.0 * angular + 1.0)) * sum;
}

// One thread per (row, feature).  Every power-spectrum entry is independent,
// so the feature dimension parallelizes the work without touching the
// per-entry summation order.
__global__ void soap_power_kernel(
    const double* coefficients,
    I64 rows,
    I64 coefficient_stride,
    int features,
    int species_count,
    int coefficient_types,
    int radial_count,
    int max_angular,
    int compression,
    double* output) {
    const I64 index = static_cast<I64>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= static_cast<I64>(rows) * features) return;
    const I64 row = index / features;
    output[index] = soap_power_feature(
        coefficients + row * coefficient_stride, static_cast<int>(index % features),
        species_count, coefficient_types, radial_count, max_angular, compression);
}

// mu1nu1 phase 1: build the species-summed density in global scratch.  Each
// thread owns one (row, coefficient) element and accumulates the species
// blocks in the same sequential order the previous per-thread local array
// used, keeping the results bit-identical without the 112 KiB per-thread
// local-memory footprint.
__global__ void soap_mu1nu1_sum_kernel(
    const double* coefficients,
    I64 rows,
    I64 coefficient_stride,
    int sum_count,
    int species_count,
    int radial_count,
    int harmonic_count,
    double* summed) {
    const I64 index = static_cast<I64>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= static_cast<I64>(rows) * sum_count) return;
    const I64 row = index / sum_count;
    const int element = static_cast<int>(index % sum_count);
    const double* source = coefficients + row * coefficient_stride;
    double value = 0.0;
    for (int type = 0; type < species_count; ++type) {
        value += source[type * radial_count * harmonic_count + element];
    }
    summed[index] = value;
}

// mu1nu1 phase 2: one thread per (row, feature), reading the species-summed
// density from scratch with the same per-feature m-summation order.
__global__ void soap_power_mu1nu1_kernel(
    const double* coefficients,
    const double* summed,
    I64 rows,
    I64 coefficient_stride,
    int features,
    int radial_count,
    int max_angular,
    double* output) {
    const I64 index = static_cast<I64>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= static_cast<I64>(rows) * features) return;
    const int harmonic_count = (max_angular + 1) * (max_angular + 1);
    const int sum_count = radial_count * harmonic_count;
    const I64 row = index / features;
    const int feature = static_cast<int>(index % features);
    const double* source = coefficients + row * coefficient_stride;
    const double* summed_row = summed + row * sum_count;
    int remainder = feature;
    const int per_type = (max_angular + 1) * radial_count * radial_count;
    const int first = feature / per_type;
    remainder %= per_type;
    int angular = remainder / (radial_count * radial_count);
    remainder %= radial_count * radial_count;
    const int n1 = remainder / radial_count;
    const int n2 = remainder % radial_count;
    double value = 0.0;
    for (int m = -angular; m <= angular; ++m) {
        const int harmonic = angular * angular + angular + m;
        value += soap_coefficient_at(
            source, first, n1, harmonic, radial_count, harmonic_count)
            * summed_row[n2 * harmonic_count + harmonic];
    }
    output[index] = kPi * sqrt(8.0 / (2.0 * angular + 1.0)) * value;
}

__global__ void soap_average_coefficients_kernel(
    const I64* offsets,
    I64 structures,
    I64 coefficient_stride,
    const double* atom_coefficients,
    double* structure_coefficients) {
    const I64 structure = static_cast<I64>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (structure >= structures) return;
    const I64 begin = offsets[structure];
    const I64 end = offsets[structure + 1];
    double* target = structure_coefficients + structure * coefficient_stride;
    for (I64 index = 0; index < coefficient_stride; ++index) target[index] = 0.0;
    if (end <= begin) return;
    for (I64 atom = begin; atom < end; ++atom) {
        const double* source = atom_coefficients + atom * coefficient_stride;
        for (I64 index = 0; index < coefficient_stride; ++index) target[index] += source[index];
    }
    const double scale = 1.0 / static_cast<double>(end - begin);
    for (I64 index = 0; index < coefficient_stride; ++index) target[index] *= scale;
}

__global__ void soap_average_power_kernel(
    const I64* offsets,
    I64 structures,
    int features,
    const double* atom_power,
    double* output) {
    const I64 structure = static_cast<I64>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (structure >= structures) return;
    const I64 begin = offsets[structure];
    const I64 end = offsets[structure + 1];
    double* target = output + structure * features;
    for (int feature = 0; feature < features; ++feature) target[feature] = 0.0;
    if (end <= begin) return;
    for (I64 atom = begin; atom < end; ++atom) {
        const double* source = atom_power + atom * features;
        for (int feature = 0; feature < features; ++feature) target[feature] += source[feature];
    }
    const double scale = 1.0 / static_cast<double>(end - begin);
    for (int feature = 0; feature < features; ++feature) target[feature] *= scale;
}

py::dict compute_soap_descriptor(
    CudaExecutionContext& context,
    DeviceBatch& batch,
    DeviceNeighborGraph& graph,
    const detail::StructureBatchView& host_batch,
    const py::dict& options) {
    (void)graph;
    const auto species = species_option(options);
    if (species.empty()) throw std::invalid_argument("SOAP species must not be empty");
    const int radial_count = option(options, "n_max", 8);
    const int max_angular = option(options, "l_max", 6);
    const double cutoff = option(options, "r_cut", 6.0);
    const double sigma = option(options, "sigma", 1.0);
    if (radial_count < 1 || radial_count > 32 || max_angular < 0 || max_angular > 20
        || cutoff <= 0.0 || sigma <= 0.0) {
        throw std::invalid_argument("invalid CUDA SOAP parameters");
    }
    const std::string radial_name = option(options, "rbf", std::string("gto"));
    const int radial_basis = radial_name == "gto" ? 0 : radial_name == "polynomial" ? 1 : -1;
    if (radial_basis < 0) throw std::invalid_argument("unsupported CUDA SOAP radial basis");
    const py::dict compression_object = child_dict(options, "compression");
    const std::string compression_name = option(
        compression_object, "mode", std::string("off"));
    const int compression = compression_name == "off" ? 0
        : compression_name == "mu2" ? 1
        : compression_name == "mu1nu1" ? 2
        : compression_name == "crossover" ? 3 : -1;
    if (compression < 0) throw std::invalid_argument("unsupported CUDA SOAP compression");
    const std::string average = option(options, "average", std::string("inner"));
    if (average != "off" && average != "inner" && average != "outer") {
        throw std::invalid_argument("unsupported CUDA SOAP average mode");
    }
    const int species_count = static_cast<int>(species.size());
    const int coefficient_types = compression == 1 ? 1 : species_count;
    I64 computed_features = 0;
    if (compression == 1) {
        computed_features = static_cast<I64>(radial_count) * (radial_count + 1) / 2
            * (max_angular + 1);
    } else if (compression == 2) {
        computed_features = static_cast<I64>(species_count) * radial_count * radial_count
            * (max_angular + 1);
    } else if (compression == 3) {
        computed_features = static_cast<I64>(species_count) * radial_count
            * (radial_count + 1) / 2 * (max_angular + 1);
    } else {
        computed_features = static_cast<I64>(species_count) * (species_count + 1) / 2
            * radial_count * radial_count * (max_angular + 1);
        // Same-species radial pairs are triangular, while cross-species pairs
        // retain the full rectangular radial block.
        computed_features = 0;
        for (int first = 0; first < species_count; ++first) {
            for (int second = first; second < species_count; ++second) {
                computed_features += static_cast<I64>(
                    first == second ? radial_count * (radial_count + 1) / 2
                                    : radial_count * radial_count) * (max_angular + 1);
            }
        }
    }
    const I64 features = feature_count_option(options, computed_features);
    if (features != computed_features) {
        throw std::invalid_argument("CUDA SOAP feature count does not match its layout");
    }
    const py::dict payload = child_dict(options, "_cuda_payload");
    const py::dict radial_payload = child_dict(payload, "radial_basis");
    auto alphas = vector_child(radial_payload, "alphas");
    auto betas = vector_child(radial_payload, "betas");
    auto radial_grid = vector_child(radial_payload, "radial_grid");
    auto radial_weights = vector_child(radial_payload, "radial_weights");
    auto radial_values = vector_child(radial_payload, "radial_values");
    if (radial_basis == 0 && (alphas.size() != static_cast<std::size_t>((max_angular + 1) * radial_count)
        || betas.size() != static_cast<std::size_t>((max_angular + 1) * radial_count * radial_count))) {
        throw std::invalid_argument("CUDA SOAP GTO payload has an invalid shape");
    }
    if (radial_basis == 1 && (radial_grid.size() < 2
        || radial_grid.size() > static_cast<std::size_t>(kSoapGridBound)
        || radial_weights.size() != radial_grid.size()
        || radial_values.size() != radial_grid.size() * static_cast<std::size_t>(radial_count))) {
        throw std::invalid_argument("CUDA SOAP polynomial payload has an invalid shape");
    }
    const py::dict weighting = child_dict(options, "weighting");
    const std::string weighting_name = option(weighting, "function", std::string());
    const int weighting_function = weighting_name == "" ? 0
        : weighting_name == "poly" ? 1
        : weighting_name == "pow" ? 2
        : weighting_name == "exp" ? 3 : -1;
    if (weighting_function < 0) throw std::invalid_argument("unsupported CUDA SOAP weighting");
    const py::object species_weight_object = compression_object.contains("species_weighting")
        ? compression_object["species_weighting"] : py::none();
    const auto species_weights = species_dictionary_values(species_weight_object, species, 1.0);
    const double padding = radial_basis == 0 ? sigma * sqrt(-2.0 * log(1e-3)) : 0.0;
    const double graph_cutoff = cutoff + padding;
    graph.build_dpa(context, batch, host_batch, graph_cutoff, true, false, true);
    DeviceBuffer<I32> d_species;
    DeviceBuffer<double> d_species_weights;
    DeviceBuffer<double> d_alphas;
    DeviceBuffer<double> d_betas;
    DeviceBuffer<double> d_grid;
    DeviceBuffer<double> d_radial_weights;
    DeviceBuffer<double> d_values;
    d_species.upload(species.data(), species.size(), context.stream(), "could not upload SOAP species");
    d_species_weights.upload(species_weights.data(), species_weights.size(), context.stream(), "could not upload SOAP species weights");
    d_alphas.upload(alphas.data(), alphas.size(), context.stream(), "could not upload SOAP GTO alphas");
    d_betas.upload(betas.data(), betas.size(), context.stream(), "could not upload SOAP GTO betas");
    d_grid.upload(radial_grid.data(), radial_grid.size(), context.stream(), "could not upload SOAP polynomial grid");
    d_radial_weights.upload(
        radial_weights.data(), radial_weights.size(), context.stream(),
        "could not upload SOAP polynomial quadrature weights");
    d_values.upload(radial_values.data(), radial_values.size(), context.stream(), "could not upload SOAP polynomial basis");
    const int harmonic_count = (max_angular + 1) * (max_angular + 1);
    const I64 coefficient_stride = static_cast<I64>(coefficient_types) * radial_count * harmonic_count;
    const std::size_t coefficient_size = static_cast<std::size_t>(batch.atoms())
        * static_cast<std::size_t>(coefficient_stride);
    const bool inner = average == "inner";
    const bool outer = average == "outer";
    const I64 rows = inner || outer ? batch.structures() : batch.atoms();
    const std::size_t output_size = static_cast<std::size_t>(rows)
        * static_cast<std::size_t>(features);
    double* output = context.output_buffer(output_size);
    const std::size_t power_size = outer ? static_cast<std::size_t>(batch.atoms())
        * static_cast<std::size_t>(features) : 0U;
    const std::size_t average_size = inner ? static_cast<std::size_t>(batch.structures())
        * static_cast<std::size_t>(coefficient_stride) : 0U;
    // mu1nu1 keeps the species-summed density in a scratch slot sized by the
    // rows the power kernel consumes (atom rows, or structure rows for the
    // inner average) instead of a per-thread local array.
    const int sum_count = radial_count * harmonic_count;
    const I64 power_rows = inner ? batch.structures() : batch.atoms();
    const std::size_t summed_size = compression == 2
        ? static_cast<std::size_t>(power_rows) * static_cast<std::size_t>(sum_count)
        : 0U;
    // Per-edge scratch slot for the two-phase expansion (see the phase-1
    // kernel): weight, destination block, radial values, radius powers, and
    // harmonics for one edge.
    const I64 edge_scratch_stride = 2 + (max_angular + 1) * radial_count
        + (max_angular + 1) + harmonic_count;
    constexpr I64 kSoapTileEdgeCap = 32768;
    constexpr unsigned soap_block_threads = 256;
    // Dense batches (many directed edges per atom) run the block-cooperative
    // kernel: its per-atom edge loop stays latency-friendly, while the
    // two-phase path's per-atom accumulate thread would starve.  Wide batches
    // and small batches run the two-phase path, whose edge-parallel
    // precompute covers both.  Measured crossover on RTX 2080 sits near 490
    // directed edges per atom at 256 atoms, with the cooperative form winning
    // from roughly 256 edges per atom upward once at least 64 blocks are
    // available.  Both paths update every coefficient element exactly once
    // per edge in edge order, so they are bit-identical.
    const bool dense = batch.atoms() >= 64
        && graph.pairs() >= 256 * static_cast<I64>(batch.atoms());
    std::size_t edge_scratch_slots = 0;
    I64 max_atom_edges = 0;
    std::vector<I64> host_offsets;
    if (output_size > 0) {
        zeroed_output(context, output, output_size, "could not clear CUDA SOAP output");
    }
    if (batch.atoms() > 0) {
        // Host copy of the graph offsets plans edge-bounded atom tiles.  The
        // per-edge scratch is sized by the tile cap -- or by a single atom
        // whose neighbor count alone exceeds the cap -- and never by the
        // whole batch.
        host_offsets.resize(batch.atoms() + 1);
        check_cuda(cudaMemcpyAsync(host_offsets.data(), graph.offsets(),
            host_offsets.size() * sizeof(I64), cudaMemcpyDeviceToHost,
            context.stream()), "could not download the SOAP graph offsets");
        check_cuda(cudaStreamSynchronize(context.stream()),
            "could not sync the SOAP graph offsets");
        for (std::size_t atom = 0; atom + 1 < host_offsets.size(); ++atom) {
            max_atom_edges = std::max(max_atom_edges,
                host_offsets[atom + 1] - host_offsets[atom]);
        }
        edge_scratch_slots = std::max<I64>(
            std::min<I64>(graph.pairs(), kSoapTileEdgeCap), max_atom_edges);
    }
    // Segment partials: one coefficient row per (atom in tile, segment).
    constexpr int kSoapReduceSegments = 8;
    I64 max_tile_atoms = 0;
    if (batch.atoms() > 0) {
        I64 atom_begin = 0;
        while (atom_begin < batch.atoms()) {
            I64 atom_end = atom_begin + 1;
            while (atom_end < batch.atoms() && atom_end - atom_begin < 256
                && host_offsets[atom_end + 1] - host_offsets[atom_begin] <= kSoapTileEdgeCap) {
                ++atom_end;
            }
            max_tile_atoms = std::max(max_tile_atoms, atom_end - atom_begin);
            atom_begin = atom_end;
        }
    }
    const std::size_t partials_size = static_cast<std::size_t>(max_tile_atoms)
        * kSoapReduceSegments * coefficient_size;
    const std::size_t workspace_size = coefficient_size + power_size + average_size
        + summed_size + edge_scratch_slots * static_cast<std::size_t>(edge_scratch_stride)
        + partials_size;
    auto* workspace = static_cast<double*>(context.workspace_buffer(
        workspace_size * sizeof(double)));
    double* coefficients = workspace;
    double* atom_power = outer ? coefficients + coefficient_size : nullptr;
    double* structure_coefficients = inner
        ? coefficients + coefficient_size + power_size : nullptr;
    double* summed_scratch = compression == 2
        ? coefficients + coefficient_size + power_size + average_size
        : nullptr;
    double* edge_scratch = coefficients + coefficient_size + power_size
        + average_size + summed_size;
    double* partials = edge_scratch
        + edge_scratch_slots * static_cast<std::size_t>(edge_scratch_stride);
    // Two-phase expansion with a deterministic segmented reduction: phase 1
    // precomputes per-edge values (edge-parallel, no barriers), phase 2a
    // folds edge segments into per-atom partial coefficient rows, and
    // phase 2b combines the partials in fixed segment order.  At most
    // kSoapReduceSegments partials are combined, so run-to-run results are
    // reproducible; wide batches have a single segment per atom and stay
    // bit-identical to the serial accumulation.
    const I64 segment_size = std::max<I64>(1,
        (max_atom_edges + kSoapReduceSegments - 1) / kSoapReduceSegments);
    if (batch.atoms() > 0 && dense) {
        soap_coefficients_block_kernel<<<static_cast<unsigned>(batch.atoms()),
            soap_block_threads, 0, context.stream()>>>(
            batch.numbers(), graph.offsets(), graph.atoms(), graph.shifts(), graph.displacements(),
            graph.distance2(), d_species.get(), species_count, coefficient_types, radial_count,
            max_angular, radial_basis, cutoff, graph_cutoff, sigma,
            d_alphas.get(), d_betas.get(), d_grid.get(),
            static_cast<int>(radial_grid.size()), d_radial_weights.get(), d_values.get(), weighting_function,
            option(weighting, "r0", 1.0), option(weighting, "c", 1.0),
            option(weighting, "d", 0.0), option(weighting, "m", 1.0),
            option(weighting, "threshold", 1e-2), option(weighting, "w0", 1.0),
            weighting.contains("w0"), d_species_weights.get(), batch.atoms(), coefficients);
        check_cuda(cudaGetLastError(), "CUDA SOAP block coefficient kernel launch failed");
    }
    if (batch.atoms() > 0 && !dense) {
        I64 atom_begin = 0;
        while (atom_begin < batch.atoms()) {
            I64 atom_end = atom_begin + 1;
            while (atom_end < batch.atoms() && atom_end - atom_begin < 256
                && host_offsets[atom_end + 1] - host_offsets[atom_begin] <= kSoapTileEdgeCap) {
                ++atom_end;
            }
            const I64 tile_edge_begin = host_offsets[atom_begin];
            const I64 tile_edge_end = host_offsets[atom_end];
            const I64 tile_edges = tile_edge_end - tile_edge_begin;
            const int tile_atoms = static_cast<int>(atom_end - atom_begin);
            soap_edge_scratch_kernel<<<static_cast<unsigned>((tile_edges + 127) / 128),
                128, 0, context.stream()>>>(
                batch.numbers(), graph.offsets(), graph.atoms(), graph.shifts(), graph.displacements(),
                graph.distance2(), d_species.get(), species_count, coefficient_types, radial_count,
                max_angular, radial_basis, cutoff, graph_cutoff, sigma,
                d_alphas.get(), d_betas.get(), d_grid.get(),
                static_cast<int>(radial_grid.size()), d_radial_weights.get(), d_values.get(), weighting_function,
                option(weighting, "r0", 1.0), option(weighting, "c", 1.0),
                option(weighting, "d", 0.0), option(weighting, "m", 1.0),
                option(weighting, "threshold", 1e-2), option(weighting, "w0", 1.0),
                weighting.contains("w0"), d_species_weights.get(), batch.atoms(),
                tile_edge_begin, tile_edge_end, edge_scratch_stride, edge_scratch);
            check_cuda(cudaGetLastError(), "CUDA SOAP edge scratch launch failed");
            const int accumulate_threads = tile_atoms * kSoapReduceSegments;
            soap_segment_accumulate_kernel<<<static_cast<unsigned>((accumulate_threads + 127) / 128),
                128, 0, context.stream()>>>(
                graph.offsets(), atom_begin, atom_end, tile_edge_begin,
                segment_size, kSoapReduceSegments, coefficient_types, radial_count,
                max_angular, radial_basis, edge_scratch_stride, edge_scratch, partials);
            check_cuda(cudaGetLastError(), "CUDA SOAP segment accumulate launch failed");
            const I64 reduce_threads = static_cast<I64>(tile_atoms) * coefficient_stride;
            soap_segment_reduce_kernel<<<static_cast<unsigned>((reduce_threads + 255) / 256),
                256, 0, context.stream()>>>(
                graph.offsets(), atom_begin, atom_end, segment_size,
                kSoapReduceSegments, coefficient_stride, partials, coefficients);
            check_cuda(cudaGetLastError(), "CUDA SOAP segment reduce launch failed");
            atom_begin = atom_end;
        }
    }
    constexpr unsigned block_size = 64;
    auto launch_power = [&](const double* source_rows, I64 power_row_count,
                            double* destination) {
        if (power_row_count <= 0) return;
        if (compression == 2) {
            soap_mu1nu1_sum_kernel<<<static_cast<unsigned>(
                (power_row_count * sum_count + block_size - 1) / block_size),
                block_size, 0, context.stream()>>>(
                source_rows, power_row_count, coefficient_stride, sum_count,
                species_count, radial_count, harmonic_count, summed_scratch);
            check_cuda(cudaGetLastError(), "CUDA SOAP mu1nu1 density sum launch failed");
            soap_power_mu1nu1_kernel<<<static_cast<unsigned>(
                (power_row_count * features + block_size - 1) / block_size),
                block_size, 0, context.stream()>>>(
                source_rows, summed_scratch, power_row_count, coefficient_stride,
                static_cast<int>(features), radial_count, max_angular, destination);
            check_cuda(cudaGetLastError(), "CUDA SOAP mu1nu1 power kernel launch failed");
        } else {
            soap_power_kernel<<<static_cast<unsigned>(
                (power_row_count * features + block_size - 1) / block_size),
                block_size, 0, context.stream()>>>(
                source_rows, power_row_count, coefficient_stride,
                static_cast<int>(features), species_count, coefficient_types,
                radial_count, max_angular, compression, destination);
            check_cuda(cudaGetLastError(), "CUDA SOAP power kernel launch failed");
        }
    };
    if (inner) {
        if (batch.structures() > 0) {
            soap_average_coefficients_kernel<<<static_cast<unsigned>((batch.structures() + block_size - 1) / block_size),
                block_size, 0, context.stream()>>>(
                batch.offsets(), batch.structures(), coefficient_stride, coefficients,
                structure_coefficients);
            check_cuda(cudaGetLastError(), "CUDA SOAP coefficient average launch failed");
            launch_power(structure_coefficients, batch.structures(), output);
        }
    } else {
        if (batch.atoms() > 0) {
            launch_power(coefficients, batch.atoms(), outer ? atom_power : output);
        }
        if (outer && batch.structures() > 0) {
            soap_average_power_kernel<<<static_cast<unsigned>((batch.structures() + block_size - 1) / block_size),
                block_size, 0, context.stream()>>>(
                batch.offsets(), batch.structures(), static_cast<int>(features), atom_power, output);
            check_cuda(cudaGetLastError(), "CUDA SOAP outer power average launch failed");
        }
    }
    const auto values = download_output_with_gil_release(
        context, output_size, rows, features);
    py::dict result;
    result["values"] = values;
    result["level"] = inner || outer ? "structure" : "atom";
    if (!inner && !outer) result["row_offsets"] = i64_array(
        host_row_offsets(host_batch));
    result["labels"] = labels_option(options, "SOAP", features);
    result["metadata"] = metadata(options, "SOAP");
    return result;
}

} // namespace
