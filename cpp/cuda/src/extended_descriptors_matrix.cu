#include "extended_descriptors_matrix.cuh"
#include "mdescriptor/detail/math3.hpp"

namespace {

// Deterministic two-level Ewald with a single scratch base.  The host
// computes every setup constant bit-identically to the serial reference
// (same math3 helpers, same expressions, same enumeration order); three
// kernels then run the heavy work in the same per-element orders.
//
// Scratch layout (doubles), all addressed through `offsets`:
//   [metadata sections, fixed size]
//   g vectors (stride 3, all structures, then)
//   g factors (stride 1, all structures, then)
//   per structure: phases (4*g*count), real (count^2), reciprocal (count^2)
//
// Offsets per structure (kEwaldMetaOffsets = 7):
//   0,1 reserved; 2 = g vector base; 3 = g factor base; 4 = phase base;
//   5 = real base; 6 = reciprocal base.

constexpr int kEwaldMetaCounts = 2;     // count, g_count
constexpr int kEwaldMetaOffsets = 7;
constexpr int kEwaldScalarStride = 32;  // alpha, volume, r_cut, g_cut,
                                        // inverse(9), norms(3), nmax(3),
                                        // reciprocal(9), pad(4)

struct EwaldPlan {
    std::vector<double> centers_prefix, g_prefix, pairs_prefix;
    std::vector<double> counts, offsets, scalars, g_vectors, g_factors;
    I64 total_G = 0, total_centers = 0, total_pairs = 0;
    std::size_t meta_doubles = 0;
    std::size_t scratch_doubles = 0;
};

EwaldPlan plan_ewald(
    const detail::StructureBatchView& host_batch,
    double accuracy,
    double weight,
    double r_cut_option,
    double g_cut_option,
    double a_option) {
    using mdescriptor::detail::Mat3;
    using mdescriptor::detail::Vec3;
    EwaldPlan plan;
    const I64 structures = host_batch.structures;
    plan.centers_prefix.push_back(0);
    plan.g_prefix.push_back(0);
    plan.pairs_prefix.push_back(0);
    for (I64 structure = 0; structure < structures; ++structure) {
        const I64 begin = host_batch.offsets[structure];
        const int count = static_cast<int>(host_batch.offsets[structure + 1] - begin);
        plan.centers_prefix.push_back(plan.centers_prefix.back() + count);
        if (count == 0) {
            plan.counts.push_back(0);
            plan.counts.push_back(0);
            plan.g_prefix.push_back(plan.g_prefix.back());
            plan.pairs_prefix.push_back(plan.pairs_prefix.back());
            continue;
        }
        const double* cell_values = host_batch.cells + structure * 9;
        Mat3 cell;
        for (int row_index = 0; row_index < 3; ++row_index) {
            for (int column = 0; column < 3; ++column) {
                cell.a[row_index][column] = cell_values[row_index * 3 + column];
            }
        }
        const Mat3 inverse_cell = mdescriptor::detail::inverse(cell);
        const double volume = std::abs(mdescriptor::detail::determinant(cell));
        const double alpha = a_option > 0.0
            ? a_option
            : std::pow(static_cast<double>(count) * weight / (volume * volume),
                1.0 / 6.0) * std::sqrt(kPi);
        double r_cut = r_cut_option;
        double g_cut = g_cut_option;
        if (r_cut <= 0.0 && g_cut <= 0.0) {
            const double factor = std::sqrt(-std::log(accuracy));
            r_cut = factor / alpha;
            g_cut = 2.0 * alpha * factor;
        }
        const double inverse_norms[3] = {
            std::sqrt(inverse_cell.a[0][0] * inverse_cell.a[0][0] + inverse_cell.a[1][0] * inverse_cell.a[1][0] + inverse_cell.a[2][0] * inverse_cell.a[2][0]),
            std::sqrt(inverse_cell.a[0][1] * inverse_cell.a[0][1] + inverse_cell.a[1][1] * inverse_cell.a[1][1] + inverse_cell.a[2][1] * inverse_cell.a[2][1]),
            std::sqrt(inverse_cell.a[0][2] * inverse_cell.a[0][2] + inverse_cell.a[1][2] * inverse_cell.a[1][2] + inverse_cell.a[2][2] * inverse_cell.a[2][2]),
        };
        const double real_nmax[3] = {
            r_cut * inverse_norms[0] + 0.01,
            r_cut * inverse_norms[1] + 0.01,
            r_cut * inverse_norms[2] + 0.01,
        };
        const Mat3 reciprocal = {
            {{2.0 * kPi * inverse_cell.a[0][0], 2.0 * kPi * inverse_cell.a[1][0], 2.0 * kPi * inverse_cell.a[2][0]},
             {2.0 * kPi * inverse_cell.a[0][1], 2.0 * kPi * inverse_cell.a[1][1], 2.0 * kPi * inverse_cell.a[2][1]},
             {2.0 * kPi * inverse_cell.a[0][2], 2.0 * kPi * inverse_cell.a[1][2], 2.0 * kPi * inverse_cell.a[2][2]}}};
        const int g_bounds[3] = {
            static_cast<int>(std::ceil(g_cut / mdescriptor::detail::norm(mdescriptor::detail::row(reciprocal, 0)))) + 1,
            static_cast<int>(std::ceil(g_cut / mdescriptor::detail::norm(mdescriptor::detail::row(reciprocal, 1)))) + 1,
            static_cast<int>(std::ceil(g_cut / mdescriptor::detail::norm(mdescriptor::detail::row(reciprocal, 2)))) + 1,
        };
        I64 g_count = 0;
        for (int gi = -g_bounds[0]; gi <= g_bounds[0]; ++gi) {
            for (int gj = -g_bounds[1]; gj <= g_bounds[1]; ++gj) {
                for (int gk = -g_bounds[2]; gk <= g_bounds[2]; ++gk) {
                    const Vec3 vector = gi * mdescriptor::detail::row(reciprocal, 0)
                        + gj * mdescriptor::detail::row(reciprocal, 1)
                        + gk * mdescriptor::detail::row(reciprocal, 2);
                    const double length2 = mdescriptor::detail::norm2(vector);
                    if (length2 > 1e-24 && length2 <= g_cut * g_cut) {
                        ++g_count;
                    }
                }
            }
        }
        plan.counts.push_back(count);
        plan.counts.push_back(g_count);
        plan.g_prefix.push_back(plan.g_prefix.back() + g_count);
        plan.pairs_prefix.push_back(plan.pairs_prefix.back()
            + static_cast<I64>(count) * count);
        plan.total_G += g_count;
        plan.total_centers += count;
        plan.total_pairs += static_cast<I64>(count) * count;
        for (int gi = -g_bounds[0]; gi <= g_bounds[0]; ++gi) {
            for (int gj = -g_bounds[1]; gj <= g_bounds[1]; ++gj) {
                for (int gk = -g_bounds[2]; gk <= g_bounds[2]; ++gk) {
                    const Vec3 vector = gi * mdescriptor::detail::row(reciprocal, 0)
                        + gj * mdescriptor::detail::row(reciprocal, 1)
                        + gk * mdescriptor::detail::row(reciprocal, 2);
                    const double length2 = mdescriptor::detail::norm2(vector);
                    if (length2 > 1e-24 && length2 <= g_cut * g_cut) {
                        plan.g_vectors.push_back(vector.x);
                        plan.g_vectors.push_back(vector.y);
                        plan.g_vectors.push_back(vector.z);
                    }
                }
            }
        }
        for (int gi = -g_bounds[0]; gi <= g_bounds[0]; ++gi) {
            for (int gj = -g_bounds[1]; gj <= g_bounds[1]; ++gj) {
                for (int gk = -g_bounds[2]; gk <= g_bounds[2]; ++gk) {
                    const Vec3 vector = gi * mdescriptor::detail::row(reciprocal, 0)
                        + gj * mdescriptor::detail::row(reciprocal, 1)
                        + gk * mdescriptor::detail::row(reciprocal, 2);
                    const double length2 = mdescriptor::detail::norm2(vector);
                    if (length2 > 1e-24 && length2 <= g_cut * g_cut) {
                        plan.g_factors.push_back(
                            std::exp(-length2 / (4.0 * alpha * alpha)) / length2);
                    }
                }
            }
        }
        plan.scalars.push_back(alpha);
        plan.scalars.push_back(volume);
        plan.scalars.push_back(r_cut);
        plan.scalars.push_back(g_cut);
        for (int row_index = 0; row_index < 3; ++row_index) {
            for (int column = 0; column < 3; ++column) {
                plan.scalars.push_back(inverse_cell.a[row_index][column]);
            }
        }
        for (int axis = 0; axis < 3; ++axis) plan.scalars.push_back(inverse_norms[axis]);
        for (int axis = 0; axis < 3; ++axis) plan.scalars.push_back(real_nmax[axis]);
        for (int row_index = 0; row_index < 3; ++row_index) {
            for (int column = 0; column < 3; ++column) {
                plan.scalars.push_back(reciprocal.a[row_index][column]);
            }
        }
        for (int pad = 0; pad < kEwaldScalarStride - 28; ++pad) plan.scalars.push_back(0.0);
    }
    // Absolute layout: metadata, g vectors, g factors, then per-structure
    // variable areas (phase, real, reciprocal) in structure order.
    plan.meta_doubles = 3 * (static_cast<std::size_t>(structures) + 1)
        + 2 * (static_cast<std::size_t>(structures) + 1)
        + kEwaldMetaOffsets * static_cast<std::size_t>(structures)
        + kEwaldScalarStride * static_cast<std::size_t>(structures);
    const I64 gvec_base = static_cast<I64>(plan.meta_doubles);
    I64 gvec_cursor = gvec_base;
    I64 gfactor_cursor = gvec_base + static_cast<I64>(plan.g_vectors.size());
    I64 variable_cursor = gfactor_cursor + static_cast<I64>(plan.g_factors.size());
    for (I64 structure = 0; structure < structures; ++structure) {
        const I64 count = static_cast<I64>(plan.counts[structure * kEwaldMetaCounts]);
        const I64 g_count = static_cast<I64>(plan.counts[structure * kEwaldMetaCounts + 1]);
        plan.offsets.push_back(0.0);
        plan.offsets.push_back(0.0);
        plan.offsets.push_back(static_cast<double>(gvec_cursor));
        plan.offsets.push_back(static_cast<double>(gfactor_cursor));
        plan.offsets.push_back(static_cast<double>(variable_cursor));
        variable_cursor += 4 * g_count * count;
        plan.offsets.push_back(static_cast<double>(variable_cursor));
        variable_cursor += count * count;
        plan.offsets.push_back(static_cast<double>(variable_cursor));
        variable_cursor += count * count;
        gvec_cursor += 3 * g_count;
        gfactor_cursor += g_count;
    }
    plan.scratch_doubles = static_cast<std::size_t>(variable_cursor);
    return plan;
}

// Phase A: per (structure, g) atom phases -- the reciprocal sum's only
// transcendental work, parallel over G-vectors.
__global__ void ewald_phase_kernel(
    const double* g_prefix,
    const double* centers_prefix,
    const double* counts,
    const double* offsets,
    double* scratch,
    const double* positions,
    I64 structures) {
    const I64 thread = static_cast<I64>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (thread >= static_cast<I64>(g_prefix[structures])) return;
    I64 lower = 0;
    I64 upper = structures;
    while (lower + 1 < upper) {
        const I64 middle = lower + (upper - lower) / 2;
        if (g_prefix[middle] <= static_cast<double>(thread)) lower = middle;
        else upper = middle;
    }
    const I64 structure = lower;
    const I64 g = thread - static_cast<I64>(g_prefix[structure]);
    const int count = static_cast<int>(counts[structure * kEwaldMetaCounts]);
    const I64 center_begin = static_cast<I64>(centers_prefix[structure]);
    const double* g_vector = scratch
        + static_cast<I64>(offsets[structure * kEwaldMetaOffsets + 2]) + 3 * g;
    double* phase_plane = scratch
        + static_cast<I64>(offsets[structure * kEwaldMetaOffsets + 4]) + 4 * g * count;
    for (int atom = 0; atom < count; ++atom) {
        const double phase = g_vector[0] * positions[(center_begin + atom) * 3 + 0]
            + g_vector[1] * positions[(center_begin + atom) * 3 + 1]
            + g_vector[2] * positions[(center_begin + atom) * 3 + 2];
        const double sine = sin(phase);
        const double cosine = cos(phase);
        phase_plane[atom * 4 + 0] = sine;
        phase_plane[atom * 4 + 1] = cosine;
        phase_plane[atom * 4 + 2] = sine + cosine;
        phase_plane[atom * 4 + 3] = sine - cosine;
    }
}

// Phase B: real-space column per center.  The image enumeration and erfc
// accumulation replicate the serial reference; the wrapped position of each
// target is recomputed inline from the device coordinates.
__global__ void ewald_real_kernel(
    const double* centers_prefix,
    const double* counts,
    const double* offsets,
    const double* scalars,
    double* scratch,
    const I32* numbers,
    const double* positions,
    const double* cells_device,
    I64 structures) {
    const I64 thread = static_cast<I64>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (thread >= static_cast<I64>(centers_prefix[structures])) return;
    I64 lower = 0;
    I64 upper = structures;
    while (lower + 1 < upper) {
        const I64 middle = lower + (upper - lower) / 2;
        if (centers_prefix[middle] <= static_cast<double>(thread)) lower = middle;
        else upper = middle;
    }
    const I64 structure = lower;
    const int center = static_cast<int>(thread - static_cast<I64>(centers_prefix[structure]));
    const int count = static_cast<int>(counts[structure * kEwaldMetaCounts]);
    const double* sc = scalars + structure * kEwaldScalarStride;
    const double alpha = sc[0];
    const double r_cut = sc[2];
    const double* inverse = sc + 4;
    const double* real_nmax = sc + 16;
    const I64 center_begin = static_cast<I64>(centers_prefix[structure]);
    const double cx = positions[(center_begin + center) * 3 + 0];
    const double cy = positions[(center_begin + center) * 3 + 1];
    const double cz = positions[(center_begin + center) * 3 + 2];
    const double fractional_x = cx * inverse[0] + cy * inverse[3] + cz * inverse[6];
    const double fractional_y = cx * inverse[1] + cy * inverse[4] + cz * inverse[7];
    const double fractional_z = cx * inverse[2] + cy * inverse[5] + cz * inverse[8];
    const int minima[3] = {
        static_cast<int>(floor(fractional_x - real_nmax[0])),
        static_cast<int>(floor(fractional_y - real_nmax[1])),
        static_cast<int>(floor(fractional_z - real_nmax[2])),
    };
    const int maxima[3] = {
        static_cast<int>(ceil(fractional_x + real_nmax[0])),
        static_cast<int>(ceil(fractional_y + real_nmax[1])),
        static_cast<int>(ceil(fractional_z + real_nmax[2])),
    };
    const double* cell = cells_device + structure * 9;
    double* real_column = scratch
        + static_cast<I64>(offsets[structure * kEwaldMetaOffsets + 5]);
    for (int target = 0; target < count; ++target) {
        const double twx = positions[(center_begin + target) * 3 + 0];
        const double twy = positions[(center_begin + target) * 3 + 1];
        const double twz = positions[(center_begin + target) * 3 + 2];
        const double wfx = twx * inverse[0] + twy * inverse[3] + twz * inverse[6];
        const double wfy = twx * inverse[1] + twy * inverse[4] + twz * inverse[7];
        const double wfz = twx * inverse[2] + twy * inverse[5] + twz * inverse[8];
        const double wx = (wfx - floor(wfx)) * cell[0] + (wfy - floor(wfy)) * cell[3] + (wfz - floor(wfz)) * cell[6];
        const double wy = (wfx - floor(wfx)) * cell[1] + (wfy - floor(wfy)) * cell[4] + (wfz - floor(wfz)) * cell[7];
        const double wz = (wfx - floor(wfx)) * cell[2] + (wfy - floor(wfy)) * cell[5] + (wfz - floor(wfz)) * cell[8];
        double real = 0.0;
        for (int si = minima[0]; si < maxima[0]; ++si) {
            for (int sj = minima[1]; sj < maxima[1]; ++sj) {
                for (int sk = minima[2]; sk < maxima[2]; ++sk) {
                    const double dx = wx - cx + si * cell[0] + sj * cell[3] + sk * cell[6];
                    const double dy = wy - cy + si * cell[1] + sj * cell[4] + sk * cell[7];
                    const double dz = wz - cz + si * cell[2] + sj * cell[5] + sk * cell[8];
                    const double distance2 = dx * dx + dy * dy + dz * dz;
                    if (distance2 > 1e-16 && distance2 <= r_cut * r_cut) {
                        real += erfc(alpha * sqrt(distance2)) / sqrt(distance2);
                    }
                }
            }
        }
        const double charge_t = static_cast<double>(numbers[center_begin + target]);
        const double charge_c = static_cast<double>(numbers[center_begin + center]);
        real_column[target * count + center] = real * charge_t * charge_c;
    }
}

// Phase C: reciprocal element plus assembly, one thread per (i, j).  The
// G-vector loop ascends exactly like the serial reference, and the assembly
// applies the diagonal self term and the neutralizing-background correction
// with the original expressions.
__global__ void ewald_recip_assemble_kernel(
    const double* pairs_prefix,
    const double* centers_prefix,
    const double* counts,
    const double* offsets,
    const double* scalars,
    const double* scratch,
    const I32* numbers,
    I64 structures,
    int n_atoms_max,
    I64 matrix_stride,
    double* matrices) {
    const I64 thread = static_cast<I64>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (thread >= static_cast<I64>(pairs_prefix[structures])) return;
    I64 lower = 0;
    I64 upper = structures;
    while (lower + 1 < upper) {
        const I64 middle = lower + (upper - lower) / 2;
        if (pairs_prefix[middle] <= static_cast<double>(thread)) lower = middle;
        else upper = middle;
    }
    const I64 structure = lower;
    const I64 within = thread - static_cast<I64>(pairs_prefix[structure]);
    const int count = static_cast<int>(counts[structure * kEwaldMetaCounts]);
    const int g_count = static_cast<int>(counts[structure * kEwaldMetaCounts + 1]);
    const int i = static_cast<int>(within / count);
    const int j = static_cast<int>(within - static_cast<I64>(i) * count);
    const double* sc = scalars + structure * kEwaldScalarStride;
    const double alpha = sc[0];
    const double volume = sc[1];
    const double* g_factors = scratch
        + static_cast<I64>(offsets[structure * kEwaldMetaOffsets + 3]);
    const double* phases_s = scratch
        + static_cast<I64>(offsets[structure * kEwaldMetaOffsets + 4]);
    const double* real_s = scratch
        + static_cast<I64>(offsets[structure * kEwaldMetaOffsets + 5]);
    const I64 center_begin = static_cast<I64>(centers_prefix[structure]);
    const double zi = static_cast<double>(numbers[center_begin + i]);
    const double zj = static_cast<double>(numbers[center_begin + j]);
    const double inverse_sqrt_two = 1.0 / sqrt(2.0);
    double reciprocal_value = 0.0;
    for (int g = 0; g < g_count; ++g) {
        const double* plane = phases_s + 4 * g * count;
        const double sine_i = plane[i * 4 + 0];
        const double cosine_i = plane[i * 4 + 1];
        const double sum_j = plane[j * 4 + 2];
        const double difference_j = plane[j * 4 + 3];
        reciprocal_value += (cosine_i * sum_j + sine_i * difference_j)
            * g_factors[g] * inverse_sqrt_two;
    }
    const double reciprocal_scale = 4.0 * kPi / volume * sqrt(2.0);
    double value = real_s[static_cast<I64>(i) * count + j]
        + reciprocal_value * reciprocal_scale * zi * zj;
    if (i == j) {
        value = 0.5 * value - alpha / sqrt(kPi) * zi * zi;
    }
    value += -kPi / (2.0 * volume * alpha * alpha) * 2.0 * zi * zj;
    if (i == j) {
        value -= -kPi / (2.0 * volume * alpha * alpha) * zi * zj;
    }
    matrices[static_cast<I64>(structure) * matrix_stride
        + static_cast<I64>(i) * n_atoms_max + j] = value;
}

} // namespace

py::dict compute_matrix_descriptor(
    CudaExecutionContext& context,
    DeviceBatch& batch,
    const detail::StructureBatchView& host_batch,
    int kind,
    const std::string& name,
    const py::dict& options) {
    int n_atoms_max = option(options, "n_atoms_max", 0);
    const std::string permutation_name = option(options, "permutation", std::string("sorted_l2"));
    const int permutation = permutation_name == "none" ? kMatrixPermutationNone
        : permutation_name == "sorted_l2" ? kMatrixPermutationSortedL2 : kMatrixPermutationEigenspectrum;
    if (permutation_name != "none" && permutation_name != "sorted_l2"
        && permutation_name != "eigenspectrum") {
        throw std::invalid_argument("invalid CUDA matrix permutation");
    }
    if (n_atoms_max <= 0) {
        for (I64 structure = 0; structure < host_batch.structures; ++structure) {
            n_atoms_max = std::max(
                n_atoms_max,
                static_cast<int>(host_batch.offsets[structure + 1] - host_batch.offsets[structure]));
        }
    }
    if (n_atoms_max <= 0 && host_batch.atoms == 0) {
        py::dict result;
        result["values"] = download_output_with_gil_release(
            context, 0, host_batch.structures, 0);
        result["level"] = "structure";
        result["labels"] = labels_option(options, name, 0);
        result["metadata"] = metadata(options, name);
        return result;
    }
    if (n_atoms_max <= 0 || n_atoms_max > 256) {
        throw std::invalid_argument(
            "CUDA matrix descriptors require 1 <= n_atoms_max <= 256");
    }
    for (I64 structure = 0; structure < host_batch.structures; ++structure) {
        const I64 count = host_batch.offsets[structure + 1] - host_batch.offsets[structure];
        if (count > n_atoms_max) {
            throw std::invalid_argument("structure exceeds n_atoms_max");
        }
    }
    const I64 columns = permutation == kMatrixPermutationEigenspectrum
        ? n_atoms_max : static_cast<I64>(n_atoms_max) * n_atoms_max;
    const std::size_t output_size = static_cast<std::size_t>(host_batch.structures)
        * static_cast<std::size_t>(columns);
    const std::size_t matrix_stride = static_cast<std::size_t>(n_atoms_max) * n_atoms_max
        + (kind == kMatrixKindEwald ? static_cast<std::size_t>(3 * n_atoms_max) : 0);
    const std::size_t matrix_size = static_cast<std::size_t>(host_batch.structures)
        * matrix_stride;
    double* output = context.output_buffer(output_size);
    double* matrices = nullptr;
    const double exponent = option(options, "exponent", 2.4);
    const double accuracy = option(options, "accuracy", 1e-5);
    const double weight = option(options, "w", 1.0);
    const double r_cut = option(options, "r_cut", 0.0);
    const double g_cut = option(options, "g_cut", 0.0);
    const double split = option(options, "a", 0.0);
    constexpr unsigned block_size = 256;
    if (kind == kMatrixKindEwald && host_batch.structures > 0) {
        // Deterministic two-level Ewald: the host computes every setup
        // constant bit-identically to the serial reference, then three
        // kernels run the transcendental and accumulation work in the same
        // per-element orders.  Matrices and all Ewald scratch share one
        // workspace allocation (the pool frees on growth, which would dangle
        // earlier pointers).
        const EwaldPlan plan = plan_ewald(
            host_batch, accuracy, weight, r_cut, g_cut, split);
        const std::size_t structures = static_cast<std::size_t>(host_batch.structures);
        // The workspace pool frees its old allocation when it grows.  Keep
        // the matrix and Ewald scratch slices from the same request so no
        // pointer is left referring to the freed allocation.
        matrices = static_cast<double*>(context.workspace_buffer(
            (matrix_size + plan.scratch_doubles) * sizeof(double)));
        double* ewald = matrices + matrix_size;
        double* d_centers_prefix = ewald;
        double* d_g_prefix = d_centers_prefix + (structures + 1);
        double* d_pairs_prefix = d_g_prefix + (structures + 1);
        double* d_counts = d_pairs_prefix + (structures + 1);
        double* d_offsets = d_counts + 2 * (structures + 1);
        double* d_scalars = d_offsets + kEwaldMetaOffsets * structures;
        double* d_g_vectors = d_scalars + kEwaldScalarStride * structures;
        double* d_g_factors = d_g_vectors + plan.g_vectors.size();
        check_cuda(cudaMemcpyAsync(d_centers_prefix, plan.centers_prefix.data(),
            plan.centers_prefix.size() * sizeof(double), cudaMemcpyHostToDevice,
            context.stream()), "could not upload Ewald centers prefix");
        check_cuda(cudaMemcpyAsync(d_g_prefix, plan.g_prefix.data(),
            plan.g_prefix.size() * sizeof(double), cudaMemcpyHostToDevice,
            context.stream()), "could not upload Ewald g prefix");
        check_cuda(cudaMemcpyAsync(d_pairs_prefix, plan.pairs_prefix.data(),
            plan.pairs_prefix.size() * sizeof(double), cudaMemcpyHostToDevice,
            context.stream()), "could not upload Ewald pairs prefix");
        check_cuda(cudaMemcpyAsync(d_counts, plan.counts.data(),
            plan.counts.size() * sizeof(double), cudaMemcpyHostToDevice,
            context.stream()), "could not upload Ewald counts");
        check_cuda(cudaMemcpyAsync(d_offsets, plan.offsets.data(),
            plan.offsets.size() * sizeof(double), cudaMemcpyHostToDevice,
            context.stream()), "could not upload Ewald offsets");
        check_cuda(cudaMemcpyAsync(d_scalars, plan.scalars.data(),
            plan.scalars.size() * sizeof(double), cudaMemcpyHostToDevice,
            context.stream()), "could not upload Ewald scalars");
        check_cuda(cudaMemcpyAsync(d_g_vectors, plan.g_vectors.data(),
            plan.g_vectors.size() * sizeof(double), cudaMemcpyHostToDevice,
            context.stream()), "could not upload Ewald g vectors");
        check_cuda(cudaMemcpyAsync(d_g_factors, plan.g_factors.data(),
            plan.g_factors.size() * sizeof(double), cudaMemcpyHostToDevice,
            context.stream()), "could not upload Ewald g factors");
        constexpr unsigned ewald_block = 128;
        // Empty structures contribute no G vectors / centers / pairs; a zero
        // grid would be rejected by the runtime, so each launch is guarded.
        if (plan.total_G > 0) {
            ewald_phase_kernel<<<static_cast<unsigned>((static_cast<std::size_t>(plan.total_G) + ewald_block - 1) / ewald_block),
                ewald_block, 0, context.stream()>>>(
                d_g_prefix, d_centers_prefix, d_counts, d_offsets, ewald,
                batch.positions(), host_batch.structures);
            check_cuda(cudaGetLastError(), "CUDA Ewald phase launch failed");
        }
        if (plan.total_centers > 0) {
            ewald_real_kernel<<<static_cast<unsigned>((static_cast<std::size_t>(plan.total_centers) + ewald_block - 1) / ewald_block),
                ewald_block, 0, context.stream()>>>(
                d_centers_prefix, d_counts, d_offsets, d_scalars, ewald,
                batch.numbers(), batch.positions(), batch.cells(), host_batch.structures);
            check_cuda(cudaGetLastError(), "CUDA Ewald real launch failed");
        }
        if (plan.total_pairs > 0) {
            ewald_recip_assemble_kernel<<<static_cast<unsigned>((static_cast<std::size_t>(plan.total_pairs) + ewald_block - 1) / ewald_block),
                ewald_block, 0, context.stream()>>>(
                d_pairs_prefix, d_centers_prefix, d_counts, d_offsets, d_scalars,
                ewald, batch.numbers(), host_batch.structures, n_atoms_max,
                static_cast<I64>(matrix_stride), matrices);
            check_cuda(cudaGetLastError(), "CUDA Ewald recip assemble launch failed");
        }
        const auto post_blocks = static_cast<unsigned>((host_batch.structures + 63) / 64);
        matrix_post_kernel<<<post_blocks, 64, 0, context.stream()>>>(
            host_batch.structures, n_atoms_max, permutation, kind, batch.offsets(),
            matrices, output);
        check_cuda(cudaGetLastError(), "CUDA Ewald post launch failed");
    } else if (host_batch.structures > 0) {
        matrices = static_cast<double*>(context.workspace_buffer(
            matrix_size * sizeof(double)));
        // Coulomb and sine elements are independent: one thread per
        // (structure, i, j) fills the raw matrix, then a per-structure pass
        // applies the permutation / eigenspectrum tail.
        const auto fill_blocks = static_cast<unsigned>(
            (static_cast<std::size_t>(host_batch.structures) * n_atoms_max * n_atoms_max
                + block_size - 1) / block_size);
        if (fill_blocks > 0) {
            matrix_fill_pairs_kernel<<<fill_blocks, block_size, 0, context.stream()>>>(
                batch.numbers(), batch.positions(), batch.cells(), batch.offsets(),
                host_batch.structures, n_atoms_max, kind, exponent, matrices);
            check_cuda(cudaGetLastError(), "CUDA matrix fill launch failed");
            const auto post_blocks = static_cast<unsigned>(
                (host_batch.structures + 63) / 64);
            matrix_post_kernel<<<post_blocks, 64, 0, context.stream()>>>(
                host_batch.structures, n_atoms_max, permutation, kind, batch.offsets(),
                matrices, output);
            check_cuda(cudaGetLastError(), "CUDA matrix post launch failed");
        }
    }
    const auto values = download_output_with_gil_release(
        context, output_size, host_batch.structures, columns);
    py::dict result;
    result["values"] = values;
    result["level"] = "structure";
    result["labels"] = labels_option(options, name, columns);
    result["metadata"] = metadata(options, name);
    return result;
}

py::dict compute_extended_matrix(
    CudaExecutionContext& context,
    DeviceBatch& batch,
    DeviceNeighborGraph& graph,
    const detail::StructureBatchView& host_batch,
    const std::string& name,
    const py::dict& options,
    RotationalPlanCache* rotational_plan) {
    (void)graph;
    (void)rotational_plan;
    const int kind = name == "CoulombMatrix" ? kMatrixKindCoulomb
        : name == "SineMatrix" ? kMatrixKindSine : kMatrixKindEwald;
    return compute_matrix_descriptor(context, batch, host_batch, kind, name, options);
}

} // namespace mdescriptor::cuda
