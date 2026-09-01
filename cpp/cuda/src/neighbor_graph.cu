#include "mdescriptor/cuda/neighbor_graph.hpp"

#include <cuda_runtime.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/gather.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace mdescriptor::cuda {
namespace {

template <typename Value>
void release(Value*& pointer) noexcept {
    if (pointer != nullptr) {
        (void)cudaFree(pointer);
        pointer = nullptr;
    }
}

template <typename Value>
void ensure_and_upload(
    Value** destination,
    std::size_t* capacity,
    const Value* source,
    std::size_t count,
    cudaStream_t stream) {
    if (count > *capacity) {
        if (*destination != nullptr) {
            if (cudaFree(*destination) != cudaSuccess) {
                throw std::runtime_error("could not release the CUDA neighbor graph");
            }
            *destination = nullptr;
        }
        const auto allocation = cudaMalloc(
            reinterpret_cast<void**>(destination), count * sizeof(Value));
        if (allocation == cudaErrorMemoryAllocation) {
            throw CudaOutOfMemory("could not allocate the CUDA neighbor graph");
        }
        if (allocation != cudaSuccess) {
            throw std::runtime_error("could not allocate the CUDA neighbor graph");
        }
        *capacity = count;
    }
    if (count != 0 && cudaMemcpyAsync(
        *destination, source, count * sizeof(Value), cudaMemcpyHostToDevice, stream) != cudaSuccess) {
        throw std::runtime_error("could not upload the CUDA neighbor graph");
    }
}

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

bool inverse_row_major3(const double* matrix, double* inverse) {
    const double determinant =
        matrix[0] * (matrix[4] * matrix[8] - matrix[5] * matrix[7])
        - matrix[1] * (matrix[3] * matrix[8] - matrix[5] * matrix[6])
        + matrix[2] * (matrix[3] * matrix[7] - matrix[4] * matrix[6]);
    if (!std::isfinite(determinant) || std::abs(determinant) <= 1.0e-12) {
        return false;
    }
    const double inverse_determinant = 1.0 / determinant;
    inverse[0] = (matrix[4] * matrix[8] - matrix[5] * matrix[7]) * inverse_determinant;
    inverse[1] = (matrix[2] * matrix[7] - matrix[1] * matrix[8]) * inverse_determinant;
    inverse[2] = (matrix[1] * matrix[5] - matrix[2] * matrix[4]) * inverse_determinant;
    inverse[3] = (matrix[5] * matrix[6] - matrix[3] * matrix[8]) * inverse_determinant;
    inverse[4] = (matrix[0] * matrix[8] - matrix[2] * matrix[6]) * inverse_determinant;
    inverse[5] = (matrix[2] * matrix[3] - matrix[0] * matrix[5]) * inverse_determinant;
    inverse[6] = (matrix[3] * matrix[7] - matrix[4] * matrix[6]) * inverse_determinant;
    inverse[7] = (matrix[1] * matrix[6] - matrix[0] * matrix[7]) * inverse_determinant;
    inverse[8] = (matrix[0] * matrix[4] - matrix[1] * matrix[3]) * inverse_determinant;
    return true;
}

std::array<int, 3> nep_cell_dimensions(
    const double* source_cell,
    double cutoff,
    std::int64_t atom_count,
    double* reference_cell,
    double* reference_inverse) {
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
            reference_cell[row * 3 + column] = source_cell[column * 3 + row];
        }
    }
    if (!inverse_row_major3(reference_cell, reference_inverse)) {
        throw std::invalid_argument("cannot build a CUDA NEP cell list for a singular cell");
    }
    std::array<int, 3> dimensions{1, 1, 1};
    for (int axis = 0; axis < 3; ++axis) {
        const double x = reference_inverse[axis * 3 + 0];
        const double y = reference_inverse[axis * 3 + 1];
        const double z = reference_inverse[axis * 3 + 2];
        const double reciprocal_norm = std::sqrt(x * x + y * y + z * z);
        const double raw = 1.0 / (cutoff * reciprocal_norm);
        if (!std::isfinite(raw) || raw > static_cast<double>(std::numeric_limits<int>::max())) {
            throw std::invalid_argument("CUDA NEP cell-list dimensions are too large");
        }
        dimensions[static_cast<std::size_t>(axis)] = std::max(
            1, static_cast<int>(std::floor(raw)));
    }
    const std::int64_t capacity = std::max<std::int64_t>(1, atom_count);
    while (static_cast<std::int64_t>(dimensions[0]) * dimensions[1] * dimensions[2]
        > capacity) {
        int axis = 0;
        if (dimensions[1] > dimensions[axis]) axis = 1;
        if (dimensions[2] > dimensions[axis]) axis = 2;
        if (dimensions[axis] <= 1) {
            throw std::invalid_argument("CUDA NEP cell-list dimensions exceed atom capacity");
        }
        --dimensions[axis];
    }
    return dimensions;
}

__device__ __forceinline__ int wrap_index(int value, int extent) {
    if (value < 0) return value + extent;
    if (value >= extent) return value - extent;
    return value;
}

__device__ __forceinline__ bool duplicate_pbc_offset(int offset, int extent) {
    if (extent == 1) return offset != 0;
    if (extent == 2) return offset > 0;
    return false;
}

__device__ __forceinline__ double fractional_component(
    const double* inverse, int axis, double x, double y, double z) {
    return inverse[axis * 3 + 0] * x
        + inverse[axis * 3 + 1] * y
        + inverse[axis * 3 + 2] * z;
}

__device__ __forceinline__ int cell_coordinate(
    const double* inverse, int axis, int dimension, double x, double y, double z) {
    double value = fractional_component(inverse, axis, x, y, z);
    value -= floor(value);
    int coordinate = static_cast<int>(floor(value * dimension));
    if (coordinate < 0) coordinate = 0;
    if (coordinate >= dimension) coordinate = dimension - 1;
    return coordinate;
}

__device__ __forceinline__ void minimum_image_delta(
    const double* cell,
    const double* inverse,
    double dx,
    double dy,
    double dz,
    float& x12,
    float& y12,
    float& z12) {
    double sx = fractional_component(inverse, 0, dx, dy, dz);
    double sy = fractional_component(inverse, 1, dx, dy, dz);
    double sz = fractional_component(inverse, 2, dx, dy, dz);
    sx -= nearbyint(sx);
    sy -= nearbyint(sy);
    sz -= nearbyint(sz);
    x12 = static_cast<float>(cell[0] * sx + cell[1] * sy + cell[2] * sz);
    y12 = static_cast<float>(cell[3] * sx + cell[4] * sy + cell[5] * sz);
    z12 = static_cast<float>(cell[6] * sx + cell[7] * sy + cell[8] * sz);
}

__global__ void assign_nep_cells_kernel(
    int atoms,
    int position_stride,
    const double* positions,
    const std::int32_t* atom_to_structure,
    const std::int32_t* structure_cell_offsets,
    const std::int32_t* structure_cell_dimensions,
    const double* reference_inverses,
    std::int32_t* atom_cells,
    std::int32_t* cell_counts) {
    const std::int64_t atom = static_cast<std::int64_t>(blockIdx.x)
        * blockDim.x + threadIdx.x;
    if (atom >= atoms) return;
    const int structure = atom_to_structure[atom];
    const int* dimensions = structure_cell_dimensions + structure * 4;
    const double* inverse = reference_inverses + structure * 9;
    const double x = positions[atom];
    const double y = positions[position_stride + atom];
    const double z = positions[2 * position_stride + atom];
    const int ix = cell_coordinate(inverse, 0, dimensions[0], x, y, z);
    const int iy = cell_coordinate(inverse, 1, dimensions[1], x, y, z);
    const int iz = cell_coordinate(inverse, 2, dimensions[2], x, y, z);
    const int cell = structure_cell_offsets[structure]
        + ix + dimensions[0] * (iy + dimensions[1] * iz);
    atom_cells[atom] = cell;
    atomicAdd(cell_counts + cell, 1);
}

__global__ void make_nep_lane_major_order_kernel(
    int atoms, int block_count, std::int32_t* order_keys) {
    const std::int64_t atom = static_cast<std::int64_t>(blockIdx.x)
        * blockDim.x + threadIdx.x;
    if (atom >= atoms) return;
    order_keys[atom] = static_cast<std::int32_t>(
        (atom % blockDim.x) * block_count + atom / blockDim.x);
}

template <bool Fill>
__global__ void build_nep_neighbors_kernel(
    int atoms,
    int position_stride,
    const double* positions,
    const std::int32_t* atom_to_structure,
    const std::int32_t* structure_pbc,
    const std::int32_t* structure_cell_offsets,
    const std::int32_t* structure_cell_dimensions,
    const double* reference_cells,
    const double* reference_inverses,
    const std::int32_t* atom_cells,
    const std::int32_t* cell_offsets,
    const std::int32_t* cell_atoms,
    double cutoff2,
    std::int64_t graph_stride,
    const std::int64_t* graph_offsets,
    std::int32_t* neighbor_counts,
    std::int32_t* graph_atoms,
    double* graph_displacements,
    double* graph_distance2) {
    const std::int64_t atom = static_cast<std::int64_t>(blockIdx.x)
        * blockDim.x + threadIdx.x;
    if (atom >= atoms) return;
    const int structure = atom_to_structure[atom];
    const std::int32_t* pbc = structure_pbc + structure * 3;
    const bool periodic = pbc[0] == 1 && pbc[1] == 1 && pbc[2] == 1;
    const int* dimensions = structure_cell_dimensions + structure * 4;
    const double* cell = reference_cells + structure * 9;
    const double* inverse = reference_inverses + structure * 9;
    const int local_cell = atom_cells[atom] - structure_cell_offsets[structure];
    const int nx = dimensions[0];
    const int ny = dimensions[1];
    const int nz = dimensions[2];
    const int ix = local_cell % nx;
    const int iy = (local_cell / nx) % ny;
    const int iz = local_cell / (nx * ny);
    const double xi = positions[atom];
    const double yi = positions[position_stride + atom];
    const double zi = positions[2 * position_stride + atom];
    std::int32_t count = 0;

    for (int dz = -1; dz <= 1; ++dz) {
        if (periodic) {
            if (duplicate_pbc_offset(dz, nz)) continue;
        } else if (iz + dz < 0 || iz + dz >= nz) {
            continue;
        }
        const int cz = periodic ? wrap_index(iz + dz, nz) : iz + dz;
        for (int dy = -1; dy <= 1; ++dy) {
            if (periodic) {
                if (duplicate_pbc_offset(dy, ny)) continue;
            } else if (iy + dy < 0 || iy + dy >= ny) {
                continue;
            }
            const int cy = periodic ? wrap_index(iy + dy, ny) : iy + dy;
            for (int dx = -1; dx <= 1; ++dx) {
                if (periodic) {
                    if (duplicate_pbc_offset(dx, nx)) continue;
                } else if (ix + dx < 0 || ix + dx >= nx) {
                    continue;
                }
                const int cx = periodic ? wrap_index(ix + dx, nx) : ix + dx;
                const int neighbor_cell = structure_cell_offsets[structure]
                    + cx + nx * (cy + ny * cz);
                for (int offset = cell_offsets[neighbor_cell];
                     offset < cell_offsets[neighbor_cell + 1]; ++offset) {
                    const std::int32_t neighbor = cell_atoms[offset];
                    if (neighbor == atom) continue;
                    float dxij = 0.0f;
                    float dyij = 0.0f;
                    float dzij = 0.0f;
                    // The NEPAdapters neighbor builder uses center-minus-
                    // neighbor for the cutoff, but its descriptor core
                    // recomputes the stored vector as neighbor-minus-center.
                    // Store that latter orientation so the descriptor's
                    // float32 distance follows the reference path.
                    const double raw_dx = positions[neighbor] - xi;
                    const double raw_dy = positions[position_stride + neighbor] - yi;
                    const double raw_dz = positions[2 * position_stride + neighbor] - zi;
                    if (periodic) {
                        minimum_image_delta(
                            cell, inverse, raw_dx, raw_dy, raw_dz,
                            dxij, dyij, dzij);
                    } else {
                        dxij = static_cast<float>(raw_dx);
                        dyij = static_cast<float>(raw_dy);
                        dzij = static_cast<float>(raw_dz);
                    }
                    const float distance2 = dxij * dxij + dyij * dyij + dzij * dzij;
                    if (static_cast<double>(distance2) >= cutoff2) continue;
                    if constexpr (Fill) {
                        const std::int64_t edge = graph_offsets != nullptr
                            ? graph_offsets[atom] + count
                            : atom * graph_stride + count;
                        graph_atoms[edge] = neighbor;
                        graph_displacements[edge * 3 + 0] = static_cast<double>(dxij);
                        graph_displacements[edge * 3 + 1] = static_cast<double>(dyij);
                        graph_displacements[edge * 3 + 2] = static_cast<double>(dzij);
                        graph_distance2[edge] = static_cast<double>(distance2);
                    }
                    ++count;
                }
            }
        }
    }
    neighbor_counts[atom] = count;
}

// GPUMD's original NEP CUDA path enumerates every source atom and the
// periodic image translations on the device.  Keeping this as a two-pass CSR
// builder lets the same kernel handle fully periodic and isolated structures
// in one batch without materializing an expanded host-side atom array.
template <bool Fill>
__global__ void build_nep_image_neighbors_kernel(
    int atoms,
    int position_stride,
    const double* positions,
    const std::int32_t* atom_to_structure,
    const std::int64_t* structure_offsets,
    const double* source_cells,
    const std::int32_t* pbc,
    const std::int32_t* image_counts,
    const double* image_cells,
    const double* image_inverses,
    float cutoff2,
    const std::int64_t* graph_offsets,
    std::int32_t* neighbor_counts,
    std::int32_t* graph_atoms,
    double* graph_displacements,
    double* graph_distance2,
    std::int32_t* overflow) {
    const std::int64_t atom = static_cast<std::int64_t>(blockIdx.x)
        * blockDim.x + threadIdx.x;
    if (atom >= atoms) return;

    const std::int32_t structure = atom_to_structure[atom];
    const std::int32_t* structure_pbc = pbc + structure * 3;
    const bool periodic = structure_pbc[0] == 1
        && structure_pbc[1] == 1 && structure_pbc[2] == 1;
    const std::int32_t* counts = image_counts + structure * 3;
    const double* source_cell = source_cells + structure * 9;
    const double* image_cell = image_cells + structure * 9;
    const double* image_inverse = image_inverses + structure * 9;
    const std::int64_t begin = structure_offsets[structure];
    const std::int64_t end = structure_offsets[structure + 1];
    const double xi = positions[atom];
    const double yi = positions[position_stride + atom];
    const double zi = positions[2 * position_stride + atom];
    int count = 0;
    constexpr int kMaxCount = 0x7fffffff;

    for (std::int64_t neighbor = begin; neighbor < end; ++neighbor) {
        const double base_x = positions[neighbor] - xi;
        const double base_y = positions[position_stride + neighbor] - yi;
        const double base_z = positions[2 * position_stride + neighbor] - zi;
        const int image_x_count = periodic ? counts[0] : 1;
        const int image_y_count = periodic ? counts[1] : 1;
        const int image_z_count = periodic ? counts[2] : 1;
        for (int ia = 0; ia < image_x_count; ++ia) {
            for (int ib = 0; ib < image_y_count; ++ib) {
                for (int ic = 0; ic < image_z_count; ++ic) {
                    if (neighbor == atom && ia == 0 && ib == 0 && ic == 0) {
                        continue;
                    }
                    float dx = static_cast<float>(base_x);
                    float dy = static_cast<float>(base_y);
                    float dz = static_cast<float>(base_z);
                    if (periodic) {
                        const double raw_x = base_x
                            + static_cast<double>(ia) * source_cell[0]
                            + static_cast<double>(ib) * source_cell[3]
                            + static_cast<double>(ic) * source_cell[6];
                        const double raw_y = base_y
                            + static_cast<double>(ia) * source_cell[1]
                            + static_cast<double>(ib) * source_cell[4]
                            + static_cast<double>(ic) * source_cell[7];
                        const double raw_z = base_z
                            + static_cast<double>(ia) * source_cell[2]
                            + static_cast<double>(ib) * source_cell[5]
                            + static_cast<double>(ic) * source_cell[8];
                        minimum_image_delta(
                            image_cell, image_inverse, raw_x, raw_y, raw_z,
                            dx, dy, dz);
                    }
                    const float distance2 = dx * dx + dy * dy + dz * dz;
                    if (distance2 >= cutoff2) {
                        continue;
                    }
                    if (count == kMaxCount) {
                        atomicExch(overflow, 1);
                        continue;
                    }
                    if constexpr (Fill) {
                        const std::int64_t edge = graph_offsets[atom] + count;
                        graph_atoms[edge] = static_cast<std::int32_t>(neighbor);
                        graph_displacements[edge * 3 + 0] = static_cast<double>(dx);
                        graph_displacements[edge * 3 + 1] = static_cast<double>(dy);
                        graph_displacements[edge * 3 + 2] = static_cast<double>(dz);
                        graph_distance2[edge] = static_cast<double>(distance2);
                    }
                    ++count;
                }
            }
        }
    }
    neighbor_counts[atom] = static_cast<std::int32_t>(count);
}

// DPA4 and DPA4C consume the same normalized, image-expanded graph as the
// native model-backed CPU path.  The two kernels below keep the per-atom
// structure lookup and periodic wrapping on the device; the host only uploads
// the compact inverse-cell/image-bound plan.
__global__ void assign_dpa_structures_kernel(
    std::int64_t atoms,
    std::int32_t structures,
    const std::int64_t* offsets,
    std::int32_t* atom_to_structure) {
    const std::int64_t atom = static_cast<std::int64_t>(blockIdx.x)
        * blockDim.x + threadIdx.x;
    if (atom >= atoms || structures <= 0) return;
    std::int32_t low = 0;
    std::int32_t high = structures;
    while (low + 1 < high) {
        const std::int32_t middle = low + (high - low) / 2;
        if (offsets[middle] <= atom) {
            low = middle;
        } else {
            high = middle;
        }
    }
    atom_to_structure[atom] = low;
}

__global__ void normalize_dpa_positions_kernel(
    std::int64_t atoms,
    const double* positions,
    const double* cells,
    const std::int32_t* pbc,
    const std::int32_t* atom_to_structure,
    const double* inverses,
    double* normalized) {
    const std::int64_t atom = static_cast<std::int64_t>(blockIdx.x)
        * blockDim.x + threadIdx.x;
    if (atom >= atoms) return;
    const std::int32_t structure = atom_to_structure[atom];
    const double* source = positions + atom * 3;
    const std::int32_t* structure_pbc = pbc + structure * 3;
    const bool periodic = structure_pbc[0] == 1
        && structure_pbc[1] == 1 && structure_pbc[2] == 1;
    if (!periodic) {
        normalized[atom * 3 + 0] = source[0];
        normalized[atom * 3 + 1] = source[1];
        normalized[atom * 3 + 2] = source[2];
        return;
    }
    const double* inverse = inverses + structure * 9;
    const double fx = source[0] * inverse[0]
        + source[1] * inverse[3] + source[2] * inverse[6];
    const double fy = source[0] * inverse[1]
        + source[1] * inverse[4] + source[2] * inverse[7];
    const double fz = source[0] * inverse[2]
        + source[1] * inverse[5] + source[2] * inverse[8];
    const double wx = fx - floor(fx);
    const double wy = fy - floor(fy);
    const double wz = fz - floor(fz);
    const double* cell = cells + structure * 9;
    normalized[atom * 3 + 0] = wx * cell[0] + wy * cell[3] + wz * cell[6];
    normalized[atom * 3 + 1] = wx * cell[1] + wy * cell[4] + wz * cell[7];
    normalized[atom * 3 + 2] = wx * cell[2] + wy * cell[5] + wz * cell[8];
}

template <bool Fill>
__device__ std::int64_t enumerate_dpa_candidates(
    std::int64_t center,
    std::int32_t thread,
    const double* normalized,
    const std::int64_t* structure_offsets,
    const double* cells,
    const std::int32_t* pbc,
    const std::int32_t* atom_to_structure,
    const std::int32_t* image_bounds,
    double cutoff2,
    const std::int64_t* graph_offsets,
    std::int32_t* graph_atoms,
    std::int32_t* graph_shifts,
    double* graph_displacements,
    double* graph_distance2,
    std::int64_t output_base,
    std::int32_t* overflow,
    bool round_edge_endpoints) {
    constexpr std::int64_t kMaxCount = 0x7fffffffLL;
    const std::int32_t structure = atom_to_structure[center];
    const std::int32_t* structure_pbc = pbc + structure * 3;
    const bool periodic = structure_pbc[0] == 1
        && structure_pbc[1] == 1 && structure_pbc[2] == 1;
    const std::int32_t* bounds = image_bounds + structure * 3;
    const double* cell = cells + structure * 9;
    const std::int64_t begin = structure_offsets[structure];
    const std::int64_t end = structure_offsets[structure + 1];
    const double cx = normalized[center * 3 + 0];
    const double cy = normalized[center * 3 + 1];
    const double cz = normalized[center * 3 + 2];
    const std::int64_t lower_x = periodic ? -static_cast<std::int64_t>(bounds[0]) : 0;
    const std::int64_t upper_x = periodic ? static_cast<std::int64_t>(bounds[0]) : 0;
    const std::int64_t lower_y = periodic ? -static_cast<std::int64_t>(bounds[1]) : 0;
    const std::int64_t upper_y = periodic ? static_cast<std::int64_t>(bounds[1]) : 0;
    const std::int64_t lower_z = periodic ? -static_cast<std::int64_t>(bounds[2]) : 0;
    const std::int64_t upper_z = periodic ? static_cast<std::int64_t>(bounds[2]) : 0;
    std::int64_t count = 0;
    for (std::int64_t sx = lower_x; sx <= upper_x; ++sx) {
        for (std::int64_t sy = lower_y; sy <= upper_y; ++sy) {
            for (std::int64_t sz = lower_z; sz <= upper_z; ++sz) {
                double tx = 0.0;
                double ty = 0.0;
                double tz = 0.0;
                if (periodic) {
                    tx = static_cast<double>(sx) * cell[0]
                        + static_cast<double>(sy) * cell[3]
                        + static_cast<double>(sz) * cell[6];
                    ty = static_cast<double>(sx) * cell[1]
                        + static_cast<double>(sy) * cell[4]
                        + static_cast<double>(sz) * cell[7];
                    tz = static_cast<double>(sx) * cell[2]
                        + static_cast<double>(sy) * cell[5]
                        + static_cast<double>(sz) * cell[8];
                }
                for (std::int64_t neighbor = begin + thread;
                     neighbor < end; neighbor += blockDim.x) {
                    if (neighbor == center && sx == 0 && sy == 0 && sz == 0) {
                        continue;
                    }
                    const double raw_dx = normalized[neighbor * 3 + 0] + tx - cx;
                    const double raw_dy = normalized[neighbor * 3 + 1] + ty - cy;
                    const double raw_dz = normalized[neighbor * 3 + 2] + tz - cz;
                    const double distance2 = raw_dx * raw_dx
                        + raw_dy * raw_dy + raw_dz * raw_dz;
                    if (distance2 > cutoff2) continue;
                    // The native DPA4 path builds edge vectors by converting
                    // the translated source and center endpoints to fp32
                    // independently before subtracting them.  DPA4C instead
                    // consumes the common graph's double displacement.  Keep
                    // the distinction explicit so periodic ULPs reproduce the
                    // corresponding CPU ABI without changing graph ordering.
                    double dx = raw_dx;
                    double dy = raw_dy;
                    double dz = raw_dz;
                    if (round_edge_endpoints) {
                        const float source_x = static_cast<float>(
                            normalized[neighbor * 3 + 0] + tx);
                        const float source_y = static_cast<float>(
                            normalized[neighbor * 3 + 1] + ty);
                        const float source_z = static_cast<float>(
                            normalized[neighbor * 3 + 2] + tz);
                        const float center_x = static_cast<float>(
                            normalized[center * 3 + 0]);
                        const float center_y = static_cast<float>(
                            normalized[center * 3 + 1]);
                        const float center_z = static_cast<float>(
                            normalized[center * 3 + 2]);
                        dx = static_cast<double>(source_x - center_x);
                        dy = static_cast<double>(source_y - center_y);
                        dz = static_cast<double>(source_z - center_z);
                    }
                    if (count >= kMaxCount) {
                        atomicExch(overflow, 1);
                        continue;
                    }
                    if constexpr (Fill) {
                        const std::int64_t edge = output_base + count;
                        graph_atoms[edge] = static_cast<std::int32_t>(neighbor);
                        graph_shifts[edge * 3 + 0] = static_cast<std::int32_t>(sx);
                        graph_shifts[edge * 3 + 1] = static_cast<std::int32_t>(sy);
                        graph_shifts[edge * 3 + 2] = static_cast<std::int32_t>(sz);
                        graph_displacements[edge * 3 + 0] = dx;
                        graph_displacements[edge * 3 + 1] = dy;
                        graph_displacements[edge * 3 + 2] = dz;
                        graph_distance2[edge] = distance2;
                    }
                    ++count;
                }
            }
        }
    }
    return count;
}

template <typename Value>
__device__ void swap_dpa_value(Value& left, Value& right) {
    const Value saved = left;
    left = right;
    right = saved;
}

__device__ bool dpa_edge_precedes(
    std::int64_t left,
    std::int64_t right,
    const std::int32_t* graph_atoms,
    const std::int32_t* graph_shifts,
    const double* graph_distance2,
    bool tie_break_shifts) {
    if (graph_distance2[left] < graph_distance2[right]) return true;
    if (graph_distance2[left] > graph_distance2[right]) return false;
    if (tie_break_shifts) {
        const std::int64_t left_norm =
            static_cast<std::int64_t>(graph_shifts[left * 3 + 0])
                * graph_shifts[left * 3 + 0]
            + static_cast<std::int64_t>(graph_shifts[left * 3 + 1])
                * graph_shifts[left * 3 + 1]
            + static_cast<std::int64_t>(graph_shifts[left * 3 + 2])
                * graph_shifts[left * 3 + 2];
        const std::int64_t right_norm =
            static_cast<std::int64_t>(graph_shifts[right * 3 + 0])
                * graph_shifts[right * 3 + 0]
            + static_cast<std::int64_t>(graph_shifts[right * 3 + 1])
                * graph_shifts[right * 3 + 1]
            + static_cast<std::int64_t>(graph_shifts[right * 3 + 2])
                * graph_shifts[right * 3 + 2];
        if (left_norm != right_norm) return left_norm < right_norm;
        for (int axis = 0; axis < 3; ++axis) {
            const std::int32_t left_shift = graph_shifts[left * 3 + axis];
            const std::int32_t right_shift = graph_shifts[right * 3 + axis];
            if (left_shift != right_shift) return left_shift < right_shift;
        }
    }
    return graph_atoms[left] < graph_atoms[right];
}

__global__ void count_dpa_neighbors_kernel(
    std::int64_t atoms,
    const double* normalized,
    const std::int64_t* structure_offsets,
    const double* cells,
    const std::int32_t* pbc,
    const std::int32_t* atom_to_structure,
    const std::int32_t* image_bounds,
    double cutoff2,
    std::int32_t* neighbor_counts,
    std::int32_t* overflow,
    bool round_edge_endpoints) {
    const std::int64_t center = static_cast<std::int64_t>(blockIdx.x);
    if (center >= atoms) return;
    __shared__ std::int64_t counts[128];
    const std::int32_t thread = static_cast<std::int32_t>(threadIdx.x);
    counts[thread] = enumerate_dpa_candidates<false>(
        center, thread, normalized, structure_offsets, cells, pbc,
        atom_to_structure, image_bounds, cutoff2, nullptr, nullptr, nullptr,
        nullptr, nullptr, 0, overflow, round_edge_endpoints);
    __syncthreads();
    for (std::int32_t step = 64; step > 0; step >>= 1) {
        if (thread < step) counts[thread] += counts[thread + step];
        __syncthreads();
    }
    if (thread == 0) {
        constexpr std::int64_t kMaxCount = 0x7fffffffLL;
        if (counts[0] > kMaxCount) {
            atomicExch(overflow, 1);
            neighbor_counts[center] = static_cast<std::int32_t>(kMaxCount);
        } else {
            neighbor_counts[center] = static_cast<std::int32_t>(counts[0]);
        }
    }
}

__global__ void fill_dpa_neighbors_kernel(
    std::int64_t atoms,
    const double* normalized,
    const std::int64_t* structure_offsets,
    const double* cells,
    const std::int32_t* pbc,
    const std::int32_t* atom_to_structure,
    const std::int32_t* image_bounds,
    double cutoff2,
    const std::int64_t* graph_offsets,
    std::int32_t* graph_atoms,
    std::int32_t* graph_shifts,
    double* graph_displacements,
    double* graph_distance2,
    std::int32_t* overflow,
    bool round_edge_endpoints) {
    const std::int64_t center = static_cast<std::int64_t>(blockIdx.x);
    if (center >= atoms) return;
    __shared__ std::int64_t counts[128];
    const std::int32_t thread = static_cast<std::int32_t>(threadIdx.x);
    const std::int64_t local_count = enumerate_dpa_candidates<false>(
        center, thread, normalized, structure_offsets, cells, pbc,
        atom_to_structure, image_bounds, cutoff2, nullptr, nullptr, nullptr,
        nullptr, nullptr, 0, overflow, round_edge_endpoints);
    counts[thread] = local_count;
    __syncthreads();
    for (std::int32_t step = 1; step < 128; step <<= 1) {
        const std::int64_t add = thread >= step ? counts[thread - step] : 0;
        __syncthreads();
        if (thread >= step) counts[thread] += add;
        __syncthreads();
    }
    const std::int64_t prefix = counts[thread] - local_count;
    (void)enumerate_dpa_candidates<true>(
        center, thread, normalized, structure_offsets, cells, pbc,
        atom_to_structure, image_bounds, cutoff2, graph_offsets, graph_atoms,
        graph_shifts, graph_displacements, graph_distance2,
        graph_offsets[center] + prefix, overflow, round_edge_endpoints);
}

__global__ void sort_dpa_neighbors_kernel(
    std::int64_t atoms,
    const std::int64_t* graph_offsets,
    std::int32_t* graph_atoms,
    std::int32_t* graph_shifts,
    double* graph_displacements,
    double* graph_distance2,
    bool tie_break_shifts,
    std::int32_t* overflow) {
    const std::int64_t center = static_cast<std::int64_t>(blockIdx.x)
        * blockDim.x + threadIdx.x;
    if (center >= atoms) return;
    const std::int64_t begin = graph_offsets[center];
    const std::int64_t end = graph_offsets[center + 1];
    if (end - begin > 0x7fffffffLL) {
        atomicExch(overflow, 1);
        return;
    }
    for (std::int64_t position = begin + 1; position < end; ++position) {
        std::int64_t current = position;
        while (current > begin && dpa_edge_precedes(
            current, current - 1, graph_atoms, graph_shifts,
            graph_distance2, tie_break_shifts)) {
            swap_dpa_value(graph_atoms[current], graph_atoms[current - 1]);
            swap_dpa_value(graph_distance2[current], graph_distance2[current - 1]);
            for (int axis = 0; axis < 3; ++axis) {
                swap_dpa_value(
                    graph_shifts[current * 3 + axis],
                    graph_shifts[(current - 1) * 3 + axis]);
                swap_dpa_value(
                    graph_displacements[current * 3 + axis],
                    graph_displacements[(current - 1) * 3 + axis]);
            }
            --current;
        }
    }
}

} // namespace

DeviceNeighborGraph::~DeviceNeighborGraph() noexcept {
    clear();
}

void DeviceNeighborGraph::upload(
    CudaExecutionContext& context,
    const std::vector<std::int64_t>& offsets,
    const std::vector<std::int32_t>& atoms,
    const std::vector<std::int32_t>& shifts,
    const std::vector<double>& displacements,
    const std::vector<double>& distance2) {
    if (offsets.empty() || shifts.size() != atoms.size() * 3
        || displacements.size() != atoms.size() * 3
        || distance2.size() != atoms.size()) {
        throw std::invalid_argument("invalid CUDA neighbor graph arrays");
    }
    if (cudaSetDevice(context.device()) != cudaSuccess) {
        throw std::runtime_error("could not select the CUDA device");
    }
    ensure_and_upload(&offsets_, &offsets_capacity_, offsets.data(), offsets.size(), context.stream());
    try {
        ensure_and_upload(&atoms_, &atoms_capacity_, atoms.data(), atoms.size(), context.stream());
        ensure_and_upload(&shifts_, &shifts_capacity_, shifts.data(), shifts.size(), context.stream());
        ensure_and_upload(
            &displacements_, &displacements_capacity_, displacements.data(),
            displacements.size(), context.stream());
        ensure_and_upload(
            &distance2_, &distance2_capacity_, distance2.data(),
            distance2.size(), context.stream());
    } catch (...) {
        clear();
        throw;
    }
    pairs_ = atoms.size();
    max_neighbors_ = 0;
    for (std::size_t center = 0; center + 1 < offsets.size(); ++center) {
        max_neighbors_ = std::max(
            max_neighbors_, offsets[center + 1] - offsets[center]);
    }
    slot_major_ = false;
    neighbor_stride_ = 0;
}

void DeviceNeighborGraph::build_dpa(
    CudaExecutionContext& context,
    DeviceBatch& batch,
    const detail::StructureBatchView& host_batch,
    double cutoff,
    bool tie_break_shifts,
    bool round_edge_endpoints) {
    if (host_batch.structures != batch.structures()
        || host_batch.atoms != batch.atoms()
        || host_batch.structures < 0 || host_batch.atoms < 0
        || host_batch.offsets == nullptr || host_batch.cells == nullptr
        || host_batch.pbc == nullptr) {
        throw std::invalid_argument("CUDA DPA host and device batches have different shapes");
    }
    if (!std::isfinite(cutoff) || cutoff <= 0.0) {
        throw std::invalid_argument("CUDA DPA graph cutoff must be positive");
    }
    if (host_batch.atoms > static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::max())
        || host_batch.structures > static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::max())) {
        throw CudaOutOfMemory("CUDA DPA batch is too large for graph index types");
    }
    pairs_ = 0;
    max_neighbors_ = 0;
    slot_major_ = false;
    neighbor_stride_ = 0;
    if (host_batch.atoms == 0) return;

    const std::size_t structure_count = static_cast<std::size_t>(host_batch.structures);
    const std::size_t atom_count = static_cast<std::size_t>(host_batch.atoms);
    if (atom_count > std::numeric_limits<std::size_t>::max() / 3U
        || structure_count > std::numeric_limits<std::size_t>::max() / 3U
        || structure_count > std::numeric_limits<std::size_t>::max() / 9U) {
        throw CudaOutOfMemory("CUDA DPA graph metadata is too large");
    }

    // This is launch planning only.  It mirrors build_neighbor_graph's image
    // range, but positions and all candidate pairs remain device-resident.
    std::vector<std::int32_t> image_bounds(structure_count * 3U, 0);
    std::vector<double> inverses(structure_count * 9U, 0.0);
    for (std::size_t structure = 0; structure < structure_count; ++structure) {
        const std::int32_t* structure_pbc = host_batch.pbc + structure * 3U;
        const bool periodic = structure_pbc[0] == 1
            && structure_pbc[1] == 1 && structure_pbc[2] == 1;
        const bool isolated = structure_pbc[0] == 0
            && structure_pbc[1] == 0 && structure_pbc[2] == 0;
        if (!periodic && !isolated) {
            throw std::invalid_argument(
                "CUDA DPA supports all-zero or all-one pbc per structure");
        }
        double* inverse = inverses.data() + structure * 9U;
        if (!periodic) {
            inverse[0] = 1.0;
            inverse[4] = 1.0;
            inverse[8] = 1.0;
            continue;
        }
        if (!inverse_row_major3(host_batch.cells + structure * 9U, inverse)) {
            throw std::invalid_argument("cannot build a CUDA DPA graph from a singular cell");
        }
        std::uint64_t image_product = 1;
        for (int axis = 0; axis < 3; ++axis) {
            const double x = inverse[axis];
            const double y = inverse[3 + axis];
            const double z = inverse[6 + axis];
            const double required = cutoff * std::sqrt(x * x + y * y + z * z) + 1.0;
            if (!std::isfinite(required) || required < 0.0
                || required > static_cast<double>(std::numeric_limits<std::int32_t>::max())) {
                throw std::invalid_argument("CUDA DPA periodic image range is too large");
            }
            const auto bound = static_cast<std::int64_t>(std::floor(required));
            image_bounds[structure * 3U + static_cast<std::size_t>(axis)] =
                static_cast<std::int32_t>(bound);
            const std::uint64_t extent =
                2U * static_cast<std::uint64_t>(bound) + 1U;
            if (image_product > static_cast<std::uint64_t>(
                    std::numeric_limits<std::int32_t>::max()) / extent) {
                throw CudaOutOfMemory("CUDA DPA periodic image count is too large");
            }
            image_product *= extent;
        }
    }

    if (cudaSetDevice(context.device()) != cudaSuccess) {
        throw std::runtime_error("could not select the CUDA device for DPA graph construction");
    }
    ensure_capacity(&atom_to_structure_, &atom_to_structure_capacity_, atom_count);
    ensure_capacity(&dpa_positions_, &dpa_positions_capacity_, atom_count * 3U);
    ensure_capacity(
        &dpa_image_bounds_, &dpa_image_bounds_capacity_, image_bounds.size());
    ensure_capacity(
        &dpa_reference_inverses_, &dpa_reference_inverses_capacity_, inverses.size());
    ensure_capacity(&neighbor_counts_, &neighbor_counts_capacity_, atom_count);
    ensure_capacity(&neighbor_overflow_, &neighbor_overflow_capacity_, 1U);
    ensure_capacity(&offsets_, &offsets_capacity_, atom_count + 1U);

    const cudaStream_t stream = context.stream();
    ensure_and_upload(
        &dpa_image_bounds_, &dpa_image_bounds_capacity_, image_bounds.data(),
        image_bounds.size(), stream);
    ensure_and_upload(
        &dpa_reference_inverses_, &dpa_reference_inverses_capacity_, inverses.data(),
        inverses.size(), stream);
    check_cuda(
        cudaMemsetAsync(neighbor_overflow_, 0, sizeof(std::int32_t), stream),
        "could not clear CUDA DPA graph overflow");

    constexpr unsigned int block_size = 128;
    const unsigned int atom_blocks = static_cast<unsigned int>(
        (atom_count + block_size - 1U) / block_size);
    assign_dpa_structures_kernel<<<atom_blocks, block_size, 0, stream>>>(
        static_cast<std::int64_t>(atom_count),
        static_cast<std::int32_t>(structure_count), batch.offsets(),
        atom_to_structure_);
    check_cuda(cudaGetLastError(), "CUDA DPA structure mapping failed");
    normalize_dpa_positions_kernel<<<atom_blocks, block_size, 0, stream>>>(
        static_cast<std::int64_t>(atom_count), batch.positions(), batch.cells(),
        batch.pbc(), atom_to_structure_, dpa_reference_inverses_, dpa_positions_);
    check_cuda(cudaGetLastError(), "CUDA DPA position normalization failed");

    // One CTA per center follows DeepMD-kit's CUDA neighbor-list layout: the
    // threads cooperatively scan that center's source atoms, then a device
    // prefix scan produces a compact CSR row.
    count_dpa_neighbors_kernel<<<static_cast<unsigned int>(atom_count), block_size, 0, stream>>>(
        static_cast<std::int64_t>(atom_count), dpa_positions_, batch.offsets(),
        batch.cells(), batch.pbc(), atom_to_structure_, dpa_image_bounds_,
        cutoff * cutoff, neighbor_counts_, neighbor_overflow_,
        round_edge_endpoints);
    check_cuda(cudaGetLastError(), "CUDA DPA graph neighbor count failed");

    check_cuda(
        cudaMemsetAsync(offsets_, 0, sizeof(std::int64_t), stream),
        "could not clear CUDA DPA graph offset zero");
    const auto execution_policy = thrust::cuda::par.on(stream);
    thrust::inclusive_scan(
        execution_policy, neighbor_counts_, neighbor_counts_ + atom_count, offsets_ + 1);
    const std::int32_t host_max_neighbors = thrust::reduce(
        execution_policy, neighbor_counts_, neighbor_counts_ + atom_count,
        std::int32_t{0}, thrust::maximum<std::int32_t>());
    std::int64_t host_pairs = 0;
    std::int32_t host_overflow = 0;
    check_cuda(
        cudaMemcpyAsync(
            &host_pairs, offsets_ + atom_count, sizeof(host_pairs),
            cudaMemcpyDeviceToHost, stream),
        "could not download CUDA DPA graph pair count");
    check_cuda(
        cudaMemcpyAsync(
            &host_overflow, neighbor_overflow_, sizeof(host_overflow),
            cudaMemcpyDeviceToHost, stream),
        "could not download CUDA DPA graph overflow");
    context.synchronize();
    if (host_overflow != 0) {
        throw CudaOutOfMemory("CUDA DPA graph neighbor count exceeds int32 capacity");
    }
    if (host_pairs < 0) {
        throw std::runtime_error("CUDA DPA graph pair count is negative");
    }
    const std::size_t pairs = static_cast<std::size_t>(host_pairs);
    if (pairs > std::numeric_limits<std::size_t>::max() / 3U) {
        throw CudaOutOfMemory("CUDA DPA graph storage is too large");
    }
    ensure_capacity(&atoms_, &atoms_capacity_, pairs);
    ensure_capacity(&shifts_, &shifts_capacity_, pairs * 3U);
    ensure_capacity(&displacements_, &displacements_capacity_, pairs * 3U);
    ensure_capacity(&distance2_, &distance2_capacity_, pairs);
    if (pairs > 0) {
        fill_dpa_neighbors_kernel<<<static_cast<unsigned int>(atom_count), block_size, 0, stream>>>(
            static_cast<std::int64_t>(atom_count), dpa_positions_, batch.offsets(),
            batch.cells(), batch.pbc(), atom_to_structure_, dpa_image_bounds_,
            cutoff * cutoff, offsets_, atoms_, shifts_, displacements_, distance2_,
            neighbor_overflow_, round_edge_endpoints);
        check_cuda(cudaGetLastError(), "CUDA DPA graph neighbor fill failed");
    }
    sort_dpa_neighbors_kernel<<<atom_blocks, block_size, 0, stream>>>(
        static_cast<std::int64_t>(atom_count), offsets_, atoms_, shifts_,
        displacements_, distance2_, tie_break_shifts, neighbor_overflow_);
    check_cuda(cudaGetLastError(), "CUDA DPA graph ordering failed");
    pairs_ = pairs;
    max_neighbors_ = static_cast<std::int64_t>(host_max_neighbors);
}

template <typename Value>
void DeviceNeighborGraph::ensure_capacity(
    Value** pointer,
    std::size_t* capacity,
    std::size_t count) {
    if (count <= *capacity) {
        return;
    }
    if (*pointer != nullptr) {
        check_cuda(cudaFree(*pointer), "could not release CUDA neighbor workspace");
        *pointer = nullptr;
    }
    *capacity = 0;
    if (count == 0) {
        return;
    }
    check_cuda(
        cudaMalloc(reinterpret_cast<void**>(pointer), count * sizeof(Value)),
        "could not allocate CUDA neighbor workspace");
    *capacity = count;
}

void DeviceNeighborGraph::build_nep(
    CudaExecutionContext& context,
    DeviceBatch& batch,
    const detail::StructureBatchView& host_batch,
    double cutoff) {
    if (!batch.expanded()) {
        build_nep_images(context, batch, host_batch, cutoff);
        return;
    }

    if (host_batch.structures != batch.structures() || host_batch.atoms != batch.atoms()) {
        throw std::invalid_argument("CUDA NEP host and device batches have different shapes");
    }
    if (host_batch.offsets == nullptr || host_batch.cells == nullptr
        || host_batch.pbc == nullptr) {
        throw std::invalid_argument("CUDA NEP expanded batch is missing metadata");
    }
    if (!std::isfinite(cutoff) || cutoff <= 0.0) {
        throw std::invalid_argument("CUDA NEP cell-list cutoff must be positive");
    }
    for (std::int64_t structure = 0; structure < host_batch.structures; ++structure) {
        const std::int32_t* pbc = host_batch.pbc + structure * 3;
        const bool periodic = pbc[0] == 1 && pbc[1] == 1 && pbc[2] == 1;
        const bool isolated = pbc[0] == 0 && pbc[1] == 0 && pbc[2] == 0;
        if (!periodic && !isolated) {
            throw std::invalid_argument(
                "CUDA NEP supports all-zero or all-one pbc per structure");
        }
    }
    if (cudaSetDevice(context.device()) != cudaSuccess) {
        throw std::runtime_error("could not select the CUDA device");
    }
    batch.ensure_positions_soa(context);

    const std::size_t structure_count = static_cast<std::size_t>(host_batch.structures);
    const std::size_t atom_count = static_cast<std::size_t>(host_batch.atoms);
    std::vector<std::int32_t> atom_to_structure(atom_count, 0);
    std::vector<std::int32_t> structure_cell_offsets(structure_count + 1, 0);
    std::vector<std::int32_t> structure_cell_dimensions(structure_count * 4, 0);
    std::vector<double> reference_cells(structure_count * 9, 0.0);
    std::vector<double> reference_inverses(structure_count * 9, 0.0);
    for (std::int64_t structure = 0; structure < host_batch.structures; ++structure) {
        const std::int32_t* pbc = host_batch.pbc + structure * 3;
        const bool periodic = pbc[0] == 1 && pbc[1] == 1 && pbc[2] == 1;
        const std::int64_t begin = host_batch.offsets[structure];
        const std::int64_t end = host_batch.offsets[structure + 1];
        const std::int64_t count = end - begin;
        if (count > static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::max())) {
            throw CudaOutOfMemory("CUDA NEP structure is too large for the cell-list index type");
        }
        std::array<int, 3> dimensions{1, 1, 1};
        if (periodic) {
            dimensions = nep_cell_dimensions(
                host_batch.cells + structure * 9, cutoff, count,
                reference_cells.data() + structure * 9,
                reference_inverses.data() + structure * 9);
        } else {
            for (int axis = 0; axis < 3; ++axis) {
                reference_cells[static_cast<std::size_t>(structure) * 9U
                    + static_cast<std::size_t>(axis) * 3U + static_cast<std::size_t>(axis)] = 1.0;
                reference_inverses[static_cast<std::size_t>(structure) * 9U
                    + static_cast<std::size_t>(axis) * 3U + static_cast<std::size_t>(axis)] = 1.0;
            }
        }
        const std::int64_t cell_count = static_cast<std::int64_t>(dimensions[0])
            * dimensions[1] * dimensions[2];
        if (cell_count > static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::max())
            || structure_cell_offsets[static_cast<std::size_t>(structure)]
                > std::numeric_limits<std::int32_t>::max() - cell_count) {
            throw CudaOutOfMemory("CUDA NEP cell-list index space is too large");
        }
        structure_cell_offsets[static_cast<std::size_t>(structure + 1)] =
            structure_cell_offsets[static_cast<std::size_t>(structure)]
            + static_cast<std::int32_t>(cell_count);
        structure_cell_dimensions[static_cast<std::size_t>(structure) * 4 + 0] = dimensions[0];
        structure_cell_dimensions[static_cast<std::size_t>(structure) * 4 + 1] = dimensions[1];
        structure_cell_dimensions[static_cast<std::size_t>(structure) * 4 + 2] = dimensions[2];
        structure_cell_dimensions[static_cast<std::size_t>(structure) * 4 + 3] =
            static_cast<std::int32_t>(cell_count);
        for (std::int64_t atom = begin; atom < end; ++atom) {
            atom_to_structure[static_cast<std::size_t>(atom)] =
                static_cast<std::int32_t>(structure);
        }
    }
    const std::size_t cell_count = static_cast<std::size_t>(structure_cell_offsets.back());
    if (atom_count > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
        throw CudaOutOfMemory("CUDA NEP batch is too large for the cell-list index type");
    }
    ensure_capacity(&atom_to_structure_, &atom_to_structure_capacity_, atom_count);
    ensure_capacity(&atom_cells_, &atom_cells_capacity_, atom_count);
    ensure_capacity(&neighbor_counts_, &neighbor_counts_capacity_, atom_count);
    ensure_capacity(&cell_atoms_, &cell_atoms_capacity_, atom_count);
    ensure_capacity(&cell_sort_keys_, &cell_sort_keys_capacity_, atom_count);
    ensure_capacity(
        &structure_cell_offsets_, &structure_cell_offsets_capacity_, structure_count + 1);
    ensure_capacity(
        &structure_cell_dims_, &structure_cell_dims_capacity_, structure_count * 4);
    ensure_capacity(&reference_cells_, &reference_cells_capacity_, structure_count * 9);
    ensure_capacity(
        &reference_cell_inverses_, &reference_cell_inverses_capacity_, structure_count * 9);
    ensure_capacity(&cell_counts_, &cell_counts_capacity_, cell_count);
    ensure_capacity(&cell_fill_, &cell_fill_capacity_, cell_count);
    ensure_capacity(&cell_offsets_, &cell_offsets_capacity_, cell_count + 1);
    ensure_capacity(&offsets_, &offsets_capacity_, atom_count + 1);

    const cudaStream_t stream = context.stream();
    ensure_and_upload(
        &atom_to_structure_, &atom_to_structure_capacity_, atom_to_structure.data(),
        atom_to_structure.size(), stream);
    ensure_and_upload(
        &structure_cell_offsets_, &structure_cell_offsets_capacity_, structure_cell_offsets.data(),
        structure_cell_offsets.size(), stream);
    ensure_and_upload(
        &structure_cell_dims_, &structure_cell_dims_capacity_, structure_cell_dimensions.data(),
        structure_cell_dimensions.size(), stream);
    ensure_and_upload(
        &reference_cells_, &reference_cells_capacity_, reference_cells.data(),
        reference_cells.size(), stream);
    ensure_and_upload(
        &reference_cell_inverses_, &reference_cell_inverses_capacity_, reference_inverses.data(),
        reference_inverses.size(), stream);
    check_cuda(
        cudaMemsetAsync(cell_counts_, 0, cell_count * sizeof(std::int32_t), stream),
        "could not clear CUDA NEP cell counts");
    check_cuda(
        cudaMemsetAsync(cell_fill_, 0, cell_count * sizeof(std::int32_t), stream),
        "could not clear CUDA NEP cell fill");

    // Match NEPAdapters' cell-list launch geometry.
    constexpr unsigned int block_size = 32;
    const unsigned int blocks = static_cast<unsigned int>(
        (atom_count + block_size - 1) / block_size);
    if (blocks > 0) {
        assign_nep_cells_kernel<<<blocks, block_size, 0, stream>>>(
            static_cast<int>(host_batch.atoms), static_cast<int>(batch.position_stride()),
            batch.positions_soa(),
            atom_to_structure_, structure_cell_offsets_,
            structure_cell_dims_, reference_cell_inverses_, atom_cells_, cell_counts_);
        check_cuda(cudaGetLastError(), "CUDA NEP cell assignment failed");
    }

    // Keep the cell and neighbor prefix scans on the device, as NEPAdapters
    // does.  The previous host round trips made every descriptor evaluation
    // wait twice and were especially visible for small expanded periodic
    // cells.
    const auto execution_policy = thrust::cuda::par.on(stream);
    thrust::exclusive_scan(
        execution_policy, cell_counts_, cell_counts_ + cell_count, cell_offsets_);
    const std::int32_t total_cell_atoms = static_cast<std::int32_t>(host_batch.atoms);
    check_cuda(
        cudaMemcpyAsync(
            cell_offsets_ + cell_count, &total_cell_atoms, sizeof(total_cell_atoms),
            cudaMemcpyHostToDevice, stream),
        "could not upload final CUDA NEP cell offset");
    thrust::sequence(
        execution_policy, cell_atoms_, cell_atoms_ + atom_count, std::int32_t{0});
    // Use the same 32-thread lane/block decomposition as the reference CUDA
    // builders before making the list cell-major.  The second stable sort
    // keeps that order for atoms sharing a cell while remaining deterministic
    // across descriptor evaluations.
    make_nep_lane_major_order_kernel<<<blocks, block_size, 0, stream>>>(
        static_cast<int>(atom_count), static_cast<int>(blocks), cell_sort_keys_);
    check_cuda(cudaGetLastError(), "CUDA NEP atom order construction failed");
    thrust::stable_sort_by_key(
        execution_policy,
        cell_sort_keys_, cell_sort_keys_ + atom_count,
        cell_atoms_);
    thrust::gather(
        execution_policy,
        cell_atoms_, cell_atoms_ + atom_count,
        atom_cells_, cell_sort_keys_);
    thrust::stable_sort_by_key(
        execution_policy,
        cell_sort_keys_, cell_sort_keys_ + atom_count,
        cell_atoms_);

    int max_neighborhood_cells = 1;
    for (std::size_t structure = 0; structure < structure_count; ++structure) {
        const std::int32_t* dimensions = structure_cell_dimensions.data() + structure * 4;
        const int cells_x = dimensions[0] == 1 ? 1 : dimensions[0] == 2 ? 2 : 3;
        const int cells_y = dimensions[1] == 1 ? 1 : dimensions[1] == 2 ? 2 : 3;
        const int cells_z = dimensions[2] == 1 ? 1 : dimensions[2] == 2 ? 2 : 3;
        max_neighborhood_cells = std::max(
            max_neighborhood_cells, cells_x * cells_y * cells_z);
    }
    const std::int32_t max_cell_occupancy = thrust::reduce(
        execution_policy, cell_counts_, cell_counts_ + cell_count,
        std::int32_t{0}, thrust::maximum<std::int32_t>());
    if (max_cell_occupancy <= 0) {
        throw std::runtime_error("CUDA NEP cell binning produced no occupied cells");
    }
    const std::size_t slot_capacity = static_cast<std::size_t>(max_cell_occupancy)
        * static_cast<std::size_t>(max_neighborhood_cells);
    constexpr std::size_t max_slot_entries = 16U * 1024U * 1024U;
    const bool use_slot_major = slot_capacity != 0
        && atom_count <= max_slot_entries / slot_capacity;

    // For the usual NEP workload, write the device-built list once in the
    // same slot-major form consumed by NEPAdapters.  The conservative cell
    // occupancy bound makes every write safe without a neighbor-count scan;
    // unusually large estimated workspaces retain the CSR fallback below.
    if (use_slot_major) {
        ensure_capacity(
            &atoms_, &atoms_capacity_, atom_count * slot_capacity);
        ensure_capacity(
            &displacements_, &displacements_capacity_, atom_count * slot_capacity * 3U);
        ensure_capacity(
            &distance2_, &distance2_capacity_, atom_count * slot_capacity);
        slot_major_ = true;
        neighbor_stride_ = static_cast<std::int64_t>(slot_capacity);
        if (blocks > 0) {
            build_nep_neighbors_kernel<true><<<blocks, block_size, 0, stream>>>(
                static_cast<int>(host_batch.atoms), static_cast<int>(batch.position_stride()),
                batch.positions_soa(),
                atom_to_structure_,
                batch.pbc(),
                structure_cell_offsets_, structure_cell_dims_, reference_cells_,
                reference_cell_inverses_, atom_cells_, cell_offsets_, cell_atoms_,
                cutoff * cutoff, neighbor_stride_, nullptr, neighbor_counts_,
                atoms_, displacements_, distance2_);
            check_cuda(cudaGetLastError(), "CUDA NEP slot-major neighbor fill failed");
        }
        pairs_ = atom_count * slot_capacity;
        return;
    }

    slot_major_ = false;
    neighbor_stride_ = 0;
    if (blocks > 0) {
        build_nep_neighbors_kernel<false><<<blocks, block_size, 0, stream>>>(
            static_cast<int>(host_batch.atoms), static_cast<int>(batch.position_stride()),
            batch.positions_soa(),
            atom_to_structure_, batch.pbc(), structure_cell_offsets_,
            structure_cell_dims_, reference_cells_, reference_cell_inverses_, atom_cells_,
            cell_offsets_, cell_atoms_, cutoff * cutoff, 0, offsets_, neighbor_counts_,
            nullptr, nullptr, nullptr);
        check_cuda(cudaGetLastError(), "CUDA NEP neighbor count failed");
    }

    // An inclusive scan into offsets_[1:] supplies both every CSR end offset
    // and the total pair count at offsets_[atoms].
    check_cuda(
        cudaMemsetAsync(offsets_, 0, sizeof(std::int64_t), stream),
        "could not clear CUDA NEP graph offset zero");
    thrust::inclusive_scan(
        execution_policy, neighbor_counts_, neighbor_counts_ + atom_count, offsets_ + 1);
    std::int64_t host_pairs = 0;
    check_cuda(
        cudaMemcpyAsync(
            &host_pairs, offsets_ + atom_count, sizeof(host_pairs),
            cudaMemcpyDeviceToHost, stream),
        "could not download CUDA NEP pair count");
    context.synchronize();
    if (host_pairs < 0) {
        throw std::runtime_error("CUDA NEP pair count is negative");
    }
    const std::size_t pairs = static_cast<std::size_t>(host_pairs);
    ensure_capacity(&atoms_, &atoms_capacity_, pairs);
    ensure_capacity(&displacements_, &displacements_capacity_, pairs * 3);
    ensure_capacity(&distance2_, &distance2_capacity_, pairs);
    if (blocks > 0 && pairs > 0) {
        build_nep_neighbors_kernel<true><<<blocks, block_size, 0, stream>>>(
            static_cast<int>(host_batch.atoms), static_cast<int>(batch.position_stride()),
            batch.positions_soa(),
            atom_to_structure_, batch.pbc(), structure_cell_offsets_,
            structure_cell_dims_, reference_cells_, reference_cell_inverses_, atom_cells_,
            cell_offsets_, cell_atoms_, cutoff * cutoff, 0, offsets_, neighbor_counts_,
            atoms_, displacements_, distance2_);
        check_cuda(cudaGetLastError(), "CUDA NEP neighbor fill failed");
    }
    pairs_ = pairs;
}

void DeviceNeighborGraph::build_nep_images(
    CudaExecutionContext& context,
    DeviceBatch& batch,
    const detail::StructureBatchView& host_batch,
    double cutoff) {
    if (host_batch.structures != batch.structures() || host_batch.atoms != batch.atoms()) {
        throw std::invalid_argument("CUDA NEP host and device batches have different shapes");
    }
    if (host_batch.structures < 0 || host_batch.atoms < 0
        || host_batch.offsets == nullptr || host_batch.cells == nullptr
        || host_batch.pbc == nullptr) {
        throw std::invalid_argument("invalid CUDA NEP batch metadata");
    }
    if (!std::isfinite(cutoff) || cutoff <= 0.0) {
        throw std::invalid_argument("CUDA NEP image cutoff must be positive");
    }
    if (host_batch.atoms > static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::max())
        || host_batch.structures > static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::max())) {
        throw CudaOutOfMemory("CUDA NEP batch is too large for image index types");
    }
    const double cutoff2_double = cutoff * cutoff;
    const float cutoff2 = static_cast<float>(cutoff2_double);
    if (!std::isfinite(cutoff2)) {
        throw std::invalid_argument("CUDA NEP image cutoff is too large");
    }

    const std::size_t structure_count = static_cast<std::size_t>(host_batch.structures);
    const std::size_t atom_count = static_cast<std::size_t>(host_batch.atoms);
    std::vector<std::int32_t> atom_to_structure(atom_count, 0);
    std::vector<std::int32_t> image_counts(structure_count * 3U, 1);
    std::vector<double> image_cells(structure_count * 9U, 0.0);
    std::vector<double> image_inverses(structure_count * 9U, 0.0);

    for (std::size_t structure = 0; structure < structure_count; ++structure) {
        const std::int32_t* structure_pbc = host_batch.pbc + structure * 3U;
        const bool periodic = structure_pbc[0] == 1
            && structure_pbc[1] == 1 && structure_pbc[2] == 1;
        const bool isolated = structure_pbc[0] == 0
            && structure_pbc[1] == 0 && structure_pbc[2] == 0;
        if (!periodic && !isolated) {
            throw std::invalid_argument(
                "CUDA NEP supports all-zero or all-one pbc per structure");
        }
        if (periodic) {
            double base_cell[9] = {};
            double base_inverse[9] = {};
            for (int row = 0; row < 3; ++row) {
                for (int column = 0; column < 3; ++column) {
                    base_cell[row * 3 + column] = host_batch.cells[
                        structure * 9U + column * 3 + row];
                }
            }
            if (!inverse_row_major3(base_cell, base_inverse)) {
                throw std::invalid_argument("cannot build a CUDA NEP image box from a singular cell");
            }
            std::int64_t image_product = 1;
            for (int axis = 0; axis < 3; ++axis) {
                const double x = base_inverse[axis * 3 + 0];
                const double y = base_inverse[axis * 3 + 1];
                const double z = base_inverse[axis * 3 + 2];
                const double reciprocal_norm = std::sqrt(x * x + y * y + z * z);
                const double required = 2.0 * cutoff * reciprocal_norm;
                if (!std::isfinite(required)
                    || required > static_cast<double>(std::numeric_limits<std::int32_t>::max())) {
                    throw std::invalid_argument("CUDA NEP periodic image range is too large");
                }
                const auto count = std::max<std::int32_t>(
                    1, static_cast<std::int32_t>(std::ceil(required - 1.0e-12)));
                image_counts[structure * 3U + static_cast<std::size_t>(axis)] = count;
                image_product *= count;
                if (image_product > std::numeric_limits<std::int32_t>::max()) {
                    throw CudaOutOfMemory("CUDA NEP periodic image count is too large");
                }
            }
            for (int row = 0; row < 3; ++row) {
                for (int column = 0; column < 3; ++column) {
                    const auto count = image_counts[
                        structure * 3U + static_cast<std::size_t>(column)];
                    image_cells[structure * 9U + row * 3 + column] =
                        base_cell[row * 3 + column] * static_cast<double>(count);
                }
            }
            if (!inverse_row_major3(
                image_cells.data() + structure * 9U,
                image_inverses.data() + structure * 9U)) {
                throw std::invalid_argument("cannot invert the CUDA NEP image cell");
            }
        } else {
            for (int axis = 0; axis < 3; ++axis) {
                image_cells[structure * 9U + axis * 3 + axis] = 1.0;
                image_inverses[structure * 9U + axis * 3 + axis] = 1.0;
            }
        }

        const std::int64_t begin = host_batch.offsets[structure];
        const std::int64_t end = host_batch.offsets[structure + 1];
        for (std::int64_t atom = begin; atom < end; ++atom) {
            atom_to_structure[static_cast<std::size_t>(atom)] =
                static_cast<std::int32_t>(structure);
        }
    }

    if (cudaSetDevice(context.device()) != cudaSuccess) {
        throw std::runtime_error("could not select the CUDA device");
    }
    batch.ensure_positions_soa(context);
    ensure_capacity(&atom_to_structure_, &atom_to_structure_capacity_, atom_count);
    ensure_capacity(&neighbor_counts_, &neighbor_counts_capacity_, atom_count);
    ensure_capacity(&image_counts_, &image_counts_capacity_, image_counts.size());
    ensure_capacity(&neighbor_overflow_, &neighbor_overflow_capacity_, 1);
    ensure_capacity(&reference_cells_, &reference_cells_capacity_, image_cells.size());
    ensure_capacity(&reference_cell_inverses_, &reference_cell_inverses_capacity_, image_inverses.size());
    ensure_capacity(&offsets_, &offsets_capacity_, atom_count + 1U);

    const cudaStream_t stream = context.stream();
    ensure_and_upload(
        &atom_to_structure_, &atom_to_structure_capacity_, atom_to_structure.data(),
        atom_to_structure.size(), stream);
    ensure_and_upload(
        &image_counts_, &image_counts_capacity_, image_counts.data(),
        image_counts.size(), stream);
    ensure_and_upload(
        &reference_cells_, &reference_cells_capacity_, image_cells.data(),
        image_cells.size(), stream);
    ensure_and_upload(
        &reference_cell_inverses_, &reference_cell_inverses_capacity_, image_inverses.data(),
        image_inverses.size(), stream);
    check_cuda(
        cudaMemsetAsync(neighbor_overflow_, 0, sizeof(std::int32_t), stream),
        "could not clear CUDA NEP image overflow");

    constexpr unsigned int block_size = 128;
    const auto blocks = static_cast<unsigned int>(
        (atom_count + block_size - 1U) / block_size);
    if (blocks > 0) {
        build_nep_image_neighbors_kernel<false><<<blocks, block_size, 0, stream>>>(
            static_cast<int>(atom_count), static_cast<int>(batch.position_stride()),
            batch.positions_soa(), atom_to_structure_, batch.offsets(), batch.cells(),
            batch.pbc(), image_counts_, reference_cells_, reference_cell_inverses_, cutoff2,
            nullptr, neighbor_counts_, nullptr, nullptr, nullptr, neighbor_overflow_);
        check_cuda(cudaGetLastError(), "CUDA NEP image neighbor count failed");
    }

    const auto execution_policy = thrust::cuda::par.on(stream);
    check_cuda(
        cudaMemsetAsync(offsets_, 0, sizeof(std::int64_t), stream),
        "could not clear CUDA NEP image graph offset zero");
    thrust::inclusive_scan(
        execution_policy, neighbor_counts_, neighbor_counts_ + atom_count, offsets_ + 1);
    std::int64_t host_pairs = 0;
    std::int32_t host_overflow = 0;
    check_cuda(
        cudaMemcpyAsync(
            &host_pairs, offsets_ + atom_count, sizeof(host_pairs),
            cudaMemcpyDeviceToHost, stream),
        "could not download CUDA NEP image pair count");
    check_cuda(
        cudaMemcpyAsync(
            &host_overflow, neighbor_overflow_, sizeof(host_overflow),
            cudaMemcpyDeviceToHost, stream),
        "could not download CUDA NEP image overflow");
    context.synchronize();
    if (host_overflow != 0) {
        throw CudaOutOfMemory("CUDA NEP image neighbor count exceeds int32 capacity");
    }
    if (host_pairs < 0) {
        throw std::runtime_error("CUDA NEP image pair count is negative");
    }
    const std::size_t pairs = static_cast<std::size_t>(host_pairs);
    ensure_capacity(&atoms_, &atoms_capacity_, pairs);
    ensure_capacity(&displacements_, &displacements_capacity_, pairs * 3U);
    ensure_capacity(&distance2_, &distance2_capacity_, pairs);
    if (blocks > 0 && pairs > 0) {
        build_nep_image_neighbors_kernel<true><<<blocks, block_size, 0, stream>>>(
            static_cast<int>(atom_count), static_cast<int>(batch.position_stride()),
            batch.positions_soa(), atom_to_structure_, batch.offsets(), batch.cells(),
            batch.pbc(), image_counts_, reference_cells_, reference_cell_inverses_, cutoff2,
            offsets_, neighbor_counts_, atoms_, displacements_, distance2_, neighbor_overflow_);
        check_cuda(cudaGetLastError(), "CUDA NEP image neighbor fill failed");
    }
    pairs_ = pairs;
    slot_major_ = false;
    neighbor_stride_ = 0;
}

void DeviceNeighborGraph::clear() noexcept {
    release(offsets_);
    release(atoms_);
    release(shifts_);
    release(displacements_);
    release(distance2_);
    release(dpa_positions_);
    release(dpa_image_bounds_);
    release(dpa_reference_inverses_);
    pairs_ = 0;
    max_neighbors_ = 0;
    slot_major_ = false;
    neighbor_stride_ = 0;
    offsets_capacity_ = 0;
    atoms_capacity_ = 0;
    shifts_capacity_ = 0;
    displacements_capacity_ = 0;
    distance2_capacity_ = 0;
    dpa_positions_capacity_ = 0;
    dpa_image_bounds_capacity_ = 0;
    dpa_reference_inverses_capacity_ = 0;
    release(atom_to_structure_);
    release(cell_counts_);
    release(cell_offsets_);
    release(cell_fill_);
    release(cell_atoms_);
    release(cell_sort_keys_);
    release(atom_cells_);
    release(neighbor_counts_);
    release(image_counts_);
    release(neighbor_overflow_);
    release(structure_cell_offsets_);
    release(structure_cell_dims_);
    release(reference_cells_);
    release(reference_cell_inverses_);
    atom_to_structure_capacity_ = 0;
    atom_cells_capacity_ = 0;
    neighbor_counts_capacity_ = 0;
    image_counts_capacity_ = 0;
    neighbor_overflow_capacity_ = 0;
    cell_atoms_capacity_ = 0;
    structure_cell_offsets_capacity_ = 0;
    structure_cell_dims_capacity_ = 0;
    reference_cells_capacity_ = 0;
    reference_cell_inverses_capacity_ = 0;
    cell_counts_capacity_ = 0;
    cell_offsets_capacity_ = 0;
    cell_fill_capacity_ = 0;
    cell_sort_keys_capacity_ = 0;
}

} // namespace mdescriptor::cuda
