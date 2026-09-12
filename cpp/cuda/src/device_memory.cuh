#pragma once

// Host-side device-memory helpers shared by the CUDA backend state objects.
// Allocations are grow-only: a buffer keeps its previous contents until a
// larger size forces a reallocation.  Callers own the raw pointers.

#include "mdescriptor/cuda/error.hpp"

#include <cuda_runtime.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>

namespace mdescriptor::cuda {

// Owning handle for one model-owned device allocation.
struct DeviceArray {
    void* pointer = nullptr;
    std::size_t bytes = 0;
    ~DeviceArray() noexcept {
        if (pointer != nullptr) (void)cudaFree(pointer);
    }
};

// The shared release pattern: free the allocation and null the pointer.
template <typename Value>
void device_free(Value*& pointer) noexcept {
    if (pointer != nullptr) {
        (void)cudaFree(pointer);
        pointer = nullptr;
    }
}

// Grow a buffer to hold at least count elements.  Contents are undefined
// afterwards; count == 0 never allocates.
template <typename Value>
void ensure_capacity(Value** destination, std::size_t* capacity, std::size_t count) {
    if (count <= *capacity) {
        return;
    }
    if (*destination != nullptr) {
        check_cuda(cudaFree(*destination), "could not release CUDA device buffer");
        *destination = nullptr;
    }
    *capacity = 0;
    if (count == 0) {
        return;
    }
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(Value)) {
        throw CudaOutOfMemory("requested CUDA device buffer is too large");
    }
    check_cuda(
        cudaMalloc(reinterpret_cast<void**>(destination), count * sizeof(Value)),
        "could not allocate CUDA device buffer");
    *capacity = count;
}

// Grow a buffer to count elements and copy them from the host.  The error
// string is per call site; reallocation failures use the shared messages.
template <typename Value>
void ensure_and_upload(
    Value** destination,
    std::size_t* capacity,
    const Value* source,
    std::size_t count,
    cudaStream_t stream,
    const char* operation) {
    ensure_capacity(destination, capacity, count);
    if (count != 0) {
        check_cuda(
            cudaMemcpyAsync(
                *destination, source, count * sizeof(Value),
                cudaMemcpyHostToDevice, stream),
            operation);
    }
}

// Invert a row-major 3x3 matrix.  Returns false for a singular or
// non-finite determinant.
inline bool inverse_row_major3(const double* matrix, double* inverse) {
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

// Shared prologue of the NEP periodic-cell planning helpers: transpose the
// row-major cell into reference_cell, invert it into reference_inverse, and
// return the Euclidean norms of the inverse rows (the reciprocal lattice
// vector lengths that scale the per-axis image counts).
inline std::array<double, 3> reciprocal_row_norms(
    const double* source_cell,
    double* reference_cell,
    double* reference_inverse,
    const char* singular_error) {
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
            reference_cell[row * 3 + column] = source_cell[column * 3 + row];
        }
    }
    if (!inverse_row_major3(reference_cell, reference_inverse)) {
        throw std::invalid_argument(singular_error);
    }
    std::array<double, 3> norms{};
    for (int axis = 0; axis < 3; ++axis) {
        const double x = reference_inverse[axis * 3 + 0];
        const double y = reference_inverse[axis * 3 + 1];
        const double z = reference_inverse[axis * 3 + 2];
        norms[static_cast<std::size_t>(axis)] = std::sqrt(x * x + y * y + z * z);
    }
    return norms;
}

} // namespace mdescriptor::cuda
