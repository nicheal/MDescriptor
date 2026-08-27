#pragma once

// scipy-openblas32 is linked as a private, prefixed CBLAS implementation.
// Keeping this tiny adapter in one header makes it difficult for a future
// kernel to accidentally call an unprefixed system/NumPy BLAS symbol.
#include <cblas.h>

#include <cstddef>
#include <limits>
#include <stdexcept>

namespace mdescriptor::detail {

inline int blas_int(std::size_t value, const char* name) {
    if (value > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument(name);
    }
    return static_cast<int>(value);
}

inline void set_blas_single_thread() noexcept {
    // DPA owns the outer OpenMP parallelism.  A process-wide setting is safe
    // here because the linked symbols are scipy_'s private, wheel-vendored
    // ABI rather than the unprefixed BLAS used by another package.
    scipy_openblas_set_num_threads(1);
}

inline void sgemm(
    std::size_t rows,
    std::size_t columns,
    std::size_t inner,
    const float* left,
    std::size_t left_stride,
    const float* right,
    std::size_t right_stride,
    float* result,
    std::size_t result_stride) {
    scipy_cblas_sgemm(
        CblasRowMajor,
        CblasNoTrans,
        CblasNoTrans,
        blas_int(rows, "BLAS row count is too large"),
        blas_int(columns, "BLAS column count is too large"),
        blas_int(inner, "BLAS inner dimension is too large"),
        1.0F,
        left,
        blas_int(left_stride, "BLAS left stride is too large"),
        right,
        blas_int(right_stride, "BLAS right stride is too large"),
        0.0F,
        result,
        blas_int(result_stride, "BLAS result stride is too large"));
}

} // namespace mdescriptor::detail
