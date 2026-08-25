#pragma once

#include "mdescriptor/descriptor.hpp"
#include "mdescriptor/detail/math3.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <unordered_map>
#include <utility>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace mdescriptor::detail {

inline Mat3 cell(const StructureBatchView& batch, std::int64_t structure) {
    Mat3 result;
    const double* c = batch.cells + structure * 9;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            result.a[i][j] = c[i * 3 + j];
        }
    }
    return result;
}

template <typename Function>
inline void run_parallel_structures(
    std::int64_t structures,
    int requested_threads,
    const std::shared_ptr<ComputeControl>& control,
    Function&& fn
) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(requested_threads > 0 ? requested_threads : omp_get_max_threads())
#endif
    for (std::int64_t s = 0; s < structures; ++s) {
        if (cancelled(control)) {
            continue;
        }
        fn(s);
        if (control) {
            control->mark_completed();
        }
    }
    if (cancelled(control)) {
        throw CancelledError();
    }
}

} // namespace mdescriptor::detail
