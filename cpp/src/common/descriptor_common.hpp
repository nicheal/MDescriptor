#pragma once

#include "mdescriptor/descriptor.hpp"
#include "mdescriptor/detail/math3.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <exception>
#include <functional>
#include <unordered_map>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace mdescriptor::detail {

// Resolves a thread request against the runtime: positive requests pass
// through, non-positive ones mean "machine default".
inline int resolved_thread_count(int requested_threads) {
#ifdef _OPENMP
    return requested_threads > 0 ? requested_threads : omp_get_max_threads();
#else
    (void)requested_threads;
    return 1;
#endif
}

// Caps the thread count at the amount of work so idle OpenMP workers are
// never spawned.
inline int effective_thread_count(std::int64_t work, int requested_threads) {
    return static_cast<int>(std::min<std::int64_t>(
        std::max<std::int64_t>(work, 1), resolved_thread_count(requested_threads)));
}

// Exceptions must not escape an OpenMP loop: doing so calls std::terminate.
// Keep each structure independent, capture failures in the worker, and
// rethrow after all workers have joined.
template <typename Function>
inline void run_captured_structures(
    std::int64_t structures,
    int threads,
    const std::shared_ptr<ComputeControl>& control,
    Function&& fn
) {
    std::vector<std::exception_ptr> exceptions(static_cast<std::size_t>(structures));
    std::atomic<bool> failed{false};
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(threads)
#endif
    for (std::int64_t structure = 0; structure < structures; ++structure) {
        if (failed.load(std::memory_order_acquire) || cancelled(control)) {
            continue;
        }
        try {
            fn(structure);
        } catch (...) {
            exceptions[static_cast<std::size_t>(structure)] = std::current_exception();
            failed.store(true, std::memory_order_release);
        }
    }
    for (const auto& exception : exceptions) {
        if (exception) {
            std::rethrow_exception(exception);
        }
    }
    if (cancelled(control)) {
        throw CancelledError();
    }
}

template <typename Function>
inline void run_parallel_structures(
    std::int64_t structures,
    int requested_threads,
    const std::shared_ptr<ComputeControl>& control,
    Function&& fn
) {
    // Structure-level kernels let this driver own progress marking (only
    // after a successful structure); matrix callers mark inside their own
    // work function.
    run_captured_structures(
        structures,
        effective_thread_count(structures, requested_threads),
        control,
        [&fn, &control](std::int64_t structure) {
            fn(structure);
            mark_completed(control);
        });
}

// Structure-parallel driver for the matrix family: it keeps the caller's raw
// thread request (no work-sized cap) because the per-structure matrices are
// large enough to fill a team on their own.
template <typename Function>
inline void run_parallel_matrix_structures(
    std::int64_t structures,
    int requested_threads,
    const std::shared_ptr<ComputeControl>& control,
    Function&& fn) {
    run_captured_structures(
        structures,
        resolved_thread_count(requested_threads),
        control,
        std::forward<Function>(fn));
}

} // namespace mdescriptor::detail
