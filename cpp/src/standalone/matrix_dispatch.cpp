#include "mdescriptor/matrix.hpp"

#include "matrix_output.hpp"
#include "matrix_values.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <exception>
#include <stdexcept>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace mdescriptor {
using namespace detail;

namespace {

template <typename Function>
void run_parallel_matrix_structures(
    std::int64_t structures,
    int requested_threads,
    const std::shared_ptr<ComputeControl>& control,
    Function&& fn) {
    // Exceptions must not escape an OpenMP loop: doing so calls std::terminate.
    // Keep each structure independent, capture failures in the worker, and
    // rethrow after all workers have joined.
    std::vector<std::exception_ptr> exceptions(static_cast<std::size_t>(structures));
    std::atomic<bool> failed{false};
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(requested_threads > 0 ? requested_threads : omp_get_max_threads())
#endif
    for (std::int64_t structure = 0; structure < structures; ++structure) {
        if (failed.load(std::memory_order_acquire) || (control && control->cancelled())) {
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
    if (control && control->cancelled()) {
        throw CancelledError();
    }
}

void validate_matrix_kind(MatrixKind kind) {
    switch (kind) {
    case MatrixKind::Sine:
    case MatrixKind::Ewald:
    case MatrixKind::Coulomb:
        return;
    }
    throw std::invalid_argument("invalid matrix descriptor kind");
}

} // namespace

void compute_matrix(
    const StructureBatchView& batch,
    const MatrixOptions& options,
    double* output,
    const std::shared_ptr<ComputeControl>& control) {
    validate_batch(batch);
    validate_matrix_kind(options.kind);
    const auto layout = make_matrix_layout(options.n_atoms_max, options.permutation);
    const bool exponent_valid = std::isfinite(options.exponent)
        && (options.kind == MatrixKind::Coulomb || options.exponent > 0.0);
    if (!exponent_valid || options.num_threads < 0) {
        throw std::invalid_argument("invalid matrix descriptor parameters");
    }
    if (options.kind == MatrixKind::Ewald
        && (!std::isfinite(options.accuracy) || options.accuracy <= 0.0 || options.accuracy >= 1.0)) {
        throw std::invalid_argument("accuracy must be between zero and one");
    }
    for (std::int64_t structure = 0; structure < batch.structures; ++structure) {
        const std::int64_t count = batch.offsets[structure + 1] - batch.offsets[structure];
        if (count > layout.n_atoms_max) {
            throw std::invalid_argument("structure exceeds n_atoms_max");
        }
    }

    check_cancelled(control);
    if (control) {
        control->reset(batch.structures);
    }
    std::fill(output, output + batch.structures * layout.stride(), 0.0);

    auto compute_structure = [&](std::int64_t structure) {
        const int count = static_cast<int>(batch.offsets[structure + 1] - batch.offsets[structure]);
        std::vector<double> matrix;
        switch (options.kind) {
        case MatrixKind::Coulomb:
            matrix = coulomb_matrix_values(batch, structure, options.exponent, options.num_threads);
            break;
        case MatrixKind::Sine:
            matrix = sine_matrix_values(batch, structure, options.exponent, options.num_threads);
            break;
        case MatrixKind::Ewald:
            matrix = ewald_matrix_values(
                batch, structure, options.exponent, options.accuracy, options.w,
                options.r_cut, options.g_cut, options.a, options.num_threads);
            break;
        }
        write_matrix(
            std::move(matrix), count, layout, output + structure * layout.stride());
        mark_completed(control);
    };
    if (batch.structures == 1) {
        compute_structure(0);
        return;
    }
    run_parallel_matrix_structures(
        batch.structures, options.num_threads, control, compute_structure);
}

} // namespace mdescriptor
