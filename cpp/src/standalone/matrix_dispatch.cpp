#include "mdescriptor/matrix.hpp"

#include "matrix_output.hpp"
#include "matrix_values.hpp"
#include "descriptor_common.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

namespace mdescriptor {
using namespace detail;

namespace {

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
