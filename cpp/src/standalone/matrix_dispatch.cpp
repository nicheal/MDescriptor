#include "mdescriptor/extra.hpp"
#include "descriptor_common.hpp"
#include "matrix_common.hpp"

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace mdescriptor {
using namespace detail;

void compute_matrix(
    const StructureBatchView& batch,
    std::int64_t n_atoms_max,
    const std::string& permutation,
    double exponent,
    MatrixKind kind,
    double accuracy,
    double w,
    double r_cut,
    double g_cut,
    double a,
    int num_threads,
    double* output,
    const std::shared_ptr<ComputeControl>& control) {
    validate_batch(batch);
    if (n_atoms_max <= 0 || exponent <= 0.0 || !std::isfinite(exponent) || num_threads < 0) {
        throw std::invalid_argument("invalid matrix descriptor parameters");
    }
    if (kind == MatrixKind::Ewald && (!std::isfinite(accuracy) || accuracy <= 0.0 || accuracy >= 1.0)) {
        throw std::invalid_argument("accuracy must be between zero and one");
    }
    check_cancelled(control);
    if (control) {
        control->reset(batch.structures);
    }
    const std::int64_t stride = permutation == "eigenspectrum" ? n_atoms_max : n_atoms_max * n_atoms_max;
    std::fill(output, output + batch.structures * stride, 0.0);
    if (kind == MatrixKind::Ewald) {
        if (batch.structures == 1) {
            auto matrix = ewald_matrix_values(
                batch, 0, exponent, accuracy, w, r_cut, g_cut, a, num_threads);
            write_matrix(std::move(matrix),
                static_cast<int>(batch.offsets[1] - batch.offsets[0]), n_atoms_max,
                permutation, output);
            mark_completed(control);
            return;
        }
        std::vector<std::vector<double>> matrices(static_cast<std::size_t>(batch.structures));
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(num_threads > 0 ? num_threads : omp_get_max_threads())
#endif
        for (std::int64_t structure = 0; structure < batch.structures; ++structure) {
            if (control && control->cancelled()) {
                continue;
            }
            matrices[static_cast<std::size_t>(structure)] = ewald_matrix_values(
                batch, structure, exponent, accuracy, w, r_cut, g_cut, a, num_threads);
        }
        check_cancelled(control);
        for (std::int64_t structure = 0; structure < batch.structures; ++structure) {
            check_cancelled(control);
            const int count = static_cast<int>(batch.offsets[structure + 1] - batch.offsets[structure]);
            write_matrix(std::move(matrices[static_cast<std::size_t>(structure)]), count, n_atoms_max,
                permutation, output + structure * stride);
            mark_completed(control);
        }
        return;
    }
    auto compute_structure = [&](std::int64_t structure) {
        const int count = static_cast<int>(batch.offsets[structure + 1] - batch.offsets[structure]);
        auto matrix = kind == MatrixKind::Coulomb
            ? coulomb_matrix_values(batch, structure, exponent, num_threads)
            : kind == MatrixKind::Sine
                ? sine_matrix_values(batch, structure, exponent, num_threads)
                : ewald_matrix_values(batch, structure, exponent, accuracy, w, r_cut, g_cut, a, num_threads);
        write_matrix(std::move(matrix), count, n_atoms_max, permutation, output + structure * stride);
    };
    if (batch.structures == 1) {
        compute_structure(0);
        mark_completed(control);
        return;
    }
    run_parallel_structures(batch.structures, num_threads, control, compute_structure);
}
} // namespace mdescriptor
