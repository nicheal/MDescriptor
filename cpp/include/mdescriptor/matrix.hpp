#pragma once

#include "descriptor.hpp"

#include <cstdint>

namespace mdescriptor {

enum class MatrixKind : std::int32_t {
    Sine = 0,
    Ewald = 1,
    Coulomb = 2,
};

enum class MatrixPermutation : std::uint8_t {
    None,
    SortedL2,
    Eigenspectrum,
};

// Parameters shared by the three matrix descriptors.  The provider-specific
// fields are intentionally kept here so the batch scheduler has one native
// entry point; each provider decides which fields it needs.
struct MatrixOptions {
    std::int64_t n_atoms_max = 0;
    MatrixPermutation permutation = MatrixPermutation::SortedL2;
    MatrixKind kind = MatrixKind::Sine;
    double exponent = 2.4;
    double accuracy = 1e-5;
    double w = 1.0;
    double r_cut = 0.0;
    double g_cut = 0.0;
    double a = 0.0;
    int num_threads = 0;
};

void compute_matrix(
    const StructureBatchView& batch,
    const MatrixOptions& options,
    double* output,
    const std::shared_ptr<ComputeControl>& control);

} // namespace mdescriptor
