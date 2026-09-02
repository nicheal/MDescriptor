#pragma once

#include "mdescriptor/matrix.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace mdescriptor::detail {

struct MatrixLayout {
    std::int64_t n_atoms_max;
    MatrixPermutation permutation;

    std::int64_t columns() const noexcept {
        return permutation == MatrixPermutation::Eigenspectrum
            ? n_atoms_max
            : n_atoms_max * n_atoms_max;
    }

    std::int64_t stride() const noexcept {
        return columns();
    }
};

MatrixPermutation matrix_permutation_from_name(const std::string& name);
MatrixLayout make_matrix_layout(std::int64_t n_atoms_max, MatrixPermutation permutation);
MatrixLayout make_matrix_layout(std::int64_t n_atoms_max, const std::string& permutation);

void write_matrix(
    std::vector<double> matrix,
    int count,
    const MatrixLayout& layout,
    double* output);

} // namespace mdescriptor::detail
