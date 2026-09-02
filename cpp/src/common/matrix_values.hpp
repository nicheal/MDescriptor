#pragma once

#include "mdescriptor/matrix.hpp"

#include <cstdint>
#include <vector>

namespace mdescriptor::detail {

std::vector<double> sine_matrix_values(
    const StructureBatchView& batch,
    std::int64_t structure,
    double exponent,
    int num_threads);

std::vector<double> ewald_matrix_values(
    const StructureBatchView& batch,
    std::int64_t structure,
    double exponent,
    double accuracy,
    double w,
    double r_cut_option,
    double g_cut_option,
    double a_option,
    int num_threads);

std::vector<double> coulomb_matrix_values(
    const StructureBatchView& batch,
    std::int64_t structure,
    double exponent,
    int num_threads);

} // namespace mdescriptor::detail
