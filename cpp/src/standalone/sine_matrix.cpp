#include "extra_common.hpp"
#include "matrix_values.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace mdescriptor::detail {

std::vector<double> sine_matrix_values(
    const StructureBatchView& batch,
    std::int64_t structure,
    double exponent,
    int num_threads) {
    const std::int64_t begin = batch.offsets[structure];
    const std::int64_t end = batch.offsets[structure + 1];
    const int count = static_cast<int>(end - begin);
    const Mat3 cell = load_cell(batch, structure);
    const Mat3 inverse_cell = inverse(cell);
    std::vector<double> matrix(static_cast<std::size_t>(count * count), 0.0);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(num_threads > 0 ? num_threads : omp_get_max_threads()) if(count >= 32 && !omp_in_parallel())
#endif
    for (int i = 0; i < count; ++i) {
        const Vec3 first = position(batch, begin + i);
        for (int j = 0; j < count; ++j) {
            const Vec3 second = position(batch, begin + j);
            const Vec3 delta = first - second;
            const double fractional[3] = {
                delta.x * inverse_cell.a[0][0] + delta.y * inverse_cell.a[1][0] + delta.z * inverse_cell.a[2][0],
                delta.x * inverse_cell.a[0][1] + delta.y * inverse_cell.a[1][1] + delta.z * inverse_cell.a[2][1],
                delta.x * inverse_cell.a[0][2] + delta.y * inverse_cell.a[1][2] + delta.z * inverse_cell.a[2][2],
            };
            const Vec3 sine_squared{std::sin(kPi * fractional[0]) * std::sin(kPi * fractional[0]),
                                    std::sin(kPi * fractional[1]) * std::sin(kPi * fractional[1]),
                                    std::sin(kPi * fractional[2]) * std::sin(kPi * fractional[2])};
            const Vec3 transformed = {
                sine_squared.x * cell.a[0][0] + sine_squared.y * cell.a[1][0] + sine_squared.z * cell.a[2][0],
                sine_squared.x * cell.a[0][1] + sine_squared.y * cell.a[1][1] + sine_squared.z * cell.a[2][1],
                sine_squared.x * cell.a[0][2] + sine_squared.y * cell.a[1][2] + sine_squared.z * cell.a[2][2],
            };
            const double denominator = norm(transformed);
            const double zi = static_cast<double>(batch.numbers[begin + i]);
            const double zj = static_cast<double>(batch.numbers[begin + j]);
            matrix[static_cast<std::size_t>(i * count + j)] = denominator > 1e-14 ? zi * zj / denominator : 0.0;
        }
        matrix[static_cast<std::size_t>(i * count + i)] = 0.5 * std::pow(static_cast<double>(batch.numbers[begin + i]), exponent);
    }
    return matrix;
}

} // namespace mdescriptor::detail
