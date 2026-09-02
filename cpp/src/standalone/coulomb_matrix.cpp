#include "extra_common.hpp"
#include "matrix_values.hpp"

#include <cmath>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(_WIN32) && defined(__MINGW32__)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace mdescriptor {
using namespace detail;

namespace {

double reference_pow(double base, double exponent) {
#if defined(_WIN32) && defined(__MINGW32__)
    // reference implementation's Windows wheels use UCRT; MinGW otherwise resolves pow from legacy msvcrt.
    using PowFunction = double(__cdecl*)(double, double);
    static const PowFunction ucrt_pow = [] {
        HMODULE module = GetModuleHandleW(L"ucrtbase.dll");
        if (module == nullptr) {
            module = LoadLibraryW(L"ucrtbase.dll");
        }
        return module == nullptr
            ? nullptr
            : reinterpret_cast<PowFunction>(GetProcAddress(module, "pow"));
    }();
    if (ucrt_pow != nullptr) {
        return ucrt_pow(base, exponent);
    }
#endif
    return std::pow(base, exponent);
}

} // namespace

namespace detail {

std::vector<double> coulomb_matrix_values(
    const StructureBatchView& batch,
    std::int64_t structure,
    double exponent,
    int num_threads) {
    const std::int64_t begin = batch.offsets[structure];
    const std::int64_t end = batch.offsets[structure + 1];
    const int count = static_cast<int>(end - begin);
    std::vector<double> matrix(static_cast<std::size_t>(count * count), 0.0);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(num_threads > 0 ? num_threads : omp_get_max_threads()) if(count >= 32 && !omp_in_parallel())
#endif
    for (int i = 0; i < count; ++i) {
        const double zi = static_cast<double>(batch.numbers[begin + i]);
        const Vec3 first = position(batch, begin + i);
        for (int j = i; j < count; ++j) {
            const double zj = static_cast<double>(batch.numbers[begin + j]);
            const double value = i == j
                ? 0.5 * reference_pow(zi, exponent)
                : zi * zj / norm(first - position(batch, begin + j));
            matrix[static_cast<std::size_t>(i * count + j)] = value;
            matrix[static_cast<std::size_t>(j * count + i)] = value;
        }
    }
    return matrix;
}

} // namespace detail
} // namespace mdescriptor
