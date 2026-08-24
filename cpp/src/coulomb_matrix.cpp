#include "mdescriptor/descriptor.hpp"
#include "matrix_common.hpp"

#include <cmath>
#include <cstddef>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

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
    // DScribe's Windows wheels use UCRT; MinGW otherwise resolves pow from legacy msvcrt.
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

void compute_coulomb_matrix(
    const StructureBatchView& batch,
    std::int64_t n_atoms_max,
    const std::string& permutation,
    double exponent,
    double* output,
    const std::shared_ptr<ComputeControl>& control
) {
    validate_batch(batch);
    if (n_atoms_max <= 0 || !std::isfinite(exponent)
        || (permutation != "none" && permutation != "sorted_l2" && permutation != "eigenspectrum")) {
        throw std::invalid_argument("invalid Coulomb matrix parameters");
    }
    if (control) {
        control->reset(batch.structures);
    }

    const std::int64_t stride = permutation == "eigenspectrum"
        ? n_atoms_max
        : n_atoms_max * n_atoms_max;
    for (std::int64_t structure = 0; structure < batch.structures; ++structure) {
        if (control && control->cancelled()) {
            throw CancelledError();
        }
        const std::int64_t begin = batch.offsets[structure];
        const std::int64_t end = batch.offsets[structure + 1];
        const std::int64_t atom_count = end - begin;
        if (atom_count <= 0) {
            throw std::invalid_argument("matrix descriptors do not accept empty structures");
        }
        if (atom_count > n_atoms_max) {
            throw std::invalid_argument("structure exceeds n_atoms_max");
        }

        std::vector<double> matrix(static_cast<std::size_t>(atom_count * atom_count), 0.0);
        for (std::int64_t i = 0; i < atom_count; ++i) {
            const double zi = static_cast<double>(batch.numbers[begin + i]);
            for (std::int64_t j = i; j < atom_count; ++j) {
                double value;
                if (i == j) {
                    value = 0.5 * reference_pow(zi, exponent);
                } else {
                    const double zj = static_cast<double>(batch.numbers[begin + j]);
                    const double* first = batch.positions + (begin + i) * 3;
                    const double* second = batch.positions + (begin + j) * 3;
                    const double dx = first[0] - second[0];
                    const double dy = first[1] - second[1];
                    const double dz = first[2] - second[2];
                    const double distance = std::sqrt(dx * dx + dy * dy + dz * dz);
                    value = zi * zj / distance;
                }
                matrix[static_cast<std::size_t>(i * atom_count + j)] = value;
                matrix[static_cast<std::size_t>(j * atom_count + i)] = value;
            }
        }

        double* row = output + structure * stride;
        write_matrix(std::move(matrix), static_cast<int>(atom_count), n_atoms_max, permutation, row);
        if (control) {
            control->mark_completed();
        }
    }
}

namespace detail {

std::vector<double> coulomb_matrix_values(const StructureBatchView& batch, std::int64_t structure, double exponent) {
    const std::int64_t begin = batch.offsets[structure];
    const std::int64_t end = batch.offsets[structure + 1];
    const int count = static_cast<int>(end - begin);
    std::vector<double> matrix(static_cast<std::size_t>(count * count), 0.0);
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
