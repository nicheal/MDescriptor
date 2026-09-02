#pragma once

#include <cmath>
#include <cstdint>

#if defined(__CUDACC__)
#define MDESCRIPTOR_MBTR_HD __host__ __device__
#else
#define MDESCRIPTOR_MBTR_HD
#endif

namespace mdescriptor::detail::mbtr {

constexpr double kPi = 3.141592653589793238462643383279502884;
// Contributions outside this window are already discarded by the existing
// MBTR cutoff policy.  Applying the same policy to individual bins avoids
// evaluating Gaussian tails that are below the descriptor's retained
// numerical support.
constexpr double kGaussianTailSigma = 8.0;

constexpr int kGeometryAtomicNumber = 0;
constexpr int kGeometryDistance = 1;
constexpr int kGeometryInverseDistance = 2;
constexpr int kGeometryAngle = 3;
constexpr int kGeometryCosine = 4;

constexpr int kWeightingUnity = 0;
constexpr int kWeightingExponential = 1;
constexpr int kWeightingInverseSquare = 2;
constexpr int kWeightingSmoothCutoff = 3;

constexpr int kNormalizationNone = 0;
constexpr int kNormalizationL2 = 1;
constexpr int kNormalizationNAtoms = 2;
constexpr int kNormalizationValleOganov = 3;

MDESCRIPTOR_MBTR_HD inline double sqrt_value(double value) {
#if defined(__CUDA_ARCH__)
    return sqrt(value);
#else
    return std::sqrt(value);
#endif
}

MDESCRIPTOR_MBTR_HD inline double exp_value(double value) {
#if defined(__CUDA_ARCH__)
    return exp(value);
#else
    return std::exp(value);
#endif
}

MDESCRIPTOR_MBTR_HD inline double erf_value(double value) {
#if defined(__CUDA_ARCH__)
    return erf(value);
#else
    return std::erf(value);
#endif
}

MDESCRIPTOR_MBTR_HD inline double pow_value(double base, double exponent) {
#if defined(__CUDA_ARCH__)
    return pow(base, exponent);
#else
    return std::pow(base, exponent);
#endif
}

MDESCRIPTOR_MBTR_HD inline double max_value(double left, double right) {
    return left > right ? left : right;
}

MDESCRIPTOR_MBTR_HD inline double min_value(double left, double right) {
    return left < right ? left : right;
}

MDESCRIPTOR_MBTR_HD inline double abs_value(double value) {
#if defined(__CUDA_ARCH__)
    return fabs(value);
#else
    return std::abs(value);
#endif
}

MDESCRIPTOR_MBTR_HD inline double ceil_value(double value) {
#if defined(__CUDA_ARCH__)
    return ceil(value);
#else
    return std::ceil(value);
#endif
}

MDESCRIPTOR_MBTR_HD inline double floor_value(double value) {
#if defined(__CUDA_ARCH__)
    return floor(value);
#else
    return std::floor(value);
#endif
}

MDESCRIPTOR_MBTR_HD inline double gaussian_bin(
    double value,
    double weight,
    double grid_min,
    double grid_max,
    double grid_sigma,
    int grid_n,
    bool normalize,
    int bin) {
    const double dx = (grid_max - grid_min) / (grid_n - 1);
    const double lower = grid_min - 0.5 * dx + bin * dx;
    const double upper = lower + dx;
    const double sigma_root = grid_sigma * sqrt_value(2.0);
    double result = 0.5 * (erf_value((upper - value) / sigma_root)
        - erf_value((lower - value) / sigma_root)) / dx;
    if (!normalize) result *= grid_sigma * sqrt_value(2.0 * kPi);
    return weight * result;
}

MDESCRIPTOR_MBTR_HD inline double gaussian_bin_precomputed(
    double value,
    double weight,
    double grid_min,
    double dx,
    double sigma_root,
    double unnormalized_scale,
    bool normalize,
    int bin) {
    const double lower = grid_min - 0.5 * dx + bin * dx;
    const double upper = lower + dx;
    double result = 0.5 * (erf_value((upper - value) / sigma_root)
        - erf_value((lower - value) / sigma_root)) / dx;
    if (!normalize) result *= unnormalized_scale;
    return weight * result;
}

MDESCRIPTOR_MBTR_HD inline void add_histogram(
    double* target,
    double value,
    double weight,
    double grid_min,
    double grid_max,
    double grid_sigma,
    int grid_n,
    bool normalize) {
    if (weight == 0.0 || value < grid_min - grid_sigma * kGaussianTailSigma
        || value > grid_max + grid_sigma * kGaussianTailSigma) {
        return;
    }
    const double dx = (grid_max - grid_min) / (grid_n - 1);
    const double half_dx = 0.5 * dx;
    const double tail = grid_sigma * kGaussianTailSigma;
    const double sigma_root = grid_sigma * sqrt_value(2.0);
    const double unnormalized_scale = grid_sigma * sqrt_value(2.0 * kPi);
    const double first_coordinate =
        (value - tail - grid_min - half_dx) / dx;
    const double last_coordinate =
        (value + tail - grid_min + half_dx) / dx;
    const int first_bin = static_cast<int>(min_value(
        static_cast<double>(grid_n - 1), max_value(0.0, ceil_value(first_coordinate))));
    const int last_bin = static_cast<int>(max_value(
        0.0, min_value(static_cast<double>(grid_n - 1), floor_value(last_coordinate))));
    if (first_bin > last_bin) return;
    for (int bin = first_bin; bin <= last_bin; ++bin) {
        target[bin] += gaussian_bin_precomputed(
            value, weight, grid_min, dx, sigma_root, unnormalized_scale,
            normalize, bin);
    }
}

MDESCRIPTOR_MBTR_HD inline double smooth_cutoff(
    double distance, double r_cut, double sharpness) {
    const double x = min_value(max_value(distance / r_cut, 0.0), 1.0);
    return 1.0 + sharpness * pow_value(x, sharpness + 1.0)
        - (sharpness + 1.0) * pow_value(x, sharpness);
}

MDESCRIPTOR_MBTR_HD inline double weight(
    int weighting,
    double scale,
    double threshold,
    double r_cut,
    double sharpness,
    double first,
    double second,
    double third) {
    if (weighting == kWeightingExponential) {
        const double value = exp_value(-scale * (first + second + third));
        return value >= threshold ? value : 0.0;
    }
    if (weighting == kWeightingInverseSquare) {
        return first <= r_cut ? 1.0 / max_value(first * first, 1e-30) : 0.0;
    }
    if (weighting == kWeightingSmoothCutoff) {
        return first <= r_cut && second <= r_cut
            ? smooth_cutoff(first, r_cut, sharpness)
                * smooth_cutoff(second, r_cut, sharpness)
            : 0.0;
    }
    return 1.0;
}

MDESCRIPTOR_MBTR_HD inline int pair_channel(
    int first, int second, int species_count) {
    const int lower = first < second ? first : second;
    const int upper = first > second ? first : second;
    return lower * species_count - lower * (lower + 1) / 2 + upper;
}

MDESCRIPTOR_MBTR_HD inline double cell_volume(const double* cell) {
    return abs_value(
        cell[0] * (cell[4] * cell[8] - cell[5] * cell[7])
        - cell[1] * (cell[3] * cell[8] - cell[5] * cell[6])
        + cell[2] * (cell[3] * cell[7] - cell[4] * cell[6]));
}

MDESCRIPTOR_MBTR_HD inline void normalize_l2(
    double* values, std::int64_t size) {
    double squared = 0.0;
    for (std::int64_t index = 0; index < size; ++index) {
        squared += values[index] * values[index];
    }
    const double norm = sqrt_value(squared);
    if (norm > 0.0) {
        for (std::int64_t index = 0; index < size; ++index) {
            values[index] /= norm;
        }
    }
}

MDESCRIPTOR_MBTR_HD inline void normalize_n_atoms(
    double* values, std::int64_t size, int atom_count) {
    if (atom_count <= 0) return;
    for (std::int64_t index = 0; index < size; ++index) {
        values[index] /= atom_count;
    }
}

MDESCRIPTOR_MBTR_HD inline void normalize_valle_oganov(
    double* values,
    double volume,
    const int* species_counts,
    int species_count,
    int geometry,
    int grid_n,
    bool local) {
    if (local || geometry == kGeometryAtomicNumber) return;
    const int pair_count = species_count * (species_count + 1) / 2;
    if (geometry == kGeometryDistance || geometry == kGeometryInverseDistance) {
        for (int first = 0; first < species_count; ++first) {
            for (int second = first; second < species_count; ++second) {
                const double count_product = first == second
                    ? 0.5 * static_cast<double>(species_counts[first]) * species_counts[second]
                    : static_cast<double>(species_counts[first]) * species_counts[second];
                if (count_product <= 0.0) continue;
                const double factor = volume / (count_product * 4.0 * kPi);
                const std::int64_t begin = static_cast<std::int64_t>(
                    pair_channel(first, second, species_count)) * grid_n;
                for (int bin = 0; bin < grid_n; ++bin) {
                    values[begin + bin] *= factor;
                }
            }
        }
        return;
    }
    for (int first = 0; first < species_count; ++first) {
        for (int center = 0; center < species_count; ++center) {
            for (int third = first; third < species_count; ++third) {
                const double count_product = static_cast<double>(species_counts[first])
                    * species_counts[center] * species_counts[third];
                if (count_product <= 0.0) continue;
                const int channel = center * pair_count
                    + pair_channel(first, third, species_count);
                const std::int64_t begin = static_cast<std::int64_t>(channel) * grid_n;
                const double factor = volume / count_product;
                for (int bin = 0; bin < grid_n; ++bin) {
                    values[begin + bin] *= factor;
                }
            }
        }
    }
}

} // namespace mdescriptor::detail::mbtr

#undef MDESCRIPTOR_MBTR_HD
