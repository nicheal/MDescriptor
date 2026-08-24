#pragma once

#include "mdescriptor/descriptor.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <unordered_map>
#include <utility>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace mdescriptor::detail {

struct Vec3 {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
};

inline Vec3 operator+(Vec3 a, Vec3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
inline Vec3 operator-(Vec3 a, Vec3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
inline Vec3 operator*(double a, Vec3 b) { return {a * b.x, a * b.y, a * b.z}; }
inline double dot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline double norm2(Vec3 a) { return dot(a, a); }

struct Mat3 {
    double a[3][3]{};
};

inline double determinant(const Mat3& m) {
    return m.a[0][0] * (m.a[1][1] * m.a[2][2] - m.a[1][2] * m.a[2][1])
         - m.a[0][1] * (m.a[1][0] * m.a[2][2] - m.a[1][2] * m.a[2][0])
         + m.a[0][2] * (m.a[1][0] * m.a[2][1] - m.a[1][1] * m.a[2][0]);
}

inline Mat3 inverse(const Mat3& m) {
    const double d = determinant(m);
    if (!std::isfinite(d) || std::abs(d) < 1e-14) {
        throw std::invalid_argument("cell matrix is singular");
    }
    Mat3 r;
    r.a[0][0] = (m.a[1][1] * m.a[2][2] - m.a[1][2] * m.a[2][1]) / d;
    r.a[0][1] = (m.a[0][2] * m.a[2][1] - m.a[0][1] * m.a[2][2]) / d;
    r.a[0][2] = (m.a[0][1] * m.a[1][2] - m.a[0][2] * m.a[1][1]) / d;
    r.a[1][0] = (m.a[1][2] * m.a[2][0] - m.a[1][0] * m.a[2][2]) / d;
    r.a[1][1] = (m.a[0][0] * m.a[2][2] - m.a[0][2] * m.a[2][0]) / d;
    r.a[1][2] = (m.a[0][2] * m.a[1][0] - m.a[0][0] * m.a[1][2]) / d;
    r.a[2][0] = (m.a[1][0] * m.a[2][1] - m.a[1][1] * m.a[2][0]) / d;
    r.a[2][1] = (m.a[0][1] * m.a[2][0] - m.a[0][0] * m.a[2][1]) / d;
    r.a[2][2] = (m.a[0][0] * m.a[1][1] - m.a[0][1] * m.a[1][0]) / d;
    return r;
}

inline Mat3 cell(const StructureBatchView& batch, std::int64_t structure) {
    Mat3 result;
    const double* c = batch.cells + structure * 9;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            result.a[i][j] = c[i * 3 + j];
        }
    }
    return result;
}

inline void validate_common(const StructureBatchView& batch) {
    if (batch.structures < 0 || batch.atoms < 0 || batch.offsets == nullptr) {
        throw std::invalid_argument("invalid structure batch");
    }
    if (batch.structures > 0 && (batch.numbers == nullptr || batch.positions == nullptr || batch.cells == nullptr || batch.pbc == nullptr)) {
        throw std::invalid_argument("structure batch contains null arrays");
    }
    if (batch.structures == 0) {
        return;
    }
    for (std::int64_t s = 0; s < batch.structures; ++s) {
        if (batch.offsets[s] > batch.offsets[s + 1]) {
            throw std::invalid_argument("offsets must be monotonic");
        }
        for (int d = 0; d < 3; ++d) {
            if (batch.pbc[s * 3 + d] != 1) {
                throw std::invalid_argument("only fully periodic structures are supported");
            }
        }
        (void)inverse(cell(batch, s));
    }
    if (batch.offsets[0] != 0 || batch.offsets[batch.structures] != batch.atoms) {
        throw std::invalid_argument("offsets do not describe the flattened arrays");
    }
    for (std::int64_t i = 0; i < batch.atoms * 3; ++i) {
        if (!std::isfinite(batch.positions[i])) {
            throw std::invalid_argument("positions must be finite");
        }
    }
    for (std::int64_t i = 0; i < batch.structures * 9; ++i) {
        if (!std::isfinite(batch.cells[i])) {
            throw std::invalid_argument("cells must be finite");
        }
    }
}

inline std::unordered_map<std::int32_t, std::int32_t> species_map(const std::vector<std::int32_t>& species) {
    std::unordered_map<std::int32_t, std::int32_t> result;
    for (std::size_t i = 0; i < species.size(); ++i) {
        if (!result.emplace(species[i], static_cast<std::int32_t>(i)).second) {
            throw std::invalid_argument("species must not contain duplicates");
        }
    }
    return result;
}

inline void validate_species(const StructureBatchView& batch, const std::vector<std::int32_t>& species) {
    const auto mapping = species_map(species);
    for (std::int64_t i = 0; i < batch.atoms; ++i) {
        if (batch.numbers[i] <= 0) {
            throw std::invalid_argument("atomic numbers must be positive");
        }
        if (mapping.find(batch.numbers[i]) == mapping.end()) {
            throw std::invalid_argument("batch contains an atomic number outside calculator species");
        }
    }
}

inline bool cancelled(const std::shared_ptr<ComputeControl>& control) {
    return control && control->cancelled();
}

template <typename Function>
inline void run_parallel_structures(
    std::int64_t structures,
    int requested_threads,
    const std::shared_ptr<ComputeControl>& control,
    Function&& fn
) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(requested_threads > 0 ? requested_threads : omp_get_max_threads())
#endif
    for (std::int64_t s = 0; s < structures; ++s) {
        if (cancelled(control)) {
            continue;
        }
        fn(s);
        if (control) {
            control->mark_completed();
        }
    }
    if (cancelled(control)) {
        throw CancelledError();
    }
}

} // namespace mdescriptor::detail

