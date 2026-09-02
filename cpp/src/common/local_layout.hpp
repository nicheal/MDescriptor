#pragma once

#include "mdescriptor/local_descriptors.hpp"

#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace mdescriptor::detail {

#if defined(__CUDACC__)
#define MDESCRIPTOR_HOST_DEVICE __host__ __device__
#else
#define MDESCRIPTOR_HOST_DEVICE
#endif

// Backend-neutral feature and coefficient layout for the local descriptor
// family. Both CPU and CUDA use these formulas so a device implementation
// cannot silently change species/radial/angular ordering or feature counts.
struct LocalFeatureLayout {
    std::int64_t species = 0;
    std::int64_t radial = 0;
    std::int64_t angular = 0;
};

inline LocalFeatureLayout local_feature_layout(const LocalDescriptorOptions& options) {
    if (options.species.empty()) {
        throw std::invalid_argument("species must not be empty");
    }
    if (options.max_radial < 0 || options.max_angular < 0) {
        throw std::invalid_argument("local descriptor orders must be non-negative");
    }
    return {
        static_cast<std::int64_t>(options.species.size()),
        static_cast<std::int64_t>(options.max_radial + 1),
        static_cast<std::int64_t>(options.max_angular + 1),
    };
}

inline int local_coefficient_max_angular(
    const LocalDescriptorOptions& options,
    LocalDescriptorKind kind) noexcept {
    return kind == LocalDescriptorKind::SoapRadialSpectrum
        ? 0
        : options.max_angular;
}

inline std::int64_t local_coefficient_count(
    const LocalDescriptorOptions& options,
    LocalDescriptorKind kind) {
    const LocalFeatureLayout layout = local_feature_layout(options);
    const auto max_angular = static_cast<std::int64_t>(
        local_coefficient_max_angular(options, kind));
    const auto angular = max_angular + 1;
    return layout.species * layout.radial * angular * angular;
}

// CPU and CUDA store one compact (2 * l + 1) block for every angular channel.
// Keeping this index in the shared layout module prevents the two backends from
// silently developing different internal coefficient representations.
inline MDESCRIPTOR_HOST_DEVICE std::size_t local_coefficient_index(
    std::size_t species,
    int radial,
    int angular,
    int m,
    int radial_count,
    int max_angular) noexcept {
    const auto angular_block = static_cast<std::size_t>(max_angular + 1);
    return ((species * static_cast<std::size_t>(radial_count)
        + static_cast<std::size_t>(radial)) * angular_block * angular_block)
        + static_cast<std::size_t>(angular * angular + angular + m);
}

inline MDESCRIPTOR_HOST_DEVICE void local_decode_spherical_feature(
    int feature,
    int radial_count,
    int max_angular,
    int& radial,
    int& angular,
    int& m) noexcept {
    int remainder = feature;
    radial = 0;
    angular = 0;
    m = 0;
    for (int candidate = 0; candidate <= max_angular; ++candidate) {
        const int block = (2 * candidate + 1) * radial_count;
        if (remainder < block) {
            angular = candidate;
            m = remainder / radial_count - candidate;
            radial = remainder % radial_count;
            return;
        }
        remainder -= block;
    }
}

inline MDESCRIPTOR_HOST_DEVICE void local_decode_species_pair(
    int pair,
    int species_count,
    int& first,
    int& second) noexcept {
    first = 0;
    second = 0;
    for (int candidate = 0; candidate < species_count; ++candidate) {
        const int count = species_count - candidate;
        if (pair < count) {
            first = candidate;
            second = candidate + pair;
            return;
        }
        pair -= count;
    }
}

inline std::int64_t local_layout_feature_count(
    const LocalDescriptorOptions& options,
    LocalDescriptorKind kind) {
    const LocalFeatureLayout layout = local_feature_layout(options);
    switch (kind) {
    case LocalDescriptorKind::SoapRadialSpectrum:
        return layout.species * layout.species * layout.radial;
    case LocalDescriptorKind::SoapPowerSpectrum:
        return layout.species * (layout.species + 1) / 2
            * layout.species * layout.angular * layout.radial * layout.radial;
    default:
        return layout.species * layout.species * layout.radial
            * layout.angular * layout.angular;
    }
}

} // namespace mdescriptor::detail

#undef MDESCRIPTOR_HOST_DEVICE
