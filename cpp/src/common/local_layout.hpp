#pragma once

#include "mdescriptor/local_descriptors.hpp"

#include <cstdint>
#include <stdexcept>

namespace mdescriptor::detail {

// Backend-neutral feature layout for the first local descriptor family.  Both
// CPU and CUDA use these formulas so a device implementation cannot silently
// change species/radial/angular ordering or feature counts.
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
