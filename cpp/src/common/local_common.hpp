#pragma once

#include "mdescriptor/local_descriptors.hpp"

#include <cmath>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace mdescriptor::detail {

inline void validate_options(const LocalDescriptorOptions& options) {
    if (!std::isfinite(options.cutoff) || options.cutoff <= 0.0
        || !std::isfinite(options.density_width) || options.density_width <= 0.0
        || options.max_radial < 0 || options.max_angular < 0
        || !std::isfinite(options.k_cutoff) || options.k_cutoff <= 0.0
        || options.exponent <= 0
        || !std::isfinite(options.radial_radius) || options.radial_radius <= 0.0) {
        throw std::invalid_argument("invalid local descriptor descriptor parameters");
    }
}

} // namespace mdescriptor::detail
