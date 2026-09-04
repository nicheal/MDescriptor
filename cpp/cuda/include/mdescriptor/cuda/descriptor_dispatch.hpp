#pragma once

#include <array>
#include <string_view>

namespace mdescriptor::cuda {

// Backend feature discovery and compute dispatch must agree on this set.
inline constexpr std::array<std::string_view, 21> kExtendedDescriptorNames = {
    "AtomicComposition", "SortedDistances", "SphericalExpansionByPair", "SOAP",
    "SOAPTurbo", "ACSF", "ACE", "LodeSphericalExpansion", "CoulombMatrix",
    "SineMatrix", "EwaldSumMatrix", "MBTR", "LMBTR", "ValleOganov", "EAD",
    "SO3", "SO4", "SNAP", "LBispectrum", "MTP", "C00PSMLFF",
};

inline bool is_extended_descriptor(std::string_view name) noexcept {
    for (const auto candidate : kExtendedDescriptorNames) {
        if (candidate == name) {
            return true;
        }
    }
    return false;
}

} // namespace mdescriptor::cuda
