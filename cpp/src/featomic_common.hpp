#pragma once

#include "mdescriptor/featomic.hpp"

#include <cmath>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace mdescriptor::detail {

using TypeMap = std::unordered_map<std::int32_t, std::size_t>;

inline void validate_species(const StructureBatchView& batch, const std::vector<std::int32_t>& species) {
    if (species.empty()) {
        throw std::invalid_argument("species must not be empty");
    }
    TypeMap mapping;
    for (const auto type : species) {
        if (type <= 0 || !mapping.emplace(type, mapping.size()).second) {
            throw std::invalid_argument("species must contain unique positive atomic numbers");
        }
    }
    for (std::int64_t atom = 0; atom < batch.atoms; ++atom) {
        if (!mapping.count(batch.numbers[atom])) {
            throw std::invalid_argument("batch contains an atomic number outside calculator species");
        }
    }
}

inline TypeMap make_type_map(const std::vector<std::int32_t>& species) {
    TypeMap result;
    for (std::size_t index = 0; index < species.size(); ++index) {
        result.emplace(species[index], index);
    }
    return result;
}

inline std::vector<std::int32_t> make_atom_types(
    const StructureBatchView& batch,
    const TypeMap& mapping) {
    std::vector<std::int32_t> result(static_cast<std::size_t>(batch.atoms));
    for (std::int64_t atom = 0; atom < batch.atoms; ++atom) {
        const auto found = mapping.find(batch.numbers[atom]);
        if (found == mapping.end()) {
            throw std::invalid_argument("batch contains an atomic number outside calculator species");
        }
        result[static_cast<std::size_t>(atom)] = static_cast<std::int32_t>(found->second);
    }
    return result;
}

inline void validate_options(const FeatomicOptions& options) {
    if (!std::isfinite(options.cutoff) || options.cutoff <= 0.0
        || !std::isfinite(options.density_width) || options.density_width <= 0.0
        || options.max_radial < 0 || options.max_angular < 0
        || !std::isfinite(options.k_cutoff) || options.k_cutoff <= 0.0
        || options.exponent <= 0
        || !std::isfinite(options.radial_radius) || options.radial_radius <= 0.0) {
        throw std::invalid_argument("invalid Featomic descriptor parameters");
    }
}

inline void check_cancelled(const std::shared_ptr<ComputeControl>& control) {
    if (control && control->cancelled()) {
        throw CancelledError();
    }
}

inline void mark_completed(const std::shared_ptr<ComputeControl>& control) {
    if (control) {
        control->mark_completed();
    }
}

} // namespace mdescriptor::detail
