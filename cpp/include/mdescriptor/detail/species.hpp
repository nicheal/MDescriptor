#pragma once

#include "batch.hpp"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace mdescriptor::detail {

using SpeciesList = std::vector<std::int32_t>;
using SpeciesMap = std::unordered_map<std::int32_t, std::int32_t>;
using TypeMap = std::unordered_map<std::int32_t, std::size_t>;

inline void validate_species(const SpeciesList& species) {
    if (species.empty()) {
        throw std::invalid_argument("species must not be empty");
    }
    for (std::size_t index = 0; index < species.size(); ++index) {
        if (species[index] <= 0) {
            throw std::invalid_argument("species must contain positive atomic numbers");
        }
        for (std::size_t previous = 0; previous < index; ++previous) {
            if (species[previous] == species[index]) {
                throw std::invalid_argument("species must contain unique atomic numbers");
            }
        }
    }
}

inline SpeciesMap species_map(const SpeciesList& species) {
    validate_species(species);
    SpeciesMap result;
    for (std::size_t index = 0; index < species.size(); ++index) {
        result.emplace(species[index], static_cast<std::int32_t>(index));
    }
    return result;
}

inline TypeMap make_type_map(const SpeciesList& species) {
    validate_species(species);
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

inline TypeMap type_map(const SpeciesList& species) { return make_type_map(species); }

inline void validate_species(
    const StructureBatchView& batch,
    const SpeciesList& species) {
    const auto mapping = species_map(species);
    for (std::int64_t atom = 0; atom < batch.atoms; ++atom) {
        if (batch.numbers[atom] <= 0) {
            throw std::invalid_argument("atomic numbers must be positive");
        }
        if (mapping.find(batch.numbers[atom]) == mapping.end()) {
            throw std::invalid_argument("batch contains an atomic number outside calculator species");
        }
    }
}

} // namespace mdescriptor::detail
