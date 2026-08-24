#pragma once

#include "descriptor.hpp"

#include <cstdint>
#include <vector>

namespace mdescriptor {

enum class LocalDescriptorKind : std::int32_t {
    SphericalExpansion = 0,
    SphericalExpansionByPair = 1,
    SoapRadialSpectrum = 2,
    SoapPowerSpectrum = 3,
    LodeSphericalExpansion = 4,
};

struct LocalDescriptorOptions {
    std::vector<std::int32_t> species;
    double cutoff = 6.0;
    double density_width = 0.3;
    int max_radial = 6;
    int max_angular = 4;
    double k_cutoff = 2.5;
    int exponent = 1;
    double radial_radius = 6.0;
    int num_threads = 0;
};

struct DescriptorPairTable {
    // columns: first, second, cell_shift_a, cell_shift_b, cell_shift_c,
    //          displacement_x, displacement_y, displacement_z, distance
    std::vector<double> values;
    std::vector<std::int64_t> offsets;
};

std::int64_t local_descriptor_feature_count(const LocalDescriptorOptions& options, LocalDescriptorKind kind);

void compute_atomic_composition(
    const StructureBatchView& batch,
    const std::vector<std::int32_t>& species,
    bool per_system,
    double* output,
    const std::shared_ptr<ComputeControl>& control);

void compute_sorted_distances(
    const StructureBatchView& batch,
    const LocalDescriptorOptions& options,
    int max_neighbors,
    bool separate_neighbor_types,
    double* output,
    const std::shared_ptr<ComputeControl>& control);

DescriptorPairTable compute_neighbor_list(
    const StructureBatchView& batch,
    double cutoff,
    bool full_neighbor_list,
    bool self_pairs,
    const std::shared_ptr<ComputeControl>& control);

void compute_spherical_expansion(
    const StructureBatchView& batch,
    const LocalDescriptorOptions& options,
    LocalDescriptorKind kind,
    double* output,
    const std::shared_ptr<ComputeControl>& control);

DescriptorPairTable compute_spherical_expansion_by_pair(
    const StructureBatchView& batch,
    const LocalDescriptorOptions& options,
    const std::shared_ptr<ComputeControl>& control);

} // namespace mdescriptor
