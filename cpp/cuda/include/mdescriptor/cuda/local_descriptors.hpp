#pragma once

#include "mdescriptor/cuda/batch.hpp"
#include "mdescriptor/cuda/context.hpp"
#include "mdescriptor/cuda/neighbor_graph.hpp"

#include <cstdint>
#include <vector>

namespace mdescriptor::cuda {

// ``kind`` uses the stable LocalDescriptorKind values for the three CUDA
// descriptors: 0 = SphericalExpansion, 2 = SoapRadialSpectrum, and 3 =
// SoapPowerSpectrum.
void compute_local_descriptors_into(
    CudaExecutionContext& context,
    const DeviceBatch& batch,
    const DeviceNeighborGraph& graph,
    const std::vector<std::int32_t>& species,
    double cutoff,
    double density_width,
    int max_radial,
    int max_angular,
    std::int32_t kind,
    double* output);

} // namespace mdescriptor::cuda
