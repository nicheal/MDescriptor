#pragma once

#include "mdescriptor/cuda/batch.hpp"
#include "mdescriptor/cuda/context.hpp"
#include "mdescriptor/cuda/neighbor_graph.hpp"

#include <pybind11/pytypes.h>

#include <cstdint>
#include <memory>
#include <vector>

namespace mdescriptor::cuda {

// Private CUDA implementation of the graph-native, spin-free and
// uncompressed DPA4C payload.  Python remains responsible for reading and
// validating the checkpoint; this class owns only the typed tensors required
// by the CUDA kernel.  The public backend deliberately exposes no CUDA
// pointers or model-specific ABI.
class DeviceDpa4cModel {
public:
    DeviceDpa4cModel(CudaExecutionContext& context, pybind11::dict payload);
    DeviceDpa4cModel(const DeviceDpa4cModel&) = delete;
    DeviceDpa4cModel& operator=(const DeviceDpa4cModel&) = delete;
    ~DeviceDpa4cModel() noexcept;

    std::int64_t feature_count() const noexcept { return feature_count_; }
    double cutoff() const noexcept { return rcut_; }

    // Returns the checkpoint type index for an atomic number, or -1 when the
    // payload did not provide a type_numbers entry or the number is absent.
    int type_index_for_number(std::int32_t number) const noexcept;

    // type_indices is intentionally host-owned.  The implementation uploads
    // it on the model's stream and returns host float64 values with shape
    // (batch.atoms, feature_count()).
    std::vector<double> compute(
        CudaExecutionContext& context,
        const DeviceBatch& batch,
        const DeviceNeighborGraph& graph,
        const std::vector<std::int32_t>& type_indices) const;

public:
    struct DeviceArray;
    struct Layout;

private:
    void release() noexcept;

    int ntypes_ = 0;
    int channels_ = 0;
    int lmax_ = 0;
    int n_radial_ = 0;
    int radial_modes_ = 0;
    int radial_hidden_ = 0;
    int pair_hidden_ = 0;
    int device_ = 0;
    double rcut_ = 0.0;
    bool calibrate_ = true;
    std::int64_t feature_count_ = 0;
    std::int64_t moment_count_ = 0;
    std::int64_t triple_count_ = 0;
    std::unique_ptr<Layout> layout_;
    std::vector<int> degree_channels_;
    std::vector<int> bispectrum_ranks_;
    std::vector<std::int64_t> degree_offsets_;
    std::vector<std::int64_t> gram_offsets_;
    std::vector<std::int32_t> gram_index_;
    std::vector<float> gram_scale_;
    std::vector<std::int32_t> type_numbers_;
    std::vector<std::int32_t> host_type_lookup_;

    // Every tensor is contiguous and device-resident.  Keeping the ownership
    // in this object makes model construction exception safe and avoids a
    // global cache whose lifetime would outlive the CUDA context.
    std::unique_ptr<DeviceArray> type_embedding_;
    std::unique_ptr<DeviceArray> degree_channels_device_;
    std::unique_ptr<DeviceArray> bispectrum_ranks_device_;
    std::unique_ptr<DeviceArray> radial_freqs_;
    std::unique_ptr<DeviceArray> radial_w0_;
    std::unique_ptr<DeviceArray> radial_w1_;
    std::unique_ptr<DeviceArray> radial_mode_w_;
    std::unique_ptr<DeviceArray> pair_scale_;
    std::unique_ptr<DeviceArray> pair_shift_;
    std::unique_ptr<DeviceArray> pair_mixing_;
    std::unique_ptr<DeviceArray> alignment_;
    std::unique_ptr<DeviceArray> alignment_offsets_;
    std::unique_ptr<DeviceArray> projections_;
    std::unique_ptr<DeviceArray> projection_offsets_;
    std::unique_ptr<DeviceArray> coupling_;
    std::unique_ptr<DeviceArray> coupling_offsets_;
    std::unique_ptr<DeviceArray> degree_triples_;
    std::unique_ptr<DeviceArray> probe_offsets_;
    std::unique_ptr<DeviceArray> probe_index_;
    std::unique_ptr<DeviceArray> probe_scale_;
    std::unique_ptr<DeviceArray> output_mean_;
    std::unique_ptr<DeviceArray> output_stddev_;
};

} // namespace mdescriptor::cuda
