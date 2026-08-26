#pragma once

#include "mdescriptor/descriptor.hpp"
#include "mdescriptor/detail/control.hpp"

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

namespace mdescriptor {

// The Python checkpoint reader owns the .pt format.  It converts the
// validated DPA4C descriptor into this compact, backend-neutral parameter
// block before inference.  The numerical work after that boundary is native
// C++: neighbor construction, edge MLPs, moment reductions, and readout.
struct Dpa4cOptions {
    double rcut = 6.0;
    int ntypes = 0;
    int channels = 0;
    int lmax = 0;
    int n_radial = 0;
    int radial_modes = 0;
    int radial_hidden = 0;
    int pair_hidden = 0;
    int num_threads = 1;
    bool calibrate = true;

    std::vector<float> type_embedding;
    std::vector<float> radial_freqs;
    std::vector<float> radial_w0;
    std::vector<float> radial_w1;
    std::vector<float> radial_mode_w;
    std::vector<float> pair_w0;
    std::vector<float> pair_w1;

    std::vector<int> degree_channels;
    std::vector<int> bispectrum_ranks;
    std::vector<float> readout_alignment;
    std::vector<float> readout_projections;
    std::vector<std::int64_t> readout_alignment_offsets;
    std::vector<std::int64_t> readout_projection_offsets;

    std::vector<float> bispectrum_coupling;
    std::vector<std::int64_t> coupling_offsets;
    std::vector<int> degree_triples;
    std::vector<std::int64_t> probe_offsets;
    std::vector<std::int64_t> probe_index;
    std::vector<float> probe_scale;

    std::vector<float> output_mean;
    std::vector<float> output_stddev;
};

class Dpa4cCalculator {
public:
    explicit Dpa4cCalculator(Dpa4cOptions options);

    std::int64_t feature_count() const noexcept;
    void close() noexcept;
    bool closed() const noexcept;

    void compute(
        const StructureBatchView& batch,
        const std::int32_t* type_indices,
        double* output,
        const std::shared_ptr<ComputeControl>& control
    ) const;

private:
    Dpa4cOptions options_;
    std::int64_t feature_count_ = 0;
    std::int64_t moment_count_ = 0;
    std::vector<int> degree_offsets_;
    std::vector<std::int64_t> gram_offsets_;
    std::vector<std::int64_t> gram_index_;
    std::vector<float> gram_scale_;
    std::vector<float> pair_scale_;
    std::vector<float> pair_shift_;
    std::vector<float> pair_mixing_;
    mutable std::mutex compute_mutex_;
    std::atomic<bool> closed_{false};
};

} // namespace mdescriptor
