#pragma once

#include "descriptor.hpp"

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace mdescriptor {

struct NepOptions {
    std::string model_path;
    // Content identity supplied by the model resolver.  The native model
    // cache must not identify mutable files by path alone.
    std::string model_digest;
    int num_threads = 0;
};

struct NepModel;

class NepCalculator {
public:
    explicit NepCalculator(NepOptions options);

    std::int64_t feature_count() const noexcept;
    const std::vector<std::int32_t>& species() const noexcept;
    const std::string& model_path() const noexcept;
    double radial_cutoff() const noexcept;
    double angular_cutoff() const noexcept;
    int n_max_radial() const noexcept;
    int n_max_angular() const noexcept;
    int l_max() const noexcept;
    bool closed() const noexcept;
    void close() noexcept;

    void compute(
        const StructureBatchView& batch,
        double* output,
        const std::shared_ptr<ComputeControl>& control
    ) const;

private:
    std::shared_ptr<const NepModel> model_;
    int num_threads_ = 0;
    mutable std::mutex compute_mutex_;
    std::atomic<bool> closed_{false};
};

} // namespace mdescriptor
