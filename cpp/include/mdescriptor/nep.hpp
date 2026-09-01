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

// Descriptor-only model data shared with the optional CUDA plugin.  The ANN
// weights are intentionally not part of this snapshot: this project exposes
// NEP descriptors, while the model-backed CPU calculator keeps the complete
// model parser and prediction path private to nep.cpp.
struct NepDescriptorParameters {
    int version = 0;
    int num_types = 0;
    int n_max_radial = 0;
    int n_max_angular = 0;
    int basis_size_radial = 0;
    int basis_size_angular = 0;
    int l_max = 0;
    int num_l = 0;
    bool has_q_222 = false;
    bool has_q_1111 = false;
    bool has_q_112 = false;
    bool has_q_123 = false;
    bool has_q_233 = false;
    bool has_q_134 = false;
    int dimension = 0;
    double radial_cutoff_max = 0.0;
    double angular_cutoff_max = 0.0;
    std::vector<std::int32_t> species;
    std::vector<double> radial_cutoff_pair;
    std::vector<double> angular_cutoff_pair;
    std::vector<double> radial_pair_coefficients;
    std::vector<double> angular_pair_coefficients;
    std::vector<double> scalers;
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
    NepDescriptorParameters descriptor_parameters() const;
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
