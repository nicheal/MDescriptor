#pragma once

#include "descriptor.hpp"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace mdescriptor {

struct NepOptions {
    std::string model_path;
    // Content identity supplied by the model resolver.  The native model
    // cache must not identify mutable files by path alone.  Without
    // model_data this remains a caller-trusted legacy cache identity.
    std::string model_digest;
    int num_threads = 0;
    // Immutable bytes supplied by the model resolver.  When present, native
    // parsing must not reopen model_path.
    std::optional<std::string> model_data;
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

// Additional immutable data needed by an energy/force predictor. Kept apart
// from descriptor_parameters() so descriptor-only users do not acquire ANN
// or ZBL state as part of their model contract.
struct NepPredictionParameters : NepDescriptorParameters {
    int hidden_neurons1 = 0;
    int hidden_neurons2 = 0;
    bool zbl_enabled = false;
    double zbl_inner = 0.0;
    double zbl_outer = 0.0;
    double neighbor_cutoff() const noexcept {
        return std::max({radial_cutoff_max, angular_cutoff_max, zbl_outer});
    }
    std::vector<double> ann_parameters;
};

struct NepModel;

class NepCalculator : public detail::CalculatorBase {
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
    NepPredictionParameters prediction_parameters() const;

    void compute(
        const StructureBatchView& batch,
        double* output,
        const std::shared_ptr<ComputeControl>& control
    ) const;

private:
    std::shared_ptr<const NepModel> model_;
    int num_threads_ = 0;
};

class NepPredictor : public detail::CalculatorBase {
public:
    explicit NepPredictor(NepOptions options);

    const std::vector<std::int32_t>& species() const noexcept;
    const std::string& model_path() const noexcept;

    void predict(
        const StructureBatchView& batch,
        double* energy,
        double* atom_energy,
        double* forces,
        const std::shared_ptr<ComputeControl>& control
    ) const;

private:
    std::shared_ptr<const NepModel> model_;
    int num_threads_ = 0;
};

} // namespace mdescriptor
