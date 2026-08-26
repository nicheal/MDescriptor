#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include "mdescriptor/detail/batch.hpp"
#include "mdescriptor/detail/control.hpp"
#include "mdescriptor/detail/species.hpp"

namespace mdescriptor {

using detail::StructureBatchView;

struct SoapOptions {
    std::vector<std::int32_t> species;
    double r_cut = 6.0;
    int n_max = 8;
    int l_max = 6;
    double sigma = 1.0;
    int radial_basis = 0; // 0: GTO, 1: polynomial
    std::vector<double> alphas;
    std::vector<double> betas;
    std::vector<double> radial_grid;
    std::vector<double> radial_weights;
    std::vector<double> radial_values;
    int weighting_function = 0; // 0: none, 1: poly, 2: pow, 3: exp
    bool weighting_has_w0 = false;
    bool weighting_has_function = false;
    double weighting_r0 = 1.0;
    double weighting_c = 1.0;
    double weighting_d = 0.0;
    double weighting_m = 1.0;
    double weighting_threshold = 1e-2;
    double weighting_w0 = 1.0;
    std::vector<double> species_weights;
    int compression = 0; // 0: off, 1: mu2, 2: mu1nu1, 3: crossover
    bool inner_average = true;
    bool outer_average = false;
    int num_threads = 0;
};

struct SoapTurboOptions {
    std::vector<std::int32_t> species;
    std::vector<std::int32_t> alpha_max;
    std::vector<std::int32_t> central_species;
    std::vector<double> atom_sigma_r;
    std::vector<double> atom_sigma_r_scaling;
    std::vector<double> atom_sigma_t;
    std::vector<double> atom_sigma_t_scaling;
    std::vector<double> amplitude_scaling;
    std::vector<double> central_weight;
    int l_max = 6;
    double rcut_hard = 5.0;
    double rcut_soft = 5.0;
    double nf = 1.0;
    int radial_enhancement = 0;
    int basis = 0; // 0: poly3, 1: poly3gauss
    std::string compression; // "", "trivial", or "0_0" through "2_2"
    int num_threads = 0;
};

struct SoapTurboPrepared;

struct AcsfOptions {
    std::vector<std::int32_t> species;
    double r_cut = 6.0;
    std::vector<double> g2_params; // row-major (eta, Rs)
    std::vector<double> g3_params; // kappa
    std::vector<double> g4_params; // row-major (eta, zeta, lambda)
    std::vector<double> g5_params; // row-major (eta, zeta, lambda)
    std::int64_t n_g2 = 0;
    std::int64_t n_g3 = 0;
    std::int64_t n_g4 = 0;
    std::int64_t n_g5 = 0;
    int num_threads = 0;
};

struct C00PSMlffOptions {
    std::vector<std::int32_t> species;
    double r_cut = 6.0;
    int n_radial = 8;
    int l_max = 4;
    int cutoff_function = 0; // 0: BP, 1: MO, 2: RJ, 3: WMC
    // Gaussian width sigma_atom in the VASP MLFF theory (Angstrom).  A
    // positive value enables the Gaussian atomic distribution; zero keeps
    // the unsmoothed delta-distribution limit.
    double radial_sigma = 0.5;
    bool include_radial = true;
    bool include_angular = true;
    bool normalize_radial = false;
    bool normalize_angular = false;
    bool super_vector = false;
    double radial_weight = 1.0;
    double angular_weight = 1.0;
    // VASP 6.6.0 MLFF LSIC semantics: subtract self terms only from PS
    // channels whose neighbour species is the centre species.
    bool exclude_self_interaction = true;
    int num_threads = 0;
};

std::int64_t soap_feature_count(const SoapOptions& options);
std::int64_t soap_turbo_feature_count(const SoapTurboOptions& options);
std::int64_t acsf_feature_count(const AcsfOptions& options);
std::int64_t c00ps_mlff_feature_count(const C00PSMlffOptions& options);

void compute_soap(
    const StructureBatchView& batch,
    const SoapOptions& options,
    double* output,
    const std::shared_ptr<ComputeControl>& control
);

void compute_soap_turbo(
    const StructureBatchView& batch,
    const SoapTurboOptions& options,
    double* output,
    const std::shared_ptr<ComputeControl>& control
);

void compute_acsf(
    const StructureBatchView& batch,
    const AcsfOptions& options,
    double* output,
    const std::shared_ptr<ComputeControl>& control
);

class C00PSMlffCalculator {
public:
    explicit C00PSMlffCalculator(C00PSMlffOptions options);

    std::int64_t feature_count() const noexcept;
    const std::vector<std::int32_t>& species() const noexcept;
    const std::vector<std::int32_t>& radial_counts() const noexcept;
    void close() noexcept;
    bool closed() const noexcept;

    void compute(
        const StructureBatchView& batch,
        double* output,
        const std::shared_ptr<ComputeControl>& control
    ) const;

private:
    C00PSMlffOptions options_;
    mutable std::mutex compute_mutex_;
    std::atomic<bool> closed_{false};
    mutable std::vector<std::vector<double>> zeros_;
    mutable std::vector<std::vector<double>> norms_;
    mutable std::vector<std::vector<double>> radial_values_;
    mutable std::vector<std::int32_t> radial_counts_;
    mutable bool basis_ready_ = false;
};

void compute_coulomb_matrix(
    const StructureBatchView& batch,
    std::int64_t n_atoms_max,
    const std::string& permutation,
    double exponent,
    double* output,
    const std::shared_ptr<ComputeControl>& control
);

class SoapCalculator {
public:
    SoapCalculator(SoapOptions options);

    std::int64_t feature_count() const noexcept;
    const std::vector<std::int32_t>& species() const noexcept;
    void close() noexcept;
    bool closed() const noexcept;

    void compute(
        const StructureBatchView& batch,
        double* output,
        const std::shared_ptr<ComputeControl>& control
    ) const;

private:
    SoapOptions options_;
    mutable std::mutex compute_mutex_;
    std::atomic<bool> closed_{false};
};

class SoapTurboCalculator {
public:
    SoapTurboCalculator(SoapTurboOptions options);

    std::int64_t feature_count() const noexcept;
    const std::vector<std::int32_t>& species() const noexcept;
    void close() noexcept;
    bool closed() const noexcept;

    void compute(
        const StructureBatchView& batch,
        double* output,
        const std::shared_ptr<ComputeControl>& control
    ) const;

private:
    SoapTurboOptions options_;
    mutable std::mutex compute_mutex_;
    mutable std::shared_ptr<SoapTurboPrepared> prepared_;
    std::atomic<bool> closed_{false};
};

class AcsfCalculator {
public:
    AcsfCalculator(AcsfOptions options);

    std::int64_t feature_count() const noexcept;
    const std::vector<std::int32_t>& species() const noexcept;
    void close() noexcept;
    bool closed() const noexcept;

    void compute(
        const StructureBatchView& batch,
        double* output,
        const std::shared_ptr<ComputeControl>& control
    ) const;

private:
    AcsfOptions options_;
    mutable std::mutex compute_mutex_;
    std::atomic<bool> closed_{false};
};

} // namespace mdescriptor
